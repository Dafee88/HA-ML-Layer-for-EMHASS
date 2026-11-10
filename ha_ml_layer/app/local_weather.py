"""Local weather outlook (ha-ml-layer): tomorrow's cloudiness from the own station.

Purpose: a forecast that works without internet (power cut, API down). It
only feeds the station fallback - while Solcast has data, nothing changes.

Target: the daily clear-sky index (CSI) of a day = measured irradiance /
clear-sky irradiance, summed over the daylight hours (sun >= 10 deg).

Every evening (influxdb_dayahead_snapshot_hour, default 20:00) the model
predicts tomorrow's daily CSI from what the station measured:
  - csi:    today's daily CSI (persistence)
  - dp3:    pressure change over the last 3 h (hPa)        [pressure sensor]
  - dp12:   pressure change over the last 12 h (hPa)       [pressure sensor]
  - p_anom: pressure now minus the mean of the last 7 days [pressure sensor]
  - rh:     relative humidity now (%)                      [humidity sensor]
Model: ridge regression on standardised features (numpy, a few dozen samples).

Honest gate: every prediction is stored with the climatology prediction
(mean daily CSI of the last 30 evaluated days) and the persistence prediction.
When the day is over, the errors of all three are recorded. The model is only
used while it beats climatology on the same days (at least MIN_EVAL days).
Then the station fallback scales its typical cloudiness per hour by
predicted / climatology CSI. Otherwise the fallback stays as before.

History: on the first start (and when the feature set changes) the samples
are rebuilt from InfluxDB, with the same evening logic in time order.
"""
import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("local_weather")

STATE_PATH = Path("/data/local_weather_v1.json")
ENTITY_OUTPUT = "sensor.pv_ml_local_weather"

MIN_ELEVATION = 10.0           # daylight slots that count for the daily CSI
MIN_COVERAGE = 0.8             # share of daylight samples with an irradiance reading
CSI_CLIP = (0.02, 1.3)
MIN_TRAIN = 10                 # samples before the regression predicts
MIN_EVAL = 10                  # evaluated days before the gate may open
EVAL_WINDOW = 30
CLIM_WINDOW = 30
MAX_SAMPLES = 180
RIDGE_LAMBDA = 10.0            # strong shrinkage: few samples, weak signals (real backtest 2026)
FACTOR_CLIP = (0.2, 2.5)
PRESSURE_TOL_S = 45 * 60       # a pressure reading this close counts as "3 h ago"
PRESSURE_KEEP_S = 8 * 86400
HISTORY_DAYS = 120

# 3-hour pressure tendency, terms as used by the UK Met Office (hPa per 3 h)
TENDENCY_TERMS = ((0.1, "steady"), (1.6, "slowly"), (3.6, ""), (6.1, "quickly"), (math.inf, "very rapidly"))

_PRESSURE_TO_HPA = {"hpa": 1.0, "mbar": 1.0, "mb": 1.0, "pa": 0.01, "kpa": 10.0,
                    "inhg": 33.8639, "mmhg": 1.33322, "psi": 68.9476}


def to_hpa(value, unit):
    """Pressure in hPa from a value and its unit_of_measurement (None if unknown)."""
    if value is None:
        return None
    f = _PRESSURE_TO_HPA.get((unit or "hPa").replace(" ", "").lower())
    if f is None:
        return None
    hpa = value * f
    return hpa if 300.0 <= hpa <= 1100.0 else None


def tendency_text(dp3):
    if dp3 is None:
        return None
    a = abs(dp3)
    for limit, word in TENDENCY_TERMS:
        if a < limit:
            if word == "steady":
                return "steady"
            verb = "rising" if dp3 > 0 else "falling"
            return f"{verb} {word}".strip()
    return None


def outlook_text(csi):
    if csi is None:
        return "unknown"
    if csi >= 0.7:
        return "sunny"
    if csi >= 0.4:
        return "partly cloudy"
    return "cloudy"


class LocalWeatherModel:
    def __init__(self, use_pressure, use_humidity, tz_name, evening_hour, state_path=STATE_PATH):
        self.features = ["csi"] + (["dp3", "dp12", "p_anom"] if use_pressure else []) + (["rh"] if use_humidity else [])
        self.tz = ZoneInfo(tz_name)
        self.evening_hour = evening_hour
        self.state_path = state_path
        self.days = {}             # "YYYY-MM-DD" -> [sum irradiance, sum clear-sky, n with reading, n daylight]
        self.pressure = []         # [[epoch, hPa], ...]
        self.samples = []          # [{"date", "x": {...}, "y"}], date = predicted day
        self.pending = None        # {"date", "x", "model", "clim", "persist"}
        self.evals = []            # [{"date", "y", "model", "clim", "persist"}]
        self.last_evening = None
        self.history_key = None
        self._coef = None
        if state_path.exists():
            try:
                s = json.loads(state_path.read_text())
                if s.get("features") == self.features:
                    for k in ("days", "pressure", "samples", "pending", "evals", "last_evening", "history_key"):
                        setattr(self, k, s.get(k, getattr(self, k)))
                else:
                    log.info("Local weather: feature set changed (%s -> %s), starting fresh",
                             s.get("features"), self.features)
            except Exception:
                log.exception("Could not load %s, starting fresh", state_path)
        self.refit()

    # --- persistence ---------------------------------------------------------------
    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "features": self.features, "days": self.days, "pressure": self.pressure,
            "samples": self.samples, "pending": self.pending, "evals": self.evals,
            "last_evening": self.last_evening, "history_key": self.history_key}))
        os.replace(tmp, self.state_path)

    # --- measurements ---------------------------------------------------------------
    def add_irradiance(self, ts, irradiance, clear_sky_ghi, elevation):
        """One cycle of the station (irradiance may be None = no reading)."""
        if elevation is None or elevation < MIN_ELEVATION:
            return
        d = self.days.setdefault(ts.astimezone(self.tz).date().isoformat(), [0.0, 0.0, 0, 0])
        d[3] += 1
        if irradiance is not None and irradiance >= 0 and clear_sky_ghi > 0:
            d[0] += irradiance
            d[1] += clear_sky_ghi
            d[2] += 1

    def add_pressure(self, ts, hpa):
        if hpa is None:
            return
        t = int(ts.timestamp())
        if self.pressure and t - self.pressure[-1][0] < 600:
            return                 # one reading per 10 min is plenty
        self.pressure.append([t, round(hpa, 2)])
        cutoff = t - PRESSURE_KEEP_S
        while self.pressure and self.pressure[0][0] < cutoff:
            self.pressure.pop(0)

    def daily_csi(self, date):
        d = self.days.get(date.isoformat())
        if not d or d[3] == 0 or d[1] <= 0 or d[2] < MIN_COVERAGE * d[3]:
            return None
        return float(min(max(d[0] / d[1], CSI_CLIP[0]), CSI_CLIP[1]))

    def _pressure_at(self, t):
        best = None
        for pt, v in self.pressure:
            if abs(pt - t) <= PRESSURE_TOL_S and (best is None or abs(pt - t) < abs(best[0] - t)):
                best = (pt, v)
        return None if best is None else best[1]

    def features_now(self, now, rh=None):
        """Feature dict for 'tomorrow' at time now, or None if a feature is missing."""
        x = {"csi": self.daily_csi(now.astimezone(self.tz).date())}
        if "dp3" in self.features:
            t = int(now.timestamp())
            p0, p3, p12 = self._pressure_at(t), self._pressure_at(t - 3 * 3600), self._pressure_at(t - 12 * 3600)
            week = [v for pt, v in self.pressure if t - 7 * 86400 <= pt <= t]
            x["dp3"] = None if p0 is None or p3 is None else p0 - p3
            x["dp12"] = None if p0 is None or p12 is None else p0 - p12
            span = (t - self.pressure[0][0]) if self.pressure else 0
            x["p_anom"] = p0 - sum(week) / len(week) if p0 is not None and week and span >= 2 * 86400 else None
        if "rh" in self.features:
            x["rh"] = rh
        return None if any(x.get(f) is None for f in self.features) else x

    # --- model ---------------------------------------------------------------------
    def refit(self):
        self._coef = None
        if len(self.samples) < MIN_TRAIN:
            return
        X = np.array([[s["x"][f] for f in self.features] for s in self.samples], dtype=float)
        y = np.array([s["y"] for s in self.samples], dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        A = Z.T @ Z + RIDGE_LAMBDA * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y - y.mean()))
        self._coef = (mu, sd, w, y.mean())

    def predict(self, x):
        if self._coef is None or x is None:
            return None
        mu, sd, w, b = self._coef
        z = (np.array([x[f] for f in self.features], dtype=float) - mu) / sd
        return float(min(max(b + z @ w, CSI_CLIP[0]), CSI_CLIP[1]))

    def climatology(self):
        ys = [e["y"] for e in self.evals[-CLIM_WINDOW:]]
        return float(sum(ys) / len(ys)) if ys else None

    # --- gate ----------------------------------------------------------------------
    def scores(self):
        """MAE of daily CSI over the last evaluated days; each method is compared
        with climatology on exactly the days on which it had a prediction."""
        ev = [e for e in self.evals[-EVAL_WINDOW:] if e["clim"] is not None]
        out = {"n": len(ev), "clim": _mae(ev, "clim")}
        for k in ("model", "persist"):
            sub = [e for e in ev if e[k] is not None]
            out[k], out[k + "_n"], out[k + "_vs_clim"] = _mae(sub, k), len(sub), _mae(sub, "clim")
        return out

    def best(self):
        """'model' or 'persist' if it beats climatology on the same days, else None."""
        s = self.scores()
        cand = [(s[k], k) for k in ("model", "persist")
                if s[k + "_n"] >= MIN_EVAL and s[k] < s[k + "_vs_clim"]]
        return min(cand)[1] if cand else None

    @property
    def active(self):
        return self.best() is not None

    def day_factor(self, date):
        """Scale for the station's typical cloudiness on a date (1.0 = unchanged)."""
        p, k = self.pending, self.best()
        if k is None or not p or p["date"] != date.isoformat() or not p.get(k) or not p.get("clim"):
            return 1.0
        return min(max(p[k] / p["clim"], FACTOR_CLIP[0]), FACTOR_CLIP[1])

    # --- daily step ----------------------------------------------------------------
    def evening(self, now, rh=None):
        """Once per local day at/after the evening hour: close yesterday's prediction
        (its target is today's CSI), learn it, predict tomorrow. True if it ran."""
        local = now.astimezone(self.tz)
        today = local.date()
        if local.hour < self.evening_hour or self.last_evening == today.isoformat():
            return False
        self.last_evening = today.isoformat()
        y = self.daily_csi(today)
        p = self.pending
        if p and p["date"] == today.isoformat() and y is not None:
            self.evals.append({"date": p["date"], "y": round(y, 4), "model": p["model"],
                               "clim": p["clim"], "persist": p["persist"]})
            self.evals = self.evals[-EVAL_WINDOW * 2:]
            if p.get("x"):
                self.samples.append({"date": p["date"], "x": p["x"], "y": round(y, 4)})
                self.samples = self.samples[-MAX_SAMPLES:]
                self.refit()
        x = self.features_now(now, rh)
        csi_today = self.daily_csi(today)
        clim = self.climatology()
        tomorrow = (today + timedelta(days=1)).isoformat()
        # a missing pressure/humidity reading only drops the regression for that day
        self.pending = None if csi_today is None and clim is None else {
            "date": tomorrow, "x": None if x is None else {k: round(float(v), 4) for k, v in x.items()},
            "model": self.predict(x), "clim": clim, "persist": csi_today}
        keep = (today - timedelta(days=10)).isoformat()
        self.days = {k: v for k, v in self.days.items() if k >= keep}
        return True

    # --- history from InfluxDB --------------------------------------------------------
    def backfill(self, influx, irradiance_entity, pressure_entity, humidity_entity, now=None):
        """Rebuild days, pressure and samples from InfluxDB once per feature set (read-only)."""
        key = "|".join([irradiance_entity or "", pressure_entity or "", humidity_entity or ""] + self.features)
        if self.history_key == key:
            return
        from history import _buckets
        from pv_model import solar_position
        end = now or datetime.now(timezone.utc)
        start = (end - timedelta(days=HISTORY_DAYS)).replace(minute=0, second=0, microsecond=0)
        irr = _buckets(influx, irradiance_entity, start, end, max_carry=1800)
        if not irr:
            log.info("Local weather history: no irradiance history in InfluxDB, learning live only")
            self.history_key = key
            self.save()
            return
        pres_unit = None
        pres = _buckets(influx, pressure_entity, start, end, max_carry=10800) if pressure_entity else {}
        hum = _buckets(influx, humidity_entity, start, end, max_carry=10800) if humidity_entity else {}
        if pressure_entity:
            from history import _meta
            meta = _meta(influx, pressure_entity, lambda f: f == "value")
            pres_unit = meta[0] if meta else None       # HA stores the unit as measurement name
        t0 = min(irr)
        epochs = list(range(t0, int(end.timestamp()), 300))
        ghi, elev, _azi = solar_position([datetime.fromtimestamp(t, timezone.utc) for t in epochs])
        self.days, self.pressure, self.samples, self.pending, self.evals = {}, [], [], None, []
        self.last_evening, self._coef = None, None
        for t, g, e in zip(epochs, ghi, elev):
            ts = datetime.fromtimestamp(t, timezone.utc)
            self.add_irradiance(ts, irr.get(t), g, e)
            if pres:
                self.add_pressure(ts, to_hpa(pres.get(t), pres_unit))
            self.evening(ts, hum.get(t))
        n_days = len(self.samples)
        self.history_key = key
        self.refit()
        self.save()
        s = self.scores()
        log.info("Local weather history: %d days rebuilt from InfluxDB, %d evaluated "
                 "(MAE CSI model %s / climatology %s / persistence %s) -> used: %s",
                 n_days, s["n"], _fmt(s["model"]), _fmt(s["clim"]), _fmt(s["persist"]), self.best() or "climatology")

    # --- output ----------------------------------------------------------------------
    def publish(self, ha, now, hpa=None):
        s = self.scores()
        p = self.pending or {}
        k = self.best() or "clim"
        pred = p.get(k)
        t = int(now.timestamp())
        p3 = self._pressure_at(t - 3 * 3600) if "dp3" in self.features else None
        dp3 = None if hpa is None or p3 is None else hpa - p3
        ha.set_state(ENTITY_OUTPUT, outlook_text(pred), attributes={
            "friendly_name": "Local Weather Outlook (station)",
            "icon": "mdi:weather-partly-cloudy",
            "forecast_date": p.get("date"),
            "csi_forecast": _r(pred),
            "forecast_method": {"model": "regression", "persist": "persistence", "clim": "climatology"}[k],
            "csi_climatology": _r(p.get("clim")),
            "csi_today": _r(self.daily_csi(now.astimezone(self.tz).date())),
            "fallback_factor": _r(self.day_factor(date.fromisoformat(p["date"]))) if p.get("date") else None,
            "fallback_active": self.active,
            "csi_regression": _r(p.get("model")),
            "pressure_hpa": _r(hpa, 1),
            "pressure_tendency_3h_hpa": _r(dp3, 1),
            "pressure_tendency": tendency_text(dp3),
            "features": self.features,
            "mae_csi_model": _r(s["model"]),
            "mae_csi_climatology": _r(s["clim"]),
            "mae_csi_persistence": _r(s["persist"]),
            "evaluated_days": s["n"],
            "training_days": len(self.samples),
        })


def _mae(rows, key):
    return float(sum(abs(e[key] - e["y"]) for e in rows) / len(rows)) if rows else None


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"
