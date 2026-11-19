"""Pre-training of the local weather outlook from a nearby professional station.

The own station has only weeks of history; tomorrow's cloudiness needs years of
examples (every autumn fog, every front). A nearby station of the national
weather service has them. This module reads its hourly history once from CSV
files (format of the GeoSphere Austria data hub, dataset "klima-v2-1h":
time,station,cglo,p,rf,tl,... with the time stamp at the END of the hour) and
fits the same kind of model as local_weather.py:

    tomorrow's daily clear-sky index (CSI) of the reference station
      ~ today's CSI, pressure change 3 h / 12 h, pressure vs 7-day mean,
        humidity, evening cooling (Tmax - T now), daily temperature range,
        season (day of year)

Ridge regression on standardised features (numpy). The result (coefficients,
day-of-year climatology, the reference CSI and features per day) is cached in
/data. Nothing is downloaded - the files are local, so everything keeps
working without internet.

The own station feeds the model through a linear mapping per feature learned
on the days both stations have data (local_weather.py). Backtest 2026 (8 years
Gumpoldskirchen, Ecowitt WH69 4 km away): tomorrow's reference CSI 14 % better
than the 30-day climatology with only the own station's inputs.
"""
import csv
import glob
import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("reference_prior")

CACHE_PATH = Path("/data/reference_prior_v1.json")
FEATURES = ["csi", "dp3", "dp12", "p_anom", "rh", "cool", "drange", "doy_s", "doy_c"]
MAPPED = ["csi", "rh", "cool", "drange"]          # sensor-specific scale, mapped from the own station
RIDGE_LAMBDA = 10.0
MIN_ELEVATION = 10.0
MIN_COVERAGE = 0.8
CSI_CLIP = (0.02, 1.3)
HOLDOUT_DAYS = 120                                 # newest days not used for fitting (they are the own station's
                                                   # history and are used to evaluate the model honestly)
MIN_DAYS = 365
CLIM_HALF_WINDOW = 15


def season(d):
    doy = d.timetuple().tm_yday
    return math.sin(2 * math.pi * doy / 365.25), math.cos(2 * math.pi * doy / 365.25)


class ReferencePrior:
    def __init__(self, pattern, tz_name, evening_hour, cache_path=CACHE_PATH):
        self.pattern = pattern
        self.tz = ZoneInfo(tz_name)
        self.evening_hour = evening_hour
        self.cache_path = cache_path
        self.coef = None           # (mu, sd, w, b)
        self.clim = {}             # day of year (str) -> mean CSI
        self.ref_csi = {}          # "YYYY-MM-DD" -> daily CSI of the reference station
        self.ref_x = {}            # "YYYY-MM-DD" -> evening features (reference scale)
        self.info = {}

    # --- loading -----------------------------------------------------------------------
    def load(self, today=None):
        files = sorted(glob.glob(self.pattern))
        if not files:
            raise FileNotFoundError(f"no file matches {self.pattern}")
        today = today or date.today()
        cutoff = (today.replace(day=1) - timedelta(days=HOLDOUT_DAYS)).isoformat()
        key = [[os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in files] + [cutoff]
        if self.cache_path.exists():
            try:
                c = json.loads(self.cache_path.read_text())
                if c.get("key") == key:
                    self._from_dict(c)
                    log.info("Reference prior loaded from cache: %s", self.info)
                    return self
            except Exception:
                log.exception("Reference prior cache unreadable, rebuilding")
        rows = self._read(files)
        self._build(rows, cutoff)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(self._to_dict(), key=key)))
        os.replace(tmp, self.cache_path)
        log.info("Reference prior fitted from %d file(s): %s", len(files), self.info)
        return self

    @staticmethod
    def _read(files):
        rows = {}
        for f in files:
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    try:
                        t = datetime.fromisoformat(r["time"]).astimezone(timezone.utc)
                    except (KeyError, ValueError):
                        continue
                    rows[t] = {k: _num(r.get(k)) for k in ("cglo", "p", "rf", "tl")}
        return rows

    def _build(self, rows, cutoff):
        from pv_model import solar_position
        times = sorted(rows)
        mids = [t - timedelta(minutes=30) for t in times]          # stamp = end of the hour
        ghi, elev, _ = solar_position(mids)
        days = {}
        for t, m, g, e in zip(times, mids, ghi, elev):
            if e is None or e < MIN_ELEVATION:
                continue
            d = days.setdefault(m.astimezone(self.tz).date(), [0.0, 0.0, 0, 0])
            d[3] += 1
            v = rows[t]["cglo"]
            if v is not None and v >= 0 and g > 0:
                d[0] += v
                d[1] += g
                d[2] += 1
        csi = {d: min(max(a / b, CSI_CLIP[0]), CSI_CLIP[1]) for d, (a, b, n, nd) in days.items()
               if nd and b > 0 and n >= MIN_COVERAGE * nd}
        by_local = {}
        for t in times:
            lt = t.astimezone(self.tz)
            by_local.setdefault(lt.date(), {})[lt.hour] = rows[t]
        X, Y = [], []
        for d in sorted(csi):
            x = self._features(rows, by_local, d, csi[d])
            if x is None:
                continue
            self.ref_x[d.isoformat()] = {k: round(v, 4) for k, v in x.items()}
            y = csi.get(d + timedelta(days=1))
            if y is not None and d.isoformat() < cutoff:
                X.append([x[f] for f in FEATURES])
                Y.append(y)
        self.ref_csi = {d.isoformat(): round(v, 4) for d, v in csi.items()}
        if len(Y) < MIN_DAYS:
            raise ValueError(f"only {len(Y)} usable days before {cutoff} (need {MIN_DAYS})")
        X, Y = np.array(X), np.array(Y)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        w = np.linalg.solve(Z.T @ Z + RIDGE_LAMBDA * np.eye(len(FEATURES)), Z.T @ (Y - Y.mean()))
        self.coef = (mu, sd, w, float(Y.mean()))
        fit_days = [date.fromisoformat(k) for k in self.ref_csi if k < cutoff]
        for doy in range(1, 367):
            vals = [self.ref_csi[d.isoformat()] for d in fit_days
                    if min((d.timetuple().tm_yday - doy) % 365, (doy - d.timetuple().tm_yday) % 365) <= CLIM_HALF_WINDOW]
            if vals:
                self.clim[str(doy)] = round(sum(vals) / len(vals), 4)
        pred = np.clip(Y.mean() + Z @ w, *CSI_CLIP)
        self.info = {"days": len(Y), "first": min(self.ref_csi), "last": max(self.ref_csi), "fit_until": cutoff,
                     "mae_fit": round(float(np.abs(pred - Y).mean()), 3)}

    def _features(self, rows, by_local, d, csi_today):
        h = by_local.get(d, {})
        now = h.get(self.evening_hour)
        if not now or now["p"] is None or now["rf"] is None or now["tl"] is None:
            return None
        ev = datetime(d.year, d.month, d.day, self.evening_hour, tzinfo=self.tz).astimezone(timezone.utc)
        p3, p12 = rows.get(ev - timedelta(hours=3), {}).get("p"), rows.get(ev - timedelta(hours=12), {}).get("p")
        week = [rows[t]["p"] for t in (ev - timedelta(hours=k) for k in range(0, 169)) if t in rows and rows[t]["p"] is not None]
        tday = [h[k]["tl"] for k in range(6, self.evening_hour + 1) if k in h and h[k]["tl"] is not None]
        tnight = [h[k]["tl"] for k in range(0, 9) if k in h and h[k]["tl"] is not None]
        if p3 is None or p12 is None or len(week) < 48 or not tday or not tnight:
            return None
        s, c = season(d)
        return {"csi": csi_today, "dp3": now["p"] - p3, "dp12": now["p"] - p12, "p_anom": now["p"] - sum(week) / len(week),
                "rh": now["rf"], "cool": max(tday) - now["tl"], "drange": max(tday) - min(tnight), "doy_s": s, "doy_c": c}

    # --- use -----------------------------------------------------------------------------
    def predict(self, x):
        """Tomorrow's reference CSI from features in reference scale (dict with FEATURES)."""
        if self.coef is None or x is None or any(x.get(f) is None for f in FEATURES):
            return None
        mu, sd, w, b = self.coef
        z = (np.array([x[f] for f in FEATURES], dtype=float) - mu) / sd
        return float(min(max(b + z @ w, CSI_CLIP[0]), CSI_CLIP[1]))

    def climatology(self, d):
        return self.clim.get(str(d.timetuple().tm_yday))

    def _to_dict(self):
        mu, sd, w, b = self.coef
        return {"coef": [list(map(float, mu)), list(map(float, sd)), list(map(float, w)), b], "clim": self.clim,
                "ref_csi": self.ref_csi, "ref_x": self.ref_x, "info": self.info}

    def _from_dict(self, c):
        mu, sd, w, b = c["coef"]
        self.coef = (np.array(mu), np.array(sd), np.array(w), float(b))
        self.clim, self.ref_csi, self.ref_x, self.info = c["clim"], c["ref_csi"], c["ref_x"], c["info"]


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
