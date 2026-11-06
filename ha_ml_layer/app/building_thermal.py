"""Building thermal model (ha-ml-layer), in the units of EMHASS' thermal model.

EMHASS (https://emhass.readthedocs.io/en/stable/thermal_model.html):
    T[k+1] = T[k] + P[k-L] * heating_rate * dt / P_nom - cooling_constant * dt * (T[k] - T_out[k])
  - cooling_constant: per hour per degree of indoor/outdoor difference
  - heating_rate:     K/h at the nominal power P_nom of the heating load
  - thermal_inertia:  lag L of the heating in hours (underfloor heating: hours)

Since 0.8.0 the parameters are fitted the way EMHASS uses them: a grey-box fit
on the error of 24 h open-loop simulations (start at the measured temperature,
then only the measured outdoor temperature, sun and heating power - the room
temperature is not seen again). Up to 0.7.x heating-off phases and daily
energy balances were used; replayed on real history that method attributed
internal gains (people, devices, neighbouring flats) to the heating and gave
heating rates several times too high.

The fitted model has two extra terms EMHASS' equation does not have:
    + dt * (solar_gain * I[k] / 100 W/m2 + internal_gain)
They are published as an equivalent outdoor temperature offset
    T_out_eff = T_out + (solar_gain * I / 100 + internal_gain) / cooling_constant
so the exact model can be used in EMHASS by adding the offset to the outdoor
temperature forecast; without it, heating_rate/cooling_constant still come
from the full fit.

Data: 30-min slots of the mean floor temperature, outdoor temperature,
irradiance and heating power. At start the slots are rebuilt from InfluxDB
(Home Assistant's InfluxDB integration, last HISTORY_DAYS days), then live
slots are added. The fit runs at start and every FIT_INTERVAL_H hours.

Valid (EMHASS may use it) only if, on the last TEST_DAYS days that were not
used for fitting, the 24 h prediction error is not worse than the trivial
forecast "the temperature stays as it is" (+ VALID_TOLERANCE_K).
"""
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

log = logging.getLogger("building_thermal")

STATE_FILE = "/data/building_thermal_v6.json"   # v6: rebuilt (0.8.4 carry last value, not bucket mean)

SLOT_S = 1800
DT_H = SLOT_S / 3600.0
HORIZON = 48                # 24 h rollouts
STRIDE = 6                  # a rollout starts every 3 h
MAX_LAG = 12                # thermal inertia 0 .. 6 h
HISTORY_DAYS = 35
TEST_DAYS = 2
MIN_TRAIN_DAYS = 4
FIT_INTERVAL_H = 24
STALE_S = 12 * 3600         # InfluxDB keeps no "unavailable": a main temperature sensor WITH fallback
                            # is carried at most 12 h (others: unlimited - HA writes only changes)
OFFSET_ALPHA = 0.02         # EMA of (main - fallback) temperature while both report
VALID_TOLERANCE_K = 0.1
MIN_SLOT_SHARE = 0.5        # slot needs half of its polls
MIN_HEATING_KWH = 2.0       # heating energy in the training data, else heating_rate is not identifiable
K_BOUNDS = (0.0005, 0.2)
H_BOUNDS = (0.05, 10.0)
SUN_BOUNDS = (0.0, 1.0)     # K/h per 100 W/m2
Q_BOUNDS = (-1.0, 1.0)      # K/h

EMHASS_SENSOR = "sensor.building_thermal_emhass"


def _f(st):
    try:
        v = float(st["state"])
        return v if math.isfinite(v) else None
    except (TypeError, KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- model
def _windows(T, To, I, P, lag_max=MAX_LAG):
    """Start indices with complete data for a 24 h rollout (incl. heating lag history)."""
    n = len(T)
    ok = ~(np.isnan(T) | np.isnan(To) | np.isnan(I) | np.isnan(P))
    cum = np.concatenate([[0], np.cumsum(ok)])
    starts = []
    for s in range(lag_max, n - HORIZON, 1):
        a, b = s - lag_max, s + HORIZON + 1
        if cum[b] - cum[a] == b - a:
            starts.append(s)
    return np.array(starts, dtype=int)


def simulate(p, lag, starts, T, To, I, P, p_nom):
    """Vectorised rollouts. Returns predictions shape (len(starts), HORIZON)."""
    k, h, a, q = p
    x = T[starts].copy()
    out = np.empty((len(starts), HORIZON))
    for i in range(HORIZON):
        j = starts + i
        x = x + P[j - lag] * h * DT_H / p_nom - k * DT_H * (x - To[j]) + DT_H * (a * I[j] / 100.0 + q)
        out[:, i] = x
    return out


def _target(starts, T):
    return np.stack([T[starts + i + 1] for i in range(HORIZON)], axis=1)


def _rmse24(pred, starts, T):
    e = pred[:, -1] - T[starts + HORIZON]
    return float(np.sqrt(np.mean(e ** 2)))


def fit(T, To, I, P, p_nom, starts, use_sun=True):
    """Least squares on the simulation error; best lag by grid search."""
    from scipy.optimize import least_squares
    y = _target(starts, T)
    lo = [K_BOUNDS[0], H_BOUNDS[0], SUN_BOUNDS[0], Q_BOUNDS[0]]
    hi = [K_BOUNDS[1], H_BOUNDS[1], SUN_BOUNDS[1] if use_sun else 1e-9, Q_BOUNDS[1]]
    best = None
    for lag in range(0, MAX_LAG + 1):
        res = least_squares(lambda p: (simulate(p, lag, starts, T, To, I, P, p_nom) - y).ravel(),
                            [0.005, 1.0, 0.0, 0.0], bounds=(lo, hi), max_nfev=200)
        if best is None or res.cost < best[1].cost:
            best = (lag, res)
    lag, res = best
    return [float(v) for v in res.x], lag


# --------------------------------------------------------------------------- class
class BuildingThermalModel:
    def __init__(self, cfg, max_gap_s=660, state_file=STATE_FILE):
        self.state_file = state_file
        self.outdoor = cfg.thermal_outdoor_temp
        self.outdoor_fallback = cfg.thermal_outdoor_temp_fallback
        self.stage = cfg.thermal_heating_stage
        self.irradiance = cfg.pv_irradiance
        self.power_threshold = cfg.thermal_heating_power_threshold_w
        self.nominal_power = cfg.thermal_nominal_power_w
        self.poll_s = max(60, max_gap_s - 60) / 2.0
        self.floors = {}
        for fl in cfg.thermal_floors:
            self.floors[fl["name"]] = dict(
                fl, output=f"sensor.building_thermal_{fl['name']}",
                friendly=f"Building Thermal {fl['name'].replace('_', ' ').title()}")
        self.power_sensors = list(dict.fromkeys(c["power"] for c in self.floors.values() if c.get("power")))
        self.slots = {}     # slot epoch -> {"t": {floor: T}, "to": , "i": , "p": }
        self.acc = None     # running slot
        self.result = {"house": None, "floors": {}, "last_fit": 0, "info": "waiting for data"}
        if os.path.exists(state_file):
            try:
                with open(state_file) as fh:
                    saved = json.load(fh)
                self.slots = {int(k): v for k, v in saved.get("slots", {}).items()}
                self.result = saved.get("result", self.result)
                log.info("Thermal model loaded: %d slots, house %s", len(self.slots), self.result.get("house"))
            except Exception:
                log.exception("Could not load %s, starting fresh", state_file)

    def save(self):
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"slots": self.slots, "result": self.result}, fh)
            os.replace(tmp, self.state_file)
        except Exception:
            log.exception("Could not save %s", self.state_file)

    # --- inputs -------------------------------------------------------------
    def _entities(self):
        out = [self.outdoor, self.outdoor_fallback, self.stage, self.irradiance]
        for c in self.floors.values():
            out += [c["temp"], c.get("temp_fallback"), c.get("heat_switch"), c.get("climate"), c.get("power")]
        return [e for e in dict.fromkeys(out) if e]

    def _floor_temp(self, floor, states):
        """Main sensor; while it is unavailable the fallback sensor + learned offset."""
        c = self.floors[floor]
        main = _f(states.get(c["temp"]))
        fb = _f(states.get(c["temp_fallback"])) if c.get("temp_fallback") else None
        offsets = self.result.setdefault("offsets", {})
        if main is not None and fb is not None:
            old = offsets.get(floor)
            offsets[floor] = round(main - fb if old is None else (1 - OFFSET_ALPHA) * old + OFFSET_ALPHA * (main - fb), 3)
        if main is not None:
            return main
        if fb is not None:
            return fb + (offsets.get(floor) or 0.0)
        return None

    def _heating_power(self, states):
        """House heating power in W (None = unknown)."""
        if self.power_sensors:
            vals = [_f(states.get(e)) for e in self.power_sensors]
            if any(v is None for v in vals):
                return None
            return sum(max(v, 0.0) for v in vals)
        p_nom = self.nominal_power or 1000.0
        total = 0.0
        stage = (states.get(self.stage) or {}).get("state") if self.stage else None
        for floor, c in self.floors.items():
            if c.get("heat_switch"):
                st = (states.get(c["heat_switch"]) or {}).get("state")
                if st in (None, "unknown", "unavailable"):
                    return None
                on = st == "on"
            else:
                action = ((states.get(c.get("climate")) or {}).get("attributes") or {}).get("hvac_action")
                if stage is None and action is None:
                    return None
                on = stage == floor or action == "heating"
            total = max(total, p_nom if on else 0.0)
        return total

    # --- live update -------------------------------------------------------------
    def update(self, ha, now=None):
        states = ha.get_states(self._entities())
        now = time.time() if now is None else now
        t_out = _f(states.get(self.outdoor))
        if t_out is None and self.outdoor_fallback:
            t_out = _f(states.get(self.outdoor_fallback))
        irr = _f(states.get(self.irradiance)) if self.irradiance else 0.0
        temps = {f: self._floor_temp(f, states) for f in self.floors}
        power = self._heating_power(states)

        slot = int(now // SLOT_S) * SLOT_S
        if self.acc is not None and self.acc["slot"] != slot:
            self._close_slot()
        if self.acc is None:
            self.acc = {"slot": slot, "n": 0, "t": {f: [] for f in self.floors}, "to": [], "i": [], "p": []}
        a = self.acc
        a["n"] += 1
        for f, v in temps.items():
            if v is not None:
                a["t"][f].append(v)
        if t_out is not None:
            a["to"].append(t_out)
        if irr is not None:
            a["i"].append(max(irr, 0.0))
        if power is not None:
            a["p"].append(power)

        if time.time() - self.result.get("last_fit", 0) >= FIT_INTERVAL_H * 3600:
            self.refit()
        self.publish(ha)

    def _close_slot(self):
        a, self.acc = self.acc, None
        need = max(1, int(MIN_SLOT_SHARE * SLOT_S / self.poll_s))
        if min(len(a["to"]), len(a["p"])) < need or any(len(v) < need for v in a["t"].values()):
            return
        self.slots[a["slot"]] = {"t": {f: sum(v) / len(v) for f, v in a["t"].items()},
                                 "to": sum(a["to"]) / len(a["to"]),
                                 "i": sum(a["i"]) / len(a["i"]) if a["i"] else 0.0,
                                 "p": sum(a["p"]) / len(a["p"])}
        cutoff = a["slot"] - HISTORY_DAYS * 86400
        self.slots = {k: v for k, v in self.slots.items() if k >= cutoff}
        self.save()
        if not (self.result.get("house") or {}).get("valid"):
            self.refit()        # not valid yet: try again with every new slot, not only once a day

    # --- history from InfluxDB ----------------------------------------------------
    @staticmethod
    def _series(influx, entity_id, start, end, max_carry=None):
        """{5-min bucket epoch: value}; empty buckets carry the last point (at most max_carry s)."""
        from history import _buckets
        return _buckets(influx, entity_id, start, end, max_carry=max_carry, lookback=STALE_S)

    def backfill(self, influx, now=None):
        """Rebuild 30-min slots from Home Assistant's InfluxDB history (read-only)."""
        if not self.power_sensors:
            log.info("Thermal history: no heating_power_sensor configured, learning from live data only")
            return
        end = datetime.fromtimestamp(now or time.time(), timezone.utc)
        start = end - timedelta(days=HISTORY_DAYS)
        limited = {c["temp"] for c in self.floors.values() if c.get("temp_fallback")}
        series = {}
        for e in [self.outdoor, self.outdoor_fallback, self.irradiance] + self.power_sensors \
                + [c["temp"] for c in self.floors.values()] + [c.get("temp_fallback") for c in self.floors.values()]:
            if e and e not in series:
                series[e] = self._series(influx, e, start, end, STALE_S if e in limited else None)
        b = 300
        first = int(start.timestamp()) // SLOT_S * SLOT_S
        slots = list(range(first, int(end.timestamp()) // SLOT_S * SLOT_S, SLOT_S))

        def mean(e, slot):
            if not e:
                return None
            vals = [series[e].get(slot + i * b) for i in range(SLOT_S // b)]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if len(vals) >= 3 else None

        # offset main - fallback from the history where both report
        offsets = self.result.setdefault("offsets", {})
        for f, c in self.floors.items():
            if c.get("temp_fallback") and offsets.get(f) is None:
                diffs = [mean(c["temp"], s) - mean(c["temp_fallback"], s) for s in slots
                         if mean(c["temp"], s) is not None and mean(c["temp_fallback"], s) is not None]
                if diffs:
                    offsets[f] = round(sum(diffs) / len(diffs), 3)
                    log.info("Thermal history: %s fallback offset %+.2f K from %d slots", f, offsets[f], len(diffs))
        n_new = n_fb = 0
        for slot in slots:
            if slot in self.slots:
                continue
            temps = {}
            for f, c in self.floors.items():
                t = mean(c["temp"], slot)
                if t is None and c.get("temp_fallback"):
                    t = mean(c["temp_fallback"], slot)
                    if t is not None:
                        t += offsets.get(f) or 0.0
                        n_fb += 1
                temps[f] = t
            to = mean(self.outdoor, slot)
            if to is None:
                to = mean(self.outdoor_fallback, slot)
            ps = [mean(e, slot) for e in self.power_sensors]
            irr = mean(self.irradiance, slot) if self.irradiance else 0.0
            if to is None or any(v is None for v in temps.values()) or any(v is None for v in ps):
                continue
            self.slots[slot] = {"t": temps, "to": to, "i": max(irr or 0.0, 0.0), "p": sum(max(v, 0.0) for v in ps)}
            n_new += 1
        log.info("Thermal history: %d slots rebuilt from InfluxDB (%d floor values from fallback sensors), "
                 "%d slots in total, %.1f days", n_new, n_fb, len(self.slots), len(self.slots) / 48.0)
        self.save()
        self.refit()

    # --- fit -------------------------------------------------------------------------
    def _arrays(self):
        if not self.slots:
            return None
        keys = sorted(self.slots)
        grid = list(range(keys[0], keys[-1] + SLOT_S, SLOT_S))
        n = len(grid)
        to, irr, p = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        temps = {f: np.full(n, np.nan) for f in self.floors}
        for i, s in enumerate(grid):
            v = self.slots.get(s)
            if v is None:
                continue
            to[i], irr[i], p[i] = v["to"], v.get("i", 0.0), v["p"]
            for f in self.floors:
                t = (v.get("t") or {}).get(f)
                temps[f][i] = np.nan if t is None else t
        stack = np.stack(list(temps.values()))
        house = stack.mean(axis=0)          # NaN if any floor is missing
        return np.array(grid), temps, house, to, irr, p

    def _fit_one(self, T, To, I, P, grid):
        p_nom = self.nominal_power or 1000.0
        starts = _windows(T, To, I, P)
        if len(starts) == 0:
            return None, "no complete 24 h window yet"
        split = grid[-1] - TEST_DAYS * 86400 - HORIZON * SLOT_S
        tr = starts[grid[starts] < split][::STRIDE // 2 or 1]
        te = starts[grid[starts] >= split][::STRIDE]
        train_days = float((grid[tr[-1]] - grid[tr[0]]) / 86400 + 1) if len(tr) else 0.0
        heat_kwh = float(np.nansum(P[grid < split]) * DT_H / 1000.0)
        if train_days < MIN_TRAIN_DAYS or len(te) < 4:
            return None, f"collecting data ({train_days:.1f} of {MIN_TRAIN_DAYS} training days + {TEST_DAYS} test days)"
        if heat_kwh < MIN_HEATING_KWH:
            return None, f"too little heating in the data ({heat_kwh:.1f} kWh, need {MIN_HEATING_KWH})"
        use_sun = bool(self.irradiance)
        params, lag = fit(T, To, I, P, p_nom, tr, use_sun)
        err = _rmse24(simulate(params, lag, te, T, To, I, P, p_nom), te, T)
        err_persist = float(np.sqrt(np.mean((T[te] - T[te + HORIZON]) ** 2)))
        # final parameters from all data
        allst = starts[::STRIDE // 2 or 1]
        params, lag = fit(T, To, I, P, p_nom, allst, use_sun)
        k, h, a, q = params
        valid = (err <= err_persist + VALID_TOLERANCE_K and bool(self.nominal_power)
                 and K_BOUNDS[0] < k < K_BOUNDS[1] and H_BOUNDS[0] < h < H_BOUNDS[1])
        return {"cooling_constant": round(k, 5), "heating_rate": round(h, 4),
                "thermal_inertia": round(lag * DT_H, 1), "solar_gain": round(a, 4),
                "internal_gain": round(q, 4),
                "outdoor_temp_offset": round(q / k, 2) if k > 0 else None,
                "rmse_24h": round(err, 3), "rmse_24h_persistence": round(err_persist, 3),
                "train_days": round(train_days, 1), "heating_kwh": round(heat_kwh, 1),
                "valid": valid}, None

    def refit(self):
        self.result["last_fit"] = time.time()
        arr = self._arrays()
        if arr is None:
            self.result["info"] = "waiting for data"
            return
        grid, temps, house, to, irr, p = arr
        t0 = time.monotonic()
        try:
            res, info = self._fit_one(house, to, irr, p, grid)
            self.result["house"], self.result["info"] = res, info or "fitted"
            for f, T in temps.items():
                self.result["floors"][f] = self._fit_one(T, to, irr, p, grid)[0]
        except Exception:
            log.exception("Thermal fit failed")
            self.result["info"] = "fit failed (see log)"
        log.info("Thermal fit (%.0f s): %s | %s", time.monotonic() - t0, self.result["info"], self.result["house"])
        self.save()

    # --- outputs ----------------------------------------------------------------
    def emhass_values(self):
        r = self.result.get("house") or {}
        return (r.get("cooling_constant"), r.get("heating_rate"), bool(self.nominal_power), bool(r.get("valid")))

    def publish(self, ha):
        src = "ha-ml-layer / building_thermal (grey-box fit on 24 h simulations, EMHASS units)"
        for floor, c in self.floors.items():
            r = self.result["floors"].get(floor) or {}
            ha.set_state(c["output"], "calibrated" if r.get("valid") else "learning", attributes=dict(
                {"friendly_name": c["friendly"], "source": src,
                 "fallback_offset": (self.result.get("offsets") or {}).get(floor)}, **r))
        r = self.result.get("house") or {}
        k, h, normalised, valid = self.emhass_values()
        ha.set_state(EMHASS_SENSOR, "valid" if valid else "learning", attributes=dict(
            {"friendly_name": "Building Thermal for EMHASS", "nominal_power_w": self.nominal_power,
             "heating_rate_at_nominal_power": normalised, "floors": len(self.floors),
             "slots": len(self.slots), "info": self.result.get("info"),
             "last_fit": datetime.fromtimestamp(self.result.get("last_fit") or 0, timezone.utc).isoformat(),
             "source": src}, **{kk: v for kk, v in r.items() if kk != "valid"}, valid=valid))
