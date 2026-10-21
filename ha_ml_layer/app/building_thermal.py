"""Building thermal model (ha-ml-layer), in the units of EMHASS' thermal model.

EMHASS (https://emhass.readthedocs.io/en/stable/thermal_model.html):
    T[k+1] = T[k] + P[k] * heating_rate * dt / P_nom - cooling_constant * dt * (T[k] - T_out[k])
  - cooling_constant: degrees per hour per degree of indoor/outdoor difference
  - heating_rate:     K/h at the nominal power P_nom of the heating load

Learned per floor from measurements only:
  - cooling constant k from heating-OFF phases:  dT/dt = -k * (T_in - T_out)
  - heating rate from a daily energy balance (heating-off rest to heating-off
    rest, >= 18 h): with the heating energy E = sum(P dt),
        heating_rate = (T_end - T_start + k * sum((T_in - T_out) dt)) * P_nom / E
    Underfloor heating releases its heat with hours of delay; over a whole
    balance window that delay cancels out (all energy arrives), while a
    regression over the heating phase alone underestimates the rate (~-25 %
    in simulation). Without a power sensor, P = P_nom while heating is on.
    Days with noticeable sun are skipped (EMHASS' model has no solar gains).

Why phases and a regression instead of short windows: room sensors resolve
0.1 K. Over 60 min at 10 K difference that alone is an error of
0.1 K/h / 10 K = 0.01/h - as large as a typical cooling constant (~0.005/h).
So a phase is collected as long as the heating state does not change (night
cool-downs: hours), a straight line is fitted through all readings, and the
phase is used only with enough duration, temperature change and fit quality.

Filters: lock-out after every heating switch (3 h after OFF, 1 h after ON; underfloor heating reacts for
hours), sun above 50 W/m2 ends a phase (solar gains), polling gaps end a
phase. Results are smoothed (EMA) and persisted in /data.
"""
import json
import logging
import os
import time

log = logging.getLogger("building_thermal")

STATE_FILE = "/data/building_thermal_v2.json"   # v1 (60-min end-point estimates) is not reused

LOCKOUT_OFF_H = 3.0         # ignore the first 3 h after heating switched OFF (underfloor heating keeps releasing heat)
LOCKOUT_ON_H = 1.0          # ignore the first 1 h after heating switched ON
MIN_COOL_H = 3.0            # usable cooling phase (after lock-out)
BALANCE_MIN_H = 18.0        # heating balance window: rest to rest, at least 18 h
BALANCE_MAX_SUN_WH = 1000.0 # skip balance windows with more irradiation (Wh/m2)
MIN_HEATING_WH = 500.0      # at least this much heating energy in the window
MAX_PHASE_H = 12.0          # evaluate and restart long phases
MIN_POINTS = 10
MIN_DELTA_K = 3.0           # cooling only with enough indoor/outdoor difference
MIN_CHANGE_K = 0.3          # at least 3 sensor steps of temperature change
MIN_R2 = 0.5
MAX_IRRADIANCE = 50.0       # W/m2
ALPHA = 0.3                 # EMA weight of one phase
K_BOUNDS = (0.0005, 0.2)    # 1/h
H_BOUNDS = (0.05, 10.0)     # K/h
VALID_MIN_PHASES = 3        # per value, before EMHASS may use it

EMHASS_SENSOR = "sensor.building_thermal_emhass"


def _f(st):
    try:
        return float(st["state"])
    except (TypeError, KeyError, ValueError):
        return None


def fit_line(ts_h, values):
    """Least-squares line through (t, value). Returns (slope per hour, r2)."""
    n = len(ts_h)
    mt, mv = sum(ts_h) / n, sum(values) / n
    sxx = sum((t - mt) ** 2 for t in ts_h)
    if sxx <= 0:
        return None, 0.0
    sxy = sum((t - mt) * (v - mv) for t, v in zip(ts_h, values))
    slope = sxy / sxx
    ss_tot = sum((v - mv) ** 2 for v in values)
    ss_res = sum((v - (mv + slope * (t - mt))) ** 2 for t, v in zip(ts_h, values))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


class BuildingThermalModel:
    def __init__(self, cfg, max_gap_s=660, state_file=STATE_FILE):
        self.max_gap_s = max_gap_s
        self.state_file = state_file
        self.outdoor = cfg.thermal_outdoor_temp
        self.outdoor_fallback = cfg.thermal_outdoor_temp_fallback
        self.stage = cfg.thermal_heating_stage
        self.irradiance = cfg.pv_irradiance
        self.power_threshold = cfg.thermal_heating_power_threshold_w
        self.nominal_power = cfg.thermal_nominal_power_w
        self.floors = {}
        for fl in cfg.thermal_floors:
            self.floors[fl["name"]] = dict(
                fl, output=f"sensor.building_thermal_{fl['name']}",
                friendly=f"Building Thermal {fl['name'].replace('_', ' ').title()}")
        self.data = {f: {"k": None, "h": None, "n_k": 0, "n_h": 0, "h_normalised": False,
                         "last_k_r2": None} for f in self.floors}
        self.phase = {f: None for f in self.floors}     # in memory only
        self.balance = {f: None for f in self.floors}   # in memory only
        self.off_since = {f: None for f in self.floors}
        if os.path.exists(state_file):
            try:
                with open(state_file) as fh:
                    saved = json.load(fh)
                for f in self.floors:
                    if f in saved:
                        self.data[f].update(saved[f])
                log.info("Thermal model loaded: %s",
                         {f: (d["k"], d["h"], d["n_k"], d["n_h"]) for f, d in self.data.items()})
            except Exception:
                log.exception("Could not load %s, starting fresh", state_file)

    def save(self):
        try:
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, self.state_file)
        except Exception:
            log.exception("Could not save %s", self.state_file)

    # --- inputs -------------------------------------------------------------
    def _entities(self):
        out = [self.outdoor, self.outdoor_fallback, self.stage, self.irradiance]
        for c in self.floors.values():
            out += [c["temp"], c.get("heat_switch"), c.get("climate"), c.get("power")]
        return [e for e in dict.fromkeys(out) if e]

    def _heating(self, floor, states):
        """(heating on?, heating power W or None). None = unknown."""
        c = self.floors[floor]
        power = _f(states.get(c["power"])) if c.get("power") else None
        if c.get("heat_switch"):
            st = states.get(c["heat_switch"])
            if st is None or st.get("state") in ("unknown", "unavailable"):
                return None, power
            return st.get("state") == "on", power
        if c.get("power"):
            return (None if power is None else power > self.power_threshold), power
        stage = (states.get(self.stage) or {}).get("state") if self.stage else None
        action = ((states.get(c["climate"]) or {}).get("attributes") or {}).get("hvac_action")
        if stage is None and action is None:
            return None, power
        return stage == floor or action == "heating", power

    # --- phase handling -------------------------------------------------------
    def update(self, ha, now=None):
        states = ha.get_states(self._entities())
        t_out = _f(states.get(self.outdoor))
        if t_out is None and self.outdoor_fallback:
            t_out = _f(states.get(self.outdoor_fallback))
        irr = _f(states.get(self.irradiance)) if self.irradiance else None
        now = time.time() if now is None else now

        for floor, c in self.floors.items():
            t_in = _f(states.get(c["temp"]))
            heat, power = self._heating(floor, states)
            p = self.phase[floor]
            if t_in is None or t_out is None or heat is None:
                self._close(floor)
                self.balance[floor] = None
                continue
            self._balance_step(floor, now, t_in, t_out, heat, power, irr)
            if p is not None and now - p["last_ts"] > self.max_gap_s:
                self._close(floor)
                p = None
            if p is not None and p["heat"] != heat:
                self._close(floor)
                p = None
            if p is None:
                lock = (LOCKOUT_ON_H if heat else LOCKOUT_OFF_H) * 3600
                p = self.phase[floor] = {"heat": heat, "start": now, "collect_from": now + lock,
                                         "last_ts": now, "pts": []}
            p["last_ts"] = now
            if irr is not None and irr > MAX_IRRADIANCE:
                # solar gains: end the usable part, keep the lock-out state
                self._evaluate(floor, p)
                p["pts"] = []
                continue
            if now >= p["collect_from"]:
                p["pts"].append((now / 3600.0, t_in, t_out, power))
                if p["pts"] and p["pts"][-1][0] - p["pts"][0][0] >= MAX_PHASE_H:
                    self._evaluate(floor, p)
                    p["pts"] = []

        self.save()
        self.publish(ha)

    def _close(self, floor):
        p = self.phase.get(floor)
        if p is not None:
            self._evaluate(floor, p)
        self.phase[floor] = None

    def _evaluate(self, floor, p):
        pts = p["pts"]
        if len(pts) < MIN_POINTS:
            return
        d = self.data[floor]
        ts = [q[0] - pts[0][0] for q in pts]
        dur = ts[-1]
        tin = [q[1] for q in pts]
        slope, r2 = fit_line(ts, tin)
        if slope is None:
            return
        delta = sum(q[1] - q[2] for q in pts) / len(pts)
        change = abs(slope * dur)
        if not p["heat"]:
            if dur < MIN_COOL_H or delta < MIN_DELTA_K or slope >= 0 or change < MIN_CHANGE_K or r2 < MIN_R2:
                log.debug("%s cooling phase rejected: %.1f h, delta %.1f K, slope %.3f K/h, r2 %.2f",
                          floor, dur, delta, slope, r2)
                return
            k_obs = -slope / delta
            if K_BOUNDS[0] <= k_obs <= K_BOUNDS[1]:
                d["k"] = k_obs if d["k"] is None else (1 - ALPHA) * d["k"] + ALPHA * k_obs
                d["n_k"] += 1
                d["last_k_r2"] = round(r2, 3)
                log.info("%s cooling phase %.1f h: k_obs=%.5f/h (r2 %.2f) -> k=%.5f/h (n=%d)",
                         floor, dur, k_obs, r2, d["k"], d["n_k"])

    def _balance_step(self, floor, now, t_in, t_out, heat, power, irr):
        """Daily energy balance for the heating rate (see module docstring)."""
        if heat:
            self.off_since[floor] = None
        elif self.off_since[floor] is None:
            self.off_since[floor] = now
        resting = (not heat and self.off_since[floor] is not None
                   and now - self.off_since[floor] >= LOCKOUT_OFF_H * 3600)
        b = self.balance[floor]
        if b is not None and now - b["last_ts"] > self.max_gap_s:
            b = self.balance[floor] = None                      # gap: unknown heating in between
        if b is None:
            if resting:
                self.balance[floor] = {"start": now, "last_ts": now, "t_start": t_in,
                                       "delta_h": 0.0, "energy_wh": 0.0, "sun_wh": 0.0}
            return
        dt_h = (now - b["last_ts"]) / 3600.0
        b["last_ts"] = now
        b["delta_h"] += (t_in - t_out) * dt_h
        p = power if power is not None else (self.nominal_power or 0.0) if heat else 0.0
        if heat and power is None and not self.nominal_power:
            p = 1.0                                          # unit power: rate per hour of heating
        b["energy_wh"] += max(p, 0.0) * dt_h
        b["sun_wh"] += max(irr or 0.0, 0.0) * dt_h
        if resting and (now - b["start"]) / 3600.0 >= BALANCE_MIN_H:
            self._evaluate_balance(floor, b, t_in, power is not None)
            self.balance[floor] = {"start": now, "last_ts": now, "t_start": t_in,
                                   "delta_h": 0.0, "energy_wh": 0.0, "sun_wh": 0.0}

    def _evaluate_balance(self, floor, b, t_end, has_power):
        d = self.data[floor]
        hours = (b["last_ts"] - b["start"]) / 3600.0
        min_energy = MIN_HEATING_WH if (has_power or self.nominal_power) else 1.0
        if d["k"] is None or b["energy_wh"] < min_energy or b["sun_wh"] > BALANCE_MAX_SUN_WH:
            log.debug("%s balance window skipped: %.1f h, energy %.0f Wh, sun %.0f Wh/m2, k=%s",
                      floor, hours, b["energy_wh"], b["sun_wh"], d["k"])
            return
        gain = t_end - b["t_start"] + d["k"] * b["delta_h"]   # K delivered by heating
        p_ref = self.nominal_power or 1.0
        h_obs = gain * p_ref / b["energy_wh"]
        normalised = bool(self.nominal_power)
        if H_BOUNDS[0] <= h_obs <= H_BOUNDS[1]:
            if d["h"] is not None and d.get("h_normalised") != normalised:
                d["h"], d["n_h"] = None, 0                       # option changed: restart
            d["h"] = h_obs if d["h"] is None else (1 - ALPHA) * d["h"] + ALPHA * h_obs
            d["h_normalised"] = normalised
            d["n_h"] += 1
            log.info("%s heating balance %.1f h: %.0f Wh -> h_obs=%.3f K/h%s -> h=%.3f (n=%d)",
                     floor, hours, b["energy_wh"], h_obs, " at nominal power" if normalised else " per heating hour",
                     d["h"], d["n_h"])

    # --- outputs ----------------------------------------------------------------
    def emhass_values(self):
        ks = [d["k"] for d in self.data.values() if d["k"] is not None]
        hs = [d["h"] for d in self.data.values() if d["h"] is not None]
        k = sum(ks) / len(ks) if ks else None
        h = sum(hs) / len(hs) if hs else None
        normalised = bool(hs) and all(d.get("h_normalised") for d in self.data.values() if d["h"] is not None)
        valid = (bool(self.data)
                 and all(d["n_k"] >= VALID_MIN_PHASES and d["n_h"] >= VALID_MIN_PHASES for d in self.data.values())
                 and k is not None and h is not None and normalised
                 and K_BOUNDS[0] <= k <= K_BOUNDS[1] and H_BOUNDS[0] <= h <= H_BOUNDS[1])
        return k, h, normalised, valid

    def publish(self, ha):
        for floor, c in self.floors.items():
            d = self.data[floor]
            ready = d["n_k"] >= VALID_MIN_PHASES and d["n_h"] >= VALID_MIN_PHASES
            ha.set_state(c["output"], "calibrated" if ready else "learning", attributes={
                "friendly_name": c["friendly"],
                "cooling_constant": round(d["k"], 5) if d["k"] is not None else None,
                "heating_rate": round(d["h"], 4) if d["h"] is not None else None,
                "heating_rate_at_nominal_power": d.get("h_normalised", False),
                "phases_cooling": d["n_k"],
                "balances_heating": d["n_h"],
                "last_cooling_r2": d.get("last_k_r2"),
                "source": "ha-ml-layer / building_thermal (phase regression, EMHASS units)",
            })
        k, h, normalised, valid = self.emhass_values()
        ha.set_state(EMHASS_SENSOR, "valid" if valid else "learning", attributes={
            "friendly_name": "Building Thermal for EMHASS",
            "cooling_constant": round(k, 5) if k is not None else None,
            "heating_rate": round(h, 4) if h is not None else None,
            "nominal_power_w": self.nominal_power,
            "heating_rate_at_nominal_power": normalised,
            "valid": valid,
            "floors": len(self.floors),
            "min_phases": VALID_MIN_PHASES,
            "source": "ha-ml-layer / building_thermal (mean of floors)",
        })
