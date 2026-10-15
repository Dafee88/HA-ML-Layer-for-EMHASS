"""Building thermal model (ha-ml-layer) - added 28.09.2026.

Learns per floor, from real measurements only (no assumed values):
  - cooling_constant k [1/h]: Newton cooling, heating OFF
        dT/dt = -k * (T_in - T_out)
  - heating_rate h [K/h]: extra warming while heating is ON
        dT/dt = h - k * (T_in - T_out)

Floors (User, 28.09.2026):
  downstairs: sensor.eve_room_e9ff_temperature (Eve Kitchen), heating ON = switch.shelly1pmminig3_5432044e9dfc 'on'
  upstairs:   sensor.eve_room_e49e_temperature (Eve Hall), heating ON = input_select.heating_stage 'upstairs'
              or Tado hvac_action 'heating' (if the Tado entity exposes it)
Outdoor: Ecowitt primary, Vitocal MQTT fallback.

Robustness:
  - learns only over a >= 60 min window with an unchanged heating state
    (underfloor heating reacts slowly - samples right after a switch are skipped)
  - learns cooling only if T_in - T_out >= 3 K (otherwise the signal is noise)
  - a window is dropped if polling had a gap (add-on restart, HA down): a
    heating switch could have happened unseen in between
  - a window is dropped if the sun shone noticeably during it: solar gains
    through the windows would bias k low and h high
  - plausibility bounds, exponential moving average, persisted in /data (atomic)
  - output sensors are rewritten every cycle (REST states do not survive an HA restart)
"""
import json
import logging
import os
import time

log = logging.getLogger("building_thermal")

STATE_FILE = "/data/building_thermal.json"

ENTITY_OUTDOOR = "sensor.ecowitt_outdoor_temp_1f335"
ENTITY_OUTDOOR_FALLBACK = "sensor.vitocal_outdoor_temperature"
ENTITY_STAGE = "input_select.heating_stage"
ENTITY_IRRADIANCE = "sensor.ecowitt_solar_radiation_1f335"

FLOORS = {
    "downstairs": {
        "temp": "sensor.eve_room_e9ff_temperature",
        "heat_switch": "switch.shelly1pmminig3_5432044e9dfc",
        "output": "sensor.building_thermal_downstairs",
        "name": "Building Thermal Downstairs",
    },
    "upstairs": {
        "temp": "sensor.eve_room_e49e_temperature",
        "climate": "climate.smart_thermostat_x_2",
        "output": "sensor.building_thermal_upstairs",
        "name": "Building Thermal Upstairs",
    },
}

MIN_WINDOW_S = 3600        # learn over >= 60 min
MIN_DELTA_K = 3.0          # cooling only with enough indoor/outdoor difference
ALPHA = 0.1                # EMA weight of a new observation
K_BOUNDS = (0.0005, 0.5)   # 1/h
H_BOUNDS = (0.01, 5.0)     # K/h
MAX_IRRADIANCE = 50.0      # W/m^2; above that solar gains distort the window


def _f(st):
    try:
        return float(st["state"])
    except (TypeError, KeyError, ValueError):
        return None


class BuildingThermalModel:
    def __init__(self, max_gap_s=660):
        self.max_gap_s = max_gap_s
        self.data = {f: {"k": None, "h": None, "n_k": 0, "n_h": 0,
                         "anchor": None} for f in FLOORS}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as fh:
                    saved = json.load(fh)
                for f in FLOORS:
                    if f in saved:
                        self.data[f].update(saved[f])
                log.info("Thermal model loaded: %s", {f: (d["k"], d["h"], d["n_k"], d["n_h"]) for f, d in self.data.items()})
            except Exception:
                log.exception("Could not load %s, starting fresh", STATE_FILE)

    def save(self):
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.data, fh)
            os.replace(tmp, STATE_FILE)
        except Exception:
            log.exception("Could not save %s", STATE_FILE)

    @staticmethod
    def _ema(old, new):
        return new if old is None else (1 - ALPHA) * old + ALPHA * new

    def _heating_on(self, floor, states):
        cfg = FLOORS[floor]
        if "heat_switch" in cfg:
            st = states.get(cfg["heat_switch"])
            return None if st is None or st.get("state") in ("unknown", "unavailable") else st.get("state") == "on"
        stage = (states.get(ENTITY_STAGE) or {}).get("state")
        action = ((states.get(cfg["climate"]) or {}).get("attributes") or {}).get("hvac_action")
        if stage is None and action is None:
            return None
        return stage == "upstairs" or action == "heating"

    def update(self, ha):
        entities = [ENTITY_OUTDOOR, ENTITY_OUTDOOR_FALLBACK, ENTITY_STAGE, ENTITY_IRRADIANCE]
        for cfg in FLOORS.values():
            entities.append(cfg["temp"])
            entities.append(cfg.get("heat_switch") or cfg["climate"])
        states = ha.get_states(entities)

        t_out = _f(states.get(ENTITY_OUTDOOR))
        if t_out is None:
            t_out = _f(states.get(ENTITY_OUTDOOR_FALLBACK))
        irr = _f(states.get(ENTITY_IRRADIANCE))  # None -> no sun filter
        now = time.time()

        for floor, cfg in FLOORS.items():
            d = self.data[floor]
            t_in = _f(states.get(cfg["temp"]))
            heat = self._heating_on(floor, states)
            if t_in is None or t_out is None or heat is None:
                log.debug("%s: missing data (t_in=%s t_out=%s heat=%s)", floor, t_in, t_out, heat)
                d["anchor"] = None
                continue

            a = d["anchor"]
            gap = a is not None and now - a.get("last_ts", a["ts"]) > self.max_gap_s
            if a is None or a["heat"] != heat or gap:
                # new window after start, a heating switch or a polling gap
                d["anchor"] = {"ts": now, "last_ts": now, "t_in": t_in, "t_out": t_out,
                               "heat": heat, "irr_max": irr or 0.0}
                continue
            a["last_ts"] = now
            a["irr_max"] = max(a.get("irr_max", 0.0), irr or 0.0)
            dt_h = (now - a["ts"]) / 3600.0
            if dt_h * 3600 < MIN_WINDOW_S:
                continue

            slope = (t_in - a["t_in"]) / dt_h              # K/h
            delta = (t_in + a["t_in"]) / 2 - (t_out + a["t_out"]) / 2

            if a["irr_max"] > MAX_IRRADIANCE:
                log.debug("%s: window skipped, sun up to %.0f W/m2", floor, a["irr_max"])
            elif not heat and delta >= MIN_DELTA_K and slope < 0:
                k_obs = -slope / delta
                if K_BOUNDS[0] <= k_obs <= K_BOUNDS[1]:
                    d["k"] = self._ema(d["k"], k_obs)
                    d["n_k"] += 1
                    log.info("%s cooling: k_obs=%.4f/h -> k=%.4f/h (n=%d)", floor, k_obs, d["k"], d["n_k"])
            elif heat and d["k"] is not None:
                h_obs = slope + d["k"] * delta
                if H_BOUNDS[0] <= h_obs <= H_BOUNDS[1]:
                    d["h"] = self._ema(d["h"], h_obs)
                    d["n_h"] += 1
                    log.info("%s heating: h_obs=%.3fK/h -> h=%.3fK/h (n=%d)", floor, h_obs, d["h"], d["n_h"])

            # next window starts here
            d["anchor"] = {"ts": now, "last_ts": now, "t_in": t_in, "t_out": t_out,
                           "heat": heat, "irr_max": irr or 0.0}

        self.save()
        self.publish(ha)

    def publish(self, ha):
        for floor, cfg in FLOORS.items():
            d = self.data[floor]
            ready = d["n_k"] >= 10 and d["n_h"] >= 5
            ha.set_state(
                cfg["output"],
                "calibrated" if ready else "learning",
                attributes={
                    "friendly_name": cfg["name"],
                    "cooling_constant_per_h": round(d["k"], 5) if d["k"] is not None else None,
                    "heating_rate_k_per_h": round(d["h"], 4) if d["h"] is not None else None,
                    "samples_cooling": d["n_k"],
                    "samples_heating": d["n_h"],
                    "source": "ha-ml-layer / building_thermal (Newton cooling, EMA, measured only)",
                },
            )
