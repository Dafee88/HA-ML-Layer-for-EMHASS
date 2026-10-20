"""Add-on options (/data/options.json, written by the Supervisor on every start).

Every value has a default that matches the original installation, so a missing
key never breaks the add-on. The form in the add-on's Configuration tab is
generated from the `schema` in config.yaml; labels come from
translations/en.yaml.
"""
import json
import logging
import os
import re

log = logging.getLogger("config")

OPTIONS_FILE = os.environ.get("OPTIONS_FILE", "/data/options.json")
_ENTITY_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")

DEFAULT_FLOORS = [
    {"name": "downstairs",
     "temperature_sensor": "sensor.eve_room_e9ff_temperature",
     "heating_switch": "switch.shelly1pmminig3_5432044e9dfc"},
    {"name": "upstairs",
     "temperature_sensor": "sensor.eve_room_e49e_temperature",
     "climate": "climate.smart_thermostat_x_2"},
]


def load_options() -> dict:
    try:
        with open(OPTIONS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("%s not found, using defaults", OPTIONS_FILE)
        return {}


def _entity(opts, key, default, problems):
    """Entity option: blank -> None (optional) or default (required)."""
    value = (opts.get(key, default) or "").strip()
    if not value:
        return None
    if not _ENTITY_RE.match(value):
        problems.append(f"{key}: '{value}' is not a valid entity id")
    return value


class Config:
    def __init__(self, opts=None):
        opts = load_options() if opts is None else opts
        problems = []

        self.ha_url = opts.get("ha_url", "http://supervisor/core")
        self.poll_interval = int(opts.get("poll_interval_seconds", 300))
        self.log_level = opts.get("log_level", "info")
        # Supervisor injects this automatically when hassio_api/homeassistant_api: true
        self.token = os.environ.get("SUPERVISOR_TOKEN", "")

        # --- models (new flat options; old "models" list still understood) ---
        legacy = {m.get("name"): m.get("enabled") for m in opts.get("models", []) or []}
        self.pv_enabled = bool(opts.get("pv_enabled", legacy.get("pv_forecast_correction", True)))
        self.thermal_enabled = bool(opts.get("thermal_enabled", legacy.get("building_thermal", False)))

        # --- site ---
        self.latitude = float(opts.get("latitude", 48.2))
        self.longitude = float(opts.get("longitude", 16.1))
        self.timezone = opts.get("timezone", "Europe/Vienna")

        # --- PV sensors ---
        self.pv_solcast_today = _entity(opts, "pv_solcast_today", "sensor.solcast_pv_forecast_forecast_today", problems)
        self.pv_solcast_tomorrow = _entity(opts, "pv_solcast_tomorrow", "sensor.solcast_pv_forecast_forecast_tomorrow", problems)
        self.pv_actual_power = _entity(opts, "pv_actual_power", "sensor.goodwe_pv_power_effective", problems)
        self.pv_actual_power_fallback = _entity(opts, "pv_actual_power_fallback", "sensor.pv_power", problems)
        self.pv_irradiance = _entity(opts, "pv_irradiance", "sensor.ecowitt_solar_radiation_1f335", problems)

        # --- thermal sensors ---
        self.thermal_outdoor_temp = _entity(opts, "thermal_outdoor_temp", "sensor.ecowitt_outdoor_temp_1f335", problems)
        self.thermal_outdoor_temp_fallback = _entity(opts, "thermal_outdoor_temp_fallback", "sensor.vitocal_outdoor_temperature", problems)
        self.thermal_heating_stage = _entity(opts, "thermal_heating_stage", "", problems)
        self.thermal_floors = []
        for i, fl in enumerate(opts.get("thermal_floors", DEFAULT_FLOORS) or []):
            name = re.sub(r"[^a-z0-9_]", "_", str(fl.get("name", "")).strip().lower())
            entry = {
                "name": name,
                "temp": _entity(fl, "temperature_sensor", "", problems),
                "heat_switch": _entity(fl, "heating_switch", "", problems),
                "climate": _entity(fl, "climate", "", problems),
            }
            if not name or not entry["temp"] or not (entry["heat_switch"] or entry["climate"]):
                problems.append(f"thermal_floors[{i}]: needs name, temperature_sensor and heating_switch or climate")
                continue
            self.thermal_floors.append(entry)

        # --- InfluxDB (v1 HTTP API, e.g. the InfluxDB add-on) ---
        self.influx_enabled = bool(opts.get("influxdb_enabled", False))
        self.influx_host = (opts.get("influxdb_host") or "a0d7b954-influxdb").strip()
        self.influx_port = int(opts.get("influxdb_port", 8086))
        self.influx_ssl = bool(opts.get("influxdb_ssl", False))
        self.influx_database = (opts.get("influxdb_database") or "homeassistant").strip()
        self.influx_username = opts.get("influxdb_username") or None
        self.influx_password = opts.get("influxdb_password") or None
        self.influx_snapshot_hour = int(opts.get("influxdb_dayahead_snapshot_hour", 20))

        if self.pv_enabled and not (self.pv_solcast_today and self.pv_solcast_tomorrow and self.pv_actual_power):
            problems.append("PV model needs pv_solcast_today, pv_solcast_tomorrow and pv_actual_power")
        self.problems = problems
