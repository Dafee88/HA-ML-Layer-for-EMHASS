"""PV forecast correction (ha-ml-layer).

Two models, rebuilt 14.10.2026:

1. Day-ahead model (DayAheadModel)
   Uses ONLY features that also exist for future time steps: Solcast P50/P10/P90,
   clear-sky GHI, solar elevation and cyclic time. Training and forecast use the
   SAME feature builder, so the curve sent to EMHASS is computed exactly the way
   the model was trained. (Before: training saw live Ecowitt cloud data, the
   forecast path always assumed a clear sky -> systematic over-forecast on
   cloudy afternoons.)

2. Nowcast model (NowcastModel)
   Corrects only the CURRENT value with live Ecowitt irradiance (clear-sky index
   and its short-term variability). Its effect on the curve fades out over the
   next ~2 hours (smart persistence), because a sensor cannot see clouds that
   have not arrived yet.

Input sources are independent of EMHASS: Solcast directly, never
sensor.p_pv_forecast (that is EMHASS' own output and would become the ML
curve itself once EMHASS receives it -> the model would learn from its own
prediction).
"""
import logging
import math
import pickle
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pvlib
from river import forest, metrics

log = logging.getLogger("pv_model")

# --- Entities (verified live 14.10.2026) ------------------------------------
ENTITY_SOLCAST_TODAY = "sensor.solcast_pv_forecast_forecast_today"        # .detailedForecast: pv_estimate/10/90 in kW (30-min average power)
ENTITY_SOLCAST_TOMORROW = "sensor.solcast_pv_forecast_forecast_tomorrow"
ENTITY_ACTUAL_POWER = "sensor.goodwe_pv_power_effective"                  # W, Modbus primary + native fallback
ENTITY_ACTUAL_POWER_FALLBACK = "sensor.pv_power"                          # W, native GoodWe (was unavailable on 14.10.)
ENTITY_IRRADIANCE = "sensor.ecowitt_solar_radiation_1f335"                # W/m^2 (entity id can change after a gateway reconnect!)
ENTITY_OUTPUT_CORRECTED = "sensor.pv_ml_corrected_forecast"               # current value (day-ahead x nowcast)
ENTITY_OUTPUT_CURVE = "sensor.pv_ml_corrected_forecast_curve"             # 48 x 30 min, read by the EMHASS rest_commands
# -----------------------------------------------------------------------------

DAYAHEAD_MODEL_PATH = Path("/data/pv_dayahead_model.pkl")
NOWCAST_MODEL_PATH = Path("/data/pv_nowcast_model.pkl")

LATITUDE = 48.2
LONGITUDE = 16.1
TZ = "Europe/Vienna"

SLOT = timedelta(minutes=30)
HORIZON_SLOTS = 48
KW_TO_W = 1000.0               # Solcast pv_estimate is kW (sum over a day = 2 x daily kWh, checked 14.10.2026)
PLAUSIBLE_MAX_W = 15000
MIN_ELEVATION_LEARN = 5.0      # degrees; below that PV is ~0 and only adds noise
NOWCAST_TAU_H = 1.0            # nowcast influence decays with exp(-dt/tau)
NOWCAST_FACTOR_BOUNDS = (0.1, 2.0)
NOWCAST_MIN_BASE_W = 100.0     # below this the ratio nowcast/day-ahead is meaningless
CSI_WINDOW = 6                 # cycles for the variability feature (6 x 5 min = 30 min)
MIN_SAMPLES_DAYAHEAD = 300     # ~2-3 days of daytime 5-min samples before the model replaces raw Solcast
MIN_SAMPLES_NOWCAST = 150      # ~1-2 days before the Ecowitt nowcast is allowed to correct

_LOCATION = pvlib.location.Location(latitude=LATITUDE, longitude=LONGITUDE, tz=TZ)


# --- Solar geometry ----------------------------------------------------------
def solar_geometry(timestamps):
    """Clear-sky GHI (Ineichen) and apparent solar elevation for a list of
    timezone-aware datetimes, in one pvlib call."""
    times = pd.DatetimeIndex(timestamps)
    if times.tz is None:
        times = times.tz_localize("UTC")
    clearsky = _LOCATION.get_clearsky(times, model="ineichen")
    solpos = _LOCATION.get_solarposition(times)
    return list(clearsky["ghi"].values), list(solpos["apparent_elevation"].values)


def _time_features(ts):
    """Cyclic time features in UTC (no DST jumps). Day of year instead of
    month, fractional hour instead of integer hour (no saw-tooth)."""
    ts_utc = pd.Timestamp(ts).tz_convert("UTC")
    hour = ts_utc.hour + ts_utc.minute / 60.0
    doy = ts_utc.dayofyear
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
    }


# --- Solcast -----------------------------------------------------------------
def parse_solcast(today_attr, tomorrow_attr):
    """Merged Solcast detailedForecast as a sorted list of
    (period_start, p50_w, p10_w, p90_w)."""
    rows = []
    for entry in (today_attr or []) + (tomorrow_attr or []):
        try:
            start = entry["period_start"]
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            rows.append((
                start,
                float(entry["pv_estimate"]) * KW_TO_W,
                float(entry.get("pv_estimate10", entry["pv_estimate"])) * KW_TO_W,
                float(entry.get("pv_estimate90", entry["pv_estimate"])) * KW_TO_W,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda r: r[0])
    return rows


def current_slot_index(rows, now):
    """Index of the Solcast slot that contains `now`, or None."""
    for i, (start, *_rest) in enumerate(rows):
        if start <= now < start + SLOT:
            return i
    return None


# --- Day-ahead features (identical for training and forecast) ---------------
def dayahead_features(ts, p50, p10, p90, clear_sky_ghi, elevation):
    x = _time_features(ts)
    x["solcast_p50"] = p50
    x["solcast_p10"] = p10
    x["solcast_p90"] = p90
    x["solcast_spread"] = p90 - p10
    x["clear_sky_ghi"] = clear_sky_ghi
    x["elevation"] = elevation
    # scale-free: how much of the clear-sky potential Solcast expects
    x["solcast_vs_clearsky"] = p50 / max(clear_sky_ghi, 1.0)
    return x


# --- Models ------------------------------------------------------------------
class _PickledRegressor:
    """River's ARFRegressor returns 0.0 (not None) before it has learned
    anything. predict() therefore returns None until `min_samples` daytime
    samples were learned, so callers can fall back instead of sending 0 W."""

    def __init__(self, path, n_models, min_samples):
        self.path = path
        self.min_samples = min_samples
        self.model = forest.ARFRegressor(n_models=n_models, seed=42)
        self.mae = metrics.MAE()
        self.n_learned = 0
        if path.exists():
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self.model = state["model"]
                self.mae = state["mae"]
                self.n_learned = state.get("n_learned", 0)
                log.info("%s loaded, MAE %.1f W, %d samples", path.name, self.mae.get(), self.n_learned)
            except Exception:
                log.exception("Could not load %s, starting fresh", path)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"model": self.model, "mae": self.mae, "n_learned": self.n_learned}, f)

    def predict(self, x):
        if self.n_learned < self.min_samples:
            return None
        y = self.model.predict_one(x)
        return None if y is None else min(max(y, 0.0), PLAUSIBLE_MAX_W)

    def learn(self, x, y):
        y_pred = self.model.predict_one(x)
        if y_pred is not None:
            self.mae.update(y, y_pred)
        self.model.learn_one(x, y)
        self.n_learned += 1

    @property
    def current_mae(self):
        return self.mae.get()


class DayAheadModel(_PickledRegressor):
    def __init__(self):
        super().__init__(DAYAHEAD_MODEL_PATH, n_models=10, min_samples=MIN_SAMPLES_DAYAHEAD)

    def predict_or_solcast(self, x):
        """Fallback to raw Solcast P50 while the model has not learned yet."""
        if x["elevation"] <= 0:
            return 0.0
        y = self.predict(x)
        return x["solcast_p50"] if y is None else y


class NowcastModel(_PickledRegressor):
    def __init__(self):
        super().__init__(NOWCAST_MODEL_PATH, n_models=5, min_samples=MIN_SAMPLES_NOWCAST)
        self.csi_history = deque(maxlen=CSI_WINDOW)

    def features(self, dayahead_now, irradiance, clear_sky_ghi, elevation):
        csi = min(irradiance / max(clear_sky_ghi, 1.0), 1.5)
        self.csi_history.append(csi)
        hist = list(self.csi_history)
        mean = sum(hist) / len(hist)
        std = math.sqrt(sum((c - mean) ** 2 for c in hist) / len(hist))
        return {
            "dayahead_now": dayahead_now,
            "irradiance": irradiance,
            "measured_csi": csi,
            "csi_mean_30min": mean,
            "csi_std_30min": std,   # high = broken clouds
            "elevation": elevation,
        }


def nowcast_factor(nowcast_now, dayahead_now):
    """Ratio nowcast / day-ahead for the current slot, clamped. 1.0 = no correction."""
    if nowcast_now is None or dayahead_now < NOWCAST_MIN_BASE_W:
        return 1.0
    lo, hi = NOWCAST_FACTOR_BOUNDS
    return min(max(nowcast_now / dayahead_now, lo), hi)


def blend_curve(slot_starts, dayahead_values, now, factor):
    """Apply the nowcast factor to the day-ahead curve, fading out with
    exp(-dt/tau): full effect now, ~37 % after 1 h, ~5 % after 3 h."""
    out = []
    for start, value in zip(slot_starts, dayahead_values):
        dt_h = max((start - now).total_seconds() / 3600.0, 0.0)
        weight = math.exp(-dt_h / NOWCAST_TAU_H)
        out.append(max(value * (1.0 + (factor - 1.0) * weight), 0.0))
    return out