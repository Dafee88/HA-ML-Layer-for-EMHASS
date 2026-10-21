"""PV forecast correction (ha-ml-layer).

Two models, rebuilt 14.10.2026:

1. Day-ahead model (DayAheadModel)
   Uses ONLY features that also exist for future time steps: Solcast P50/P10/P90,
   clear-sky GHI, solar elevation and cyclic time. Training and forecast use the
   SAME feature builder, so the curve sent to EMHASS is computed exactly the way
   the model was trained. (Before: training saw live weather-station data, the
   forecast path always assumed a clear sky -> systematic over-forecast on
   cloudy afternoons.)

2. Nowcast model (NowcastModel)
   Corrects only the CURRENT value with live measured irradiance (clear-sky index
   and its short-term variability). Its effect on the curve fades out over the
   next ~2 hours (smart persistence), because a sensor cannot see clouds that
   have not arrived yet.

Learning scheme (model files v4):
  - Both models learn the RESIDUAL on top of a baseline (day-ahead: Solcast P50,
    nowcast: day-ahead value) instead of absolute watts. Trees cannot
    extrapolate; a residual keeps the baseline's shape and falls back to it
    naturally.
  - The day-ahead model learns once per 30-min slot on the slot-MEAN actual
    power (SlotAccumulator) -- the same quantity Solcast forecasts and EMHASS
    consumes -- with solar geometry at the slot midpoint.
  - A model is only used while its rolling MAE beats its baseline's rolling MAE
    on the same samples, so the ML layer can never make the forecast worse than
    raw Solcast for long.

Input sources are independent of EMHASS: Solcast directly, never
sensor.p_pv_forecast (that is EMHASS' own output and would become the ML
curve itself once EMHASS receives it -> the model would learn from its own
prediction).
"""
import logging
import math
import os
import pickle
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pvlib
from river import forest, metrics, preprocessing, utils

log = logging.getLogger("pv_model")

# --- Output entities (input entities come from the add-on options) ---------
ENTITY_OUTPUT_CORRECTED = "sensor.pv_ml_corrected_forecast"               # current value (day-ahead x nowcast)
ENTITY_OUTPUT_CURVE = "sensor.pv_ml_corrected_forecast_curve"             # 48 x 30 min, read by the EMHASS rest_commands
# Solcast detailedForecast: pv_estimate/10/90 in kW (30-min average power)
# -----------------------------------------------------------------------------

# v4: + second forecast provider features; older pickles are incompatible
DAYAHEAD_MODEL_PATH = Path("/data/pv_dayahead_model_v4.pkl")
NOWCAST_MODEL_PATH = Path("/data/pv_nowcast_model_v4.pkl")

SLOT = timedelta(minutes=30)
SLOT_MID = SLOT / 2            # solar geometry at the slot midpoint (Solcast value = slot mean)
HORIZON_SLOTS = 48
KW_TO_W = 1000.0               # Solcast pv_estimate is kW (sum over a day = 2 x daily kWh, checked 14.10.2026)
PLAUSIBLE_MAX_W = 15000
MIN_ELEVATION_LEARN = 5.0      # degrees; below that PV is ~0 and only adds noise
NOWCAST_TAU_H = 1.0            # nowcast influence decays with exp(-dt/tau)
NOWCAST_FACTOR_BOUNDS = (0.1, 2.0)
NOWCAST_MIN_BASE_W = 100.0     # below this the ratio nowcast/day-ahead is meaningless
CSI_WINDOW = 6                 # cycles for the variability feature (6 x 5 min = 30 min)
MIN_SAMPLES_DAYAHEAD = 60      # slots (~3-4 days of daylight) before the model may replace raw Solcast
MIN_SAMPLES_NOWCAST = 150      # 5-min samples (~1-2 days) before the irradiance nowcast is allowed to correct
ROLLING_DAYAHEAD = 300         # slots (~2 weeks) for the model-vs-Solcast comparison
ROLLING_NOWCAST = 600          # 5-min samples (~4-5 days) for the nowcast-vs-day-ahead comparison
MIN_SLOT_COVERAGE = 0.6        # share of a slot that must have actual readings to learn on it
CSI_MAX_GAP = timedelta(minutes=20)  # older CSI history (e.g. yesterday evening) is dropped

_LOCATION = None  # set by set_location() at start-up


def set_location(latitude, longitude, tz):
    """Site for clear-sky and sun position (from the add-on options)."""
    global _LOCATION
    _LOCATION = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=tz)


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


def solar_position(timestamps):
    """Like solar_geometry, plus the solar azimuth (degrees from north)."""
    times = pd.DatetimeIndex(timestamps)
    if times.tz is None:
        times = times.tz_localize("UTC")
    clearsky = _LOCATION.get_clearsky(times, model="ineichen")
    solpos = _LOCATION.get_solarposition(times)
    return (list(clearsky["ghi"].values), list(solpos["apparent_elevation"].values),
            list(solpos["azimuth"].values))


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


def daily_kwh(rows, tz):
    """Sum of the P50 slots per local date in kWh (30-min slots of W)."""
    out = {}
    for start, p50, _p10, _p90 in rows:
        d = pd.Timestamp(start).tz_convert(tz).date()
        out[d] = out.get(d, 0.0) + p50 * 0.5 / 1000.0
    return out


def current_slot_index(rows, now):
    """Index of the Solcast slot that contains `now`, or None."""
    for i, (start, *_rest) in enumerate(rows):
        if start <= now < start + SLOT:
            return i
    return None


# --- Day-ahead features (identical for training and forecast) ---------------
def dayahead_features(ts, p50, p10, p90, clear_sky_ghi, elevation, day=None):
    """`day`: optional daily context for the slot's local day, e.g.
    {"solcast_day_kwh": .., "secondary_day_kwh": ..} (second forecast provider).
    Missing keys are simply left out (River handles absent features)."""
    x = _time_features(ts)
    x["solcast_p50"] = p50
    x["solcast_p10"] = p10
    x["solcast_p90"] = p90
    x["solcast_spread"] = p90 - p10
    x["clear_sky_ghi"] = clear_sky_ghi
    x["elevation"] = elevation
    # scale-free: how much of the clear-sky potential Solcast expects
    x["solcast_vs_clearsky"] = p50 / max(clear_sky_ghi, 1.0)
    if day:
        sc, sec = day.get("solcast_day_kwh"), day.get("secondary_day_kwh")
        if sec is not None:
            x["secondary_day_kwh"] = sec
            if sc:
                # >1: the second provider expects a sunnier day than Solcast
                x["secondary_vs_solcast"] = min(sec / sc, 5.0)
    return x


# --- Models ------------------------------------------------------------------
def _clip_w(y):
    return min(max(y, 0.0), PLAUSIBLE_MAX_W)


class _ResidualRegressor:
    """ARFRegressor on the residual y - base.

    predict() returns None -- so callers keep the baseline -- until `min_samples`
    were learned and while the rolling MAE of the model is not lower than the
    rolling MAE of the baseline on the same (prequential) samples."""

    def __init__(self, path, n_models, min_samples, window):
        self.path = path
        self.min_samples = min_samples
        # StandardScaler is required: ARFRegressor's default leaf_prediction
        # "adaptive" fits linear models in the leaves, and on unscaled watt-range
        # features they diverge (seen live: single predictions of several MW;
        # reproduced offline).
        self.model = preprocessing.StandardScaler() | forest.ARFRegressor(n_models=n_models, seed=42)
        self.mae = metrics.MAE()
        self.mae_rolling = utils.Rolling(metrics.MAE(), window_size=window)
        self.mae_base_rolling = utils.Rolling(metrics.MAE(), window_size=window)
        self.n_learned = 0
        if path.exists():
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                self.model = state["model"]
                self.mae = state["mae"]
                self.mae_rolling = state["mae_rolling"]
                self.mae_base_rolling = state["mae_base_rolling"]
                self.n_learned = state.get("n_learned", 0)
                log.info("%s loaded, rolling MAE %.1f W (baseline %.1f W), %d samples",
                         path.name, self.mae_rolling.get(), self.mae_base_rolling.get(), self.n_learned)
            except Exception:
                log.exception("Could not load %s, starting fresh", path)

    def save(self):
        """Atomic write: a crash mid-dump must not destroy the learned model."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"model": self.model, "mae": self.mae, "mae_rolling": self.mae_rolling,
                         "mae_base_rolling": self.mae_base_rolling, "n_learned": self.n_learned}, f)
        os.replace(tmp, self.path)

    @property
    def active(self):
        return (self.n_learned >= self.min_samples
                and self.mae_rolling.get() < self.mae_base_rolling.get())

    def _raw(self, x, base):
        return _clip_w(base + (self.model.predict_one(x) or 0.0))

    def predict(self, x, base):
        return self._raw(x, base) if self.active else None

    def learn(self, x, y, base):
        if self.n_learned > 0:
            y_pred = self._raw(x, base)
            self.mae.update(y, y_pred)
            self.mae_rolling.update(y, y_pred)
            self.mae_base_rolling.update(y, base)
        self.model.learn_one(x, y - base)
        self.n_learned += 1

    @property
    def current_mae(self):
        return self.mae_rolling.get()

    @property
    def baseline_mae(self):
        return self.mae_base_rolling.get()


class DayAheadModel(_ResidualRegressor):
    """Baseline = Solcast P50 of the slot."""

    def __init__(self, path=DAYAHEAD_MODEL_PATH):
        super().__init__(path, n_models=10, min_samples=MIN_SAMPLES_DAYAHEAD,
                         window=ROLLING_DAYAHEAD)

    def predict_or_solcast(self, x):
        """Fallback to raw Solcast P50 while the model is not (yet) better."""
        if x["elevation"] <= 0:
            return 0.0
        y = self.predict(x, x["solcast_p50"])
        return x["solcast_p50"] if y is None else y

    def learn_slot(self, x, y_mean):
        self.learn(x, y_mean, x["solcast_p50"])


class NowcastModel(_ResidualRegressor):
    """Baseline = day-ahead value of the current slot."""

    def __init__(self):
        super().__init__(NOWCAST_MODEL_PATH, n_models=5, min_samples=MIN_SAMPLES_NOWCAST,
                         window=ROLLING_NOWCAST)
        self.csi_history = deque(maxlen=CSI_WINDOW)
        self._csi_last_ts = None

    def features(self, now, dayahead_now, irradiance, clear_sky_ghi, elevation):
        if self._csi_last_ts is not None and now - self._csi_last_ts > CSI_MAX_GAP:
            self.csi_history.clear()
        self._csi_last_ts = now
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


class SlotAccumulator:
    """Averages the actual readings over one Solcast slot. The day-ahead model
    then learns ONCE per slot on the slot mean -- the quantity Solcast forecasts
    and EMHASS consumes -- instead of six noisy instantaneous samples that all
    share identical features (which over-weighted noise six-fold)."""

    def __init__(self, min_readings):
        self.min_readings = min_readings
        self.start = None
        self.x = None
        self.total = 0.0
        self.n = 0

    def add(self, start, x, actual):
        """Record one reading for slot `start`. Returns (x, mean_actual) of the
        previous slot once it is complete and covered well enough, else None."""
        done = None
        if start != self.start:
            if self.start is not None and self.n >= self.min_readings:
                done = (self.x, self.total / self.n)
            self.start, self.total, self.n = start, 0.0, 0
        self.x = x  # latest Solcast values for this slot
        if actual is not None:
            self.total += actual
            self.n += 1
        return done


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