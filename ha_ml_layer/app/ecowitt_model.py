"""Weather-station model (ha-ml-layer): measured irradiance -> PV power.

Three jobs for a local irradiance sensor (e.g. a weather station):

1. Plant model ("potential PV power"): learns what the plant delivers at a
   given measured irradiance, sun elevation and sun azimuth (azimuth matters
   for arrays facing different directions, because the sensor is horizontal).
   Baseline = a single learned factor W per W/m2; the River model on top is
   only used while its rolling MAE beats that baseline. Learns only from
   samples that are not curtailed.

2. Stand-in for the actual power while the inverter curtails (full battery,
   export limit): the forecast models then learn from the potential instead
   of the curtailed power.

3. Fallback forecast for slots the forecast APIs do not cover (API down, no
   internet): clear-sky irradiance x cloudiness, where the cloudiness starts
   at the current measured clear-sky index and fades towards the typical
   value for that hour of day (persistence blended with climatology, the
   strongest simple benchmark beyond ~1 h).

The sensor is used relatively (clear-sky index) and calibrated by learning,
so a constant sensor offset (e.g. lux-based sensors) does not hurt.
"""
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd

from pv_model import _ResidualRegressor, _time_features

log = logging.getLogger("ecowitt_model")

PLANT_MODEL_PATH = Path("/data/pv_station_plant_model_v1.pkl")
STATE_PATH = Path("/data/pv_station_state_v1.json")

PLANT_MIN_SAMPLES = 300        # 5-min samples (~3-4 sunny days) before the ML part may replace the factor
PLANT_WINDOW = 600
RATIO_ALPHA = 0.02             # EMA weight for the W per W/m2 factor
RATIO_MIN_SAMPLES = 30         # factor usable after ~2-3 h of good sun
RATIO_MIN_ELEVATION = 15.0
RATIO_MIN_IRRADIANCE = 100.0
CSI_ALPHA = 0.05               # EMA weight per hour-of-day bucket
CSI_MIN_SAMPLES = 5
CSI_MAX = 1.5
FALLBACK_TAU_H = 2.0           # current cloudiness fades towards the typical value
REF_ALPHA = 0.05
REF_MIN_ELEVATION = 20.0


def csi(irradiance, clear_sky_ghi):
    return min(irradiance / max(clear_sky_ghi, 1.0), CSI_MAX)


def plant_features(irradiance, clear_sky_ghi, elevation, azimuth, ts):
    x = _time_features(ts)
    x["irradiance"] = irradiance
    x["clear_sky_ghi"] = clear_sky_ghi
    x["csi"] = csi(irradiance, clear_sky_ghi)
    x["elevation"] = elevation
    x["azimuth_sin"] = math.sin(math.radians(azimuth))
    x["azimuth_cos"] = math.cos(math.radians(azimuth))
    return x


class StationModel:
    def __init__(self, plant_path=PLANT_MODEL_PATH, state_path=STATE_PATH):
        self.state_path = state_path
        self.plant = _ResidualRegressor(plant_path, n_models=5, min_samples=PLANT_MIN_SAMPLES,
                                        window=PLANT_WINDOW)
        self.ratio = None          # W per W/m2
        self.n_ratio = 0
        self.csi_clim = {}         # "hour" -> [mean, n]
        self.ref_ratio = None      # station irradiance / reference irradiance
        self.n_ref = 0
        if state_path.exists():
            try:
                s = json.loads(state_path.read_text())
                self.ratio, self.n_ratio = s.get("ratio"), s.get("n_ratio", 0)
                self.csi_clim = s.get("csi_clim", {})
                self.ref_ratio, self.n_ref = s.get("ref_ratio"), s.get("n_ref", 0)
                log.info("Station model loaded: %.3f W per W/m2 (n=%d), %d hours with typical CSI",
                         self.ratio or 0.0, self.n_ratio, sum(1 for v in self.csi_clim.values() if v[1] >= CSI_MIN_SAMPLES))
            except Exception:
                log.exception("Could not load %s, starting fresh", state_path)

    # --- persistence ---
    def save(self):
        self.plant.save()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ratio": self.ratio, "n_ratio": self.n_ratio, "csi_clim": self.csi_clim,
                                   "ref_ratio": self.ref_ratio, "n_ref": self.n_ref}))
        os.replace(tmp, self.state_path)

    # --- plant model ---
    @property
    def ready(self):
        return self.ratio is not None and self.n_ratio >= RATIO_MIN_SAMPLES

    def _base(self, irradiance):
        return self.ratio * irradiance

    def potential(self, x):
        """Potential PV power in W for the features x, or None if not learned yet."""
        if not self.ready:
            return None
        if x["elevation"] <= 0:
            return 0.0
        base = self._base(x["irradiance"])
        y = self.plant.predict(x, base)
        return base if y is None else y

    def learn(self, x, actual):
        """One not-curtailed sample with measured irradiance and actual PV power."""
        irr = x["irradiance"]
        if x["elevation"] >= RATIO_MIN_ELEVATION and irr >= RATIO_MIN_IRRADIANCE:
            r = actual / irr
            self.ratio = r if self.ratio is None else (1 - RATIO_ALPHA) * self.ratio + RATIO_ALPHA * r
            self.n_ratio += 1
        if self.ratio is not None:
            self.plant.learn(x, actual, self._base(irr))

    # --- typical cloudiness per local hour ---
    def update_csi(self, local_hour, value):
        key = str(local_hour)
        mean, n = self.csi_clim.get(key, [None, 0])
        mean = value if mean is None else (1 - CSI_ALPHA) * mean + CSI_ALPHA * value
        self.csi_clim[key] = [mean, n + 1]

    def typical_csi(self, local_hour):
        mean, n = self.csi_clim.get(str(local_hour), [None, 0])
        if mean is not None and n >= CSI_MIN_SAMPLES:
            return mean
        learned = [m for m, k in self.csi_clim.values() if k >= CSI_MIN_SAMPLES]
        return sum(learned) / len(learned) if learned else None

    # --- fallback forecast ---
    def fallback(self, slot_mid, clear_sky_ghi, elevation, azimuth, local_hour, csi_now, dt_h):
        """PV power for one future slot from the station alone, or None."""
        if elevation <= 0:
            return 0.0
        if not self.ready:
            return None
        typical = self.typical_csi(local_hour)
        if typical is None and csi_now is None:
            return None
        if typical is None:
            k = csi_now
        elif csi_now is None:
            k = typical
        else:
            k = typical + (csi_now - typical) * math.exp(-max(dt_h, 0.0) / FALLBACK_TAU_H)
        irr = max(k, 0.0) * clear_sky_ghi
        return self.potential(plant_features(irr, clear_sky_ghi, elevation, azimuth, slot_mid))

    # --- reference sensor (calibration check only) ---
    def update_reference(self, irradiance, reference, elevation):
        if elevation >= REF_MIN_ELEVATION and irradiance > 50 and reference > 50:
            r = irradiance / reference
            self.ref_ratio = r if self.ref_ratio is None else (1 - REF_ALPHA) * self.ref_ratio + REF_ALPHA * r
            self.n_ref += 1


def local_hour(ts, tz):
    return pd.Timestamp(ts).tz_convert(tz).hour
