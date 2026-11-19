"""Correction of the own station's irradiance against a nearby reference station.

Many hobby stations (e.g. Ecowitt WH69/WS90) do not measure irradiance: they
measure illuminance (lux) and divide it by a fixed factor. Daylight under
clouds has a different spectrum than direct sun, so one factor cannot fit
both - a gain in the station's settings only moves the level. Backtest 2026
(Ecowitt vs GeoSphere Gumpoldskirchen, 4 km): one gain 32 W/m2 MAE, a ratio
table by clearness (own clear-sky index) and sun elevation 29 W/m2; hourly
correlation with the PV power 0.925 -> 0.943, daily PV error of the best
linear fit 22 % -> 18 % (reference station itself 9 %).

Learning: hourly means of both sensors (the reference updates less often),
ratio reference / own per cell (6 clearness x 3 elevation bins), median of
the last CELL_KEEP hours; cells with too few hours use the overall gain.
Every finished hour is first predicted and scored (raw, gain, table), then
learned - so the published errors are honest out-of-sample values. The
method with the lowest error over the last SCORE_KEEP hours is used.
Without the reference (internet down) the learned table keeps working.
"""
import json
import logging
import os
import statistics
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("irradiance_correction")

STATE_PATH = Path("/data/irradiance_correction_v1.json")
KT_BINS = [0.15, 0.3, 0.5, 0.7, 0.9]
ELEV_BINS = [10.0, 20.0]
MIN_ELEVATION = 5.0
MIN_W_M2 = 5.0
MIN_SAMPLES_HOUR = 6                # of 12 five-minute values
RATIO_CLIP = (0.3, 3.0)
CELL_KEEP = 60
CELL_MIN = 5
GAIN_KEEP = 300
READY_HOURS = 24
SCORE_KEEP = 336                    # ~4 weeks of daylight hours
SCORE_MIN = 48
HISTORY_DAYS = 60


def _bin(v, edges):
    return sum(v >= e for e in edges)


class IrradianceCorrection:
    def __init__(self, path=STATE_PATH):
        self.path = path
        self.cells = {}                       # "i,j" -> deque of ratios
        self.gains = deque(maxlen=GAIN_KEEP)
        self.scores = deque(maxlen=SCORE_KEEP)   # (err_raw, err_gain, err_table)
        self.hour = None                      # [hour epoch, sum own, sum ref, sum ghi, sum elev, n]
        self.history_key = None
        if path.exists():
            try:
                s = json.loads(path.read_text())
                self.cells = {k: deque(v, maxlen=CELL_KEEP) for k, v in s.get("cells", {}).items()}
                self.gains = deque(s.get("gains", []), maxlen=GAIN_KEEP)
                self.scores = deque([tuple(x) for x in s.get("scores", [])], maxlen=SCORE_KEEP)
                self.history_key = s.get("history_key")
            except Exception:
                log.exception("Irradiance correction state unreadable, starting empty")

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"cells": {k: list(v) for k, v in self.cells.items()}, "gains": list(self.gains),
                                   "scores": [list(x) for x in self.scores], "history_key": self.history_key}))
        os.replace(tmp, self.path)

    # --- model ---------------------------------------------------------------------------
    @property
    def ready(self):
        return len(self.gains) >= READY_HOURS

    def gain(self):
        return statistics.median(self.gains) if self.gains else None

    def factor(self, own, clear_sky_ghi, elevation, method="table"):
        if not self.ready or own is None or clear_sky_ghi is None or clear_sky_ghi <= 0:
            return None
        if method == "table":
            c = self.cells.get(f"{_bin(own / clear_sky_ghi, KT_BINS)},{_bin(elevation, ELEV_BINS)}")
            if c is not None and len(c) >= CELL_MIN:
                return statistics.median(c)
        return self.gain()

    def method(self):
        """raw / gain / table: lowest error over the scored hours (needs SCORE_MIN hours)."""
        if not self.ready or len(self.scores) < SCORE_MIN:
            return "raw"
        mae = self.mae()
        return min(("raw", "gain", "table"), key=lambda k: mae[k])

    def mae(self):
        if not self.scores:
            return {"raw": None, "gain": None, "table": None}
        n = len(self.scores)
        return {k: round(sum(s[i] for s in self.scores) / n, 1) for i, k in enumerate(("raw", "gain", "table"))}

    def corrected(self, own, clear_sky_ghi, elevation):
        """Best available value of the own sensor (W/m2), the raw value while not learned."""
        if own is None:
            return None
        if elevation is None or elevation <= 0:
            return own
        m = self.method()
        f = None if m == "raw" else self.factor(own, clear_sky_ghi, elevation, m)
        return own if f is None else own * f

    # --- learning ------------------------------------------------------------------------
    def add(self, ts, own, reference, clear_sky_ghi, elevation):
        """Five-minute sample; learns when an hour (UTC) is complete."""
        h = int(ts.timestamp() // 3600 * 3600)
        if self.hour is not None and self.hour[0] != h:
            self._finish_hour()
            self.hour = None
        if own is None or reference is None or own < 0 or reference < 0 or elevation is None or clear_sky_ghi is None:
            return
        if self.hour is None:
            self.hour = [h, 0.0, 0.0, 0.0, 0.0, 0]
        x = self.hour
        x[1] += own; x[2] += reference; x[3] += clear_sky_ghi; x[4] += elevation; x[5] += 1

    def _finish_hour(self):
        _, so, sr, sg, se, n = self.hour
        if n < MIN_SAMPLES_HOUR:
            return
        own, ref, ghi, elev = float(so / n), float(sr / n), float(sg / n), float(se / n)
        if elev < MIN_ELEVATION or own < MIN_W_M2 or ref < MIN_W_M2 or ghi <= 0:
            return
        if self.ready:                       # score before learning = out of sample
            g, t = self.factor(own, ghi, elev, "gain"), self.factor(own, ghi, elev, "table")
            self.scores.append((round(abs(own - ref), 2), round(abs(own * g - ref), 2), round(abs(own * t - ref), 2)))
        r = min(max(ref / own, RATIO_CLIP[0]), RATIO_CLIP[1])
        self.gains.append(round(r, 4))
        key = f"{_bin(own / ghi, KT_BINS)},{_bin(elev, ELEV_BINS)}"
        self.cells.setdefault(key, deque(maxlen=CELL_KEEP)).append(round(r, 4))

    def backfill(self, influx, own_entity, reference_entity, now=None):
        """Learn once from InfluxDB history (both sensors, read-only)."""
        key = f"{own_entity}|{reference_entity}"
        if self.history_key == key:
            return
        from history import _buckets
        from pv_model import solar_position
        end = now or datetime.now(timezone.utc)
        start = (end - timedelta(days=HISTORY_DAYS)).replace(minute=0, second=0, microsecond=0)
        own = _buckets(influx, own_entity, start, end, max_carry=1800)
        ref = _buckets(influx, reference_entity, start, end, max_carry=1800)
        self.cells, self.gains, self.scores, self.hour = {}, deque(maxlen=GAIN_KEEP), deque(maxlen=SCORE_KEEP), None
        epochs = sorted(set(own) & set(ref))
        if epochs:
            ghi, elev, _ = solar_position([datetime.fromtimestamp(t, timezone.utc) for t in epochs])
            for t, g, e in zip(epochs, ghi, elev):
                self.add(datetime.fromtimestamp(t, timezone.utc), own.get(t), ref.get(t), g, e)
            if self.hour is not None:
                self._finish_hour()
                self.hour = None
        self.history_key = key
        self.save()
        log.info("Irradiance correction: %d hours learned from InfluxDB, %d scored, MAE W/m2 %s -> used: %s",
                 len(self.gains), len(self.scores), self.mae(), self.method())

    # --- output --------------------------------------------------------------------------
    def attributes(self):
        return {"method": self.method(), "gain": None if self.gain() is None else round(self.gain(), 3),
                "learned_hours": len(self.gains), "scored_hours": len(self.scores),
                "mae_raw_w_m2": self.mae()["raw"], "mae_gain_w_m2": self.mae()["gain"],
                "mae_table_w_m2": self.mae()["table"],
                "table": {k: round(statistics.median(v), 3) for k, v in sorted(self.cells.items()) if len(v) >= CELL_MIN}}
