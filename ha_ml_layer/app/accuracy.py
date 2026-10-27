"""Daily check of the forecast quality (sensor.pv_ml_forecast_accuracy).

Every evening at the snapshot hour (default 20:00, before the EMHASS
day-ahead run) the add-on notes tomorrow's forecast in kWh -- ML and raw
Solcast, i.e. what EMHASS plans with. After midnight the day is closed with
the measured PV energy (sum of the readings of every cycle) and both errors
are published. Days with too few readings are not scored.
"""
import json
import logging
import os
from datetime import date
from pathlib import Path

log = logging.getLogger("accuracy")

STATE_PATH = Path("/data/pv_accuracy_v1.json")
ENTITY = "sensor.pv_ml_forecast_accuracy"
KEEP_DAYS = 30
WINDOW_DAYS = 7
MIN_COVERAGE = 0.9        # share of the day that must be covered by readings


class DailyAccuracy:
    def __init__(self, path=STATE_PATH):
        self.path = path
        self.forecasts = {}       # "YYYY-MM-DD" -> {"ml_kwh", "solcast_kwh", "source"}
        self.day = None           # running day "YYYY-MM-DD"
        self.energy_wh = 0.0
        self.covered_s = 0.0
        self.curtailed_s = 0.0
        self.low_output_s = 0.0
        self.last_ts = None
        self.history = []         # list of scored days (dicts), newest last
        if path.exists():
            try:
                s = json.loads(path.read_text())
                self.forecasts = s.get("forecasts", {})
                self.day = s.get("day")
                self.energy_wh = s.get("energy_wh", 0.0)
                self.covered_s = s.get("covered_s", 0.0)
                self.curtailed_s = s.get("curtailed_s", 0.0)
                self.low_output_s = s.get("low_output_s", 0.0)
                self.history = s.get("history", [])
            except Exception:
                log.exception("Could not load %s, starting fresh", path)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "forecasts": self.forecasts, "day": self.day, "energy_wh": self.energy_wh,
            "covered_s": self.covered_s, "curtailed_s": self.curtailed_s,
            "low_output_s": self.low_output_s, "history": self.history}))
        os.replace(tmp, self.path)

    def note_forecast(self, day, ml_kwh, solcast_kwh, source):
        """Tomorrow's forecast as known in the evening (first call per day wins)."""
        key = day.isoformat()
        if key not in self.forecasts and ml_kwh is not None:
            self.forecasts[key] = {"ml_kwh": round(ml_kwh, 3),
                                   "solcast_kwh": None if solcast_kwh is None else round(solcast_kwh, 3),
                                   "source": source}

    def add(self, now, local_day, actual_w, max_gap_s, curtailed=False, low_output=False):
        """One cycle. Integrates the actual power; closes the previous day at midnight."""
        key = local_day.isoformat()
        if self.day != key:
            if self.day is not None:
                self._close()
            self.day, self.energy_wh, self.covered_s = key, 0.0, 0.0
            self.curtailed_s = self.low_output_s = 0.0
            self.last_ts = None
        if self.last_ts is not None and actual_w is not None:
            dt = (now - self.last_ts).total_seconds()
            if 0 < dt <= max_gap_s:
                self.energy_wh += actual_w * dt / 3600.0
                self.covered_s += dt
                if curtailed:
                    self.curtailed_s += dt
                if low_output:
                    self.low_output_s += dt
        self.last_ts = now if actual_w is not None else None

    def _close(self):
        fc = self.forecasts.pop(self.day, None)
        coverage = self.covered_s / 86400.0
        if fc is None or coverage < MIN_COVERAGE:
            log.info("Accuracy: %s not scored (forecast noted: %s, coverage %.0f %%)",
                     self.day, fc is not None, coverage * 100)
            return
        actual = self.energy_wh / 1000.0
        rec = {"date": self.day, "actual_kwh": round(actual, 2), "ml_kwh": round(fc["ml_kwh"], 2),
               "solcast_kwh": None if fc["solcast_kwh"] is None else round(fc["solcast_kwh"], 2),
               "ml_source": fc.get("source"),
               "curtailed_h": round(self.curtailed_s / 3600.0, 1),
               "low_output_h": round(self.low_output_s / 3600.0, 1)}
        rec["ml_error_kwh"] = round(rec["ml_kwh"] - actual, 2)
        rec["solcast_error_kwh"] = None if rec["solcast_kwh"] is None else round(rec["solcast_kwh"] - actual, 2)
        self.history = (self.history + [rec])[-KEEP_DAYS:]
        log.info("Accuracy: %s actual %.2f kWh | ML %.2f kWh (%+.2f) | Solcast %s kWh (%s) | curtailed %.1f h",
                 self.day, actual, rec["ml_kwh"], rec["ml_error_kwh"], rec["solcast_kwh"],
                 rec["solcast_error_kwh"], rec["curtailed_h"])
        # forecasts for days that can no longer be scored
        today = date.fromisoformat(self.day)
        self.forecasts = {k: v for k, v in self.forecasts.items() if date.fromisoformat(k) > today}

    @staticmethod
    def _pct(err, actual):
        return None if err is None or actual <= 0.5 else round(100.0 * err / actual, 1)

    def publish(self, ha):
        last = self.history[-1] if self.history else None
        win = self.history[-WINDOW_DAYS:]

        def mean_abs(key):
            vals = [abs(r[key]) for r in win if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def mean_abs_pct(key):
            vals = [abs(r[key]) / r["actual_kwh"] * 100.0 for r in win
                    if r.get(key) is not None and r["actual_kwh"] > 0.5]
            return round(sum(vals) / len(vals), 1) if vals else None

        state = "unknown" if last is None else self._pct(last["ml_error_kwh"], last["actual_kwh"])
        ha.set_state(ENTITY, "unknown" if state is None else state, attributes={
            "unit_of_measurement": "%",
            "friendly_name": "PV Forecast Accuracy (ML vs actual)",
            "date": last["date"] if last else None,
            "actual_kwh": last["actual_kwh"] if last else None,
            "ml_kwh": last["ml_kwh"] if last else None,
            "solcast_kwh": last["solcast_kwh"] if last else None,
            "ml_error_kwh": last["ml_error_kwh"] if last else None,
            "solcast_error_kwh": last["solcast_error_kwh"] if last else None,
            "solcast_error_pct": self._pct(last["solcast_error_kwh"], last["actual_kwh"]) if last else None,
            "curtailed_h": last["curtailed_h"] if last else None,
            "low_output_h": last["low_output_h"] if last else None,
            "days": len(win),
            "mean_abs_error_ml_kwh": mean_abs("ml_error_kwh"),
            "mean_abs_error_solcast_kwh": mean_abs("solcast_error_kwh"),
            "mean_abs_error_ml_pct": mean_abs_pct("ml_error_kwh"),
            "mean_abs_error_solcast_pct": mean_abs_pct("solcast_error_kwh"),
            "history": win,
            "source": "ha-ml-layer / forecast noted at the snapshot hour vs measured PV energy",
        })
