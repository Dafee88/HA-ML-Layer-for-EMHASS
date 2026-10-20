"""Optional InfluxDB 1.x writer (HTTP /write, line protocol).

Purpose: keep the data a real backtest needs, which Home Assistant's recorder
does not keep (Solcast's 30-min curves are not recorded):
  - ml_layer_pv           every cycle: Solcast P50/P10/P90 of the current slot,
                          model values, actual power, Ecowitt, MAE
  - ml_layer_pv_slot      once per finished 30-min slot: mean actual power and
                          the Solcast values the day-ahead model learned on
  - ml_layer_dayahead     once per day (evening): tomorrow's 48 Solcast slots
                          and the ML curve, i.e. what EMHASS plans with

A failing InfluxDB never stops the add-on: errors are logged (at most once per
10 minutes per error text) and the points are dropped.
"""
import logging
import time

import requests

log = logging.getLogger("influx")

_ERROR_LOG_INTERVAL_S = 600


def _escape_key(s):
    return str(s).replace("\\", "\\\\").replace(",", r"\,").replace(" ", r"\ ").replace("=", r"\=")


def _field_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}i"
    if isinstance(v, float):
        return repr(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def line(measurement, fields, tags=None, ts=None):
    """One line-protocol line. None fields are skipped; returns None if no field is left."""
    f = {k: v for k, v in fields.items() if v is not None}
    if not f:
        return None
    head = _escape_key(measurement)
    for k, v in sorted((tags or {}).items()):
        if v is not None and v != "":
            head += f",{_escape_key(k)}={_escape_key(v)}"
    body = ",".join(f"{_escape_key(k)}={_field_value(v)}" for k, v in f.items())
    out = f"{head} {body}"
    if ts is not None:
        out += f" {int(ts.timestamp())}"
    return out


class InfluxWriter:
    def __init__(self, host, port, database, username=None, password=None, ssl=False, timeout=10):
        scheme = "https" if ssl else "http"
        self.url = f"{scheme}://{host}:{port}/write"
        self.database = database
        self.auth = (username, password) if username else None
        self.timeout = timeout
        self._last_error = {}
        self.ok_count = 0

    def describe(self):
        return f"{self.url} db={self.database} user={'set' if self.auth else 'none'}"

    def write(self, lines):
        lines = [ln for ln in lines if ln]
        if not lines:
            return True
        try:
            r = requests.post(
                self.url,
                params={"db": self.database, "precision": "s"},
                data="\n".join(lines).encode("utf-8"),
                auth=self.auth,
                timeout=self.timeout,
            )
            if r.status_code == 204:
                self.ok_count += 1
                return True
            self._error(f"HTTP {r.status_code}: {r.text.strip()[:200]}")
        except requests.RequestException as exc:
            self._error(f"{type(exc).__name__}: {exc}")
        return False

    def _error(self, text):
        now = time.monotonic()
        if now - self._last_error.get(text, -1e9) >= _ERROR_LOG_INTERVAL_S:
            self._last_error[text] = now
            log.warning("InfluxDB write failed (%s): %s", self.describe(), text)
