"""Learn the day-ahead model from InfluxDB history (once, at start-up).

Home Assistant's InfluxDB integration stores every state change of the
configured sensors, including the Solcast `detailedForecast` attribute (as a
string field). From that history the day-ahead model can learn weeks of slots
at once instead of waiting ~60 live slots.

What is rebuilt, in time order, into a FRESH model:
  1. history slots before the add-on's own records: for every 30-min slot the
     latest Solcast "today" curve stored before the slot ended (the same
     values the live model learns on) and the slot-mean actual PV power;
  2. the add-on's own slot records (measurement ml_layer_pv_slot), i.e. the
     samples the live model already learned on.
The rolling MAE gate is computed prequentially over all samples, exactly as
live, so the model only replaces raw Solcast if it really is better.

Curtailed slots are skipped (never learned as "weak sun"): slots where the
curtailment indicator was on, and -- with a weather station -- slots where the
actual power is far below the station's potential power.

The old model file is kept as *.pre_history.pkl. A marker file makes sure
this runs only once per model version.
"""
import ast
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from ecowitt_model import plant_features
from pv_model import (
    DAYAHEAD_MODEL_PATH,
    MIN_ELEVATION_LEARN,
    PLAUSIBLE_MAX_W,
    SLOT,
    SLOT_MID,
    DayAheadModel,
    dayahead_features,
    parse_solcast,
    solar_position,
)

log = logging.getLogger("history")

MARKER_PATH = Path("/data/dayahead_history_v2.json")   # v2: only history after the last Solcast site change
BACKUP_PATH = DAYAHEAD_MODEL_PATH.with_suffix(".pre_history.pkl")
MAX_DAYS = 90                  # older history is not used
BUCKET = timedelta(minutes=5)
MIN_BUCKETS = 4                # of 6 five-minute buckets per slot with a reading
CURTAIL_MIN_POTENTIAL_W = 300.0
CURTAIL_RATIO = 0.6            # actual < 60 % of the station potential -> treated as curtailed

_SITE_FIELD_RE = re.compile(r"^[0-9a-f]{4}(_[0-9a-f]{4}){3}$")   # Solcast per-site attribute (resource id)
_DT_RE = re.compile(
    r"datetime\.datetime\((\d+(?:,\s*\d+){2,6})"
    r"(?:,\s*tzinfo=(zoneinfo\.ZoneInfo\(key='[^']+'\)|datetime\.timezone\.utc|[A-Za-z_.]+\([^()]*\)|[A-Za-z_.]+))?\)")
_TZ_KEY_RE = re.compile(r"key='([^']+)'")


def _dt_literal(match, default_tz):
    parts = [int(p) for p in match.group(1).split(",")]
    tz_text = match.group(2) or ""
    if not tz_text:
        tz = default_tz
    elif "ZoneInfo" in tz_text:
        tz = ZoneInfo(_TZ_KEY_RE.search(tz_text).group(1))
    elif tz_text.endswith("utc") or "UTC" in tz_text:
        tz = timezone.utc
    else:
        return "None"          # unknown time zone object -> entry dropped
    return repr(datetime(*parts, tzinfo=tz).isoformat())


def parse_forecast_field(text, default_tz):
    """Solcast detailedForecast as stored by HA's InfluxDB integration: a JSON
    list or a Python repr with datetime objects. Returns a list of dicts with
    ISO period_start strings (unparsable entries dropped)."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        try:
            data = ast.literal_eval(_DT_RE.sub(lambda m: _dt_literal(m, default_tz), text))
        except (ValueError, SyntaxError, AttributeError, TypeError):
            return []
    return [e for e in data if isinstance(e, dict) and e.get("period_start")] if isinstance(data, list) else []


def _q(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ident(s):
    return '"' + str(s).replace('"', '\\"') + '"'


def _obj(entity_id):
    return entity_id.split(".", 1)[-1]


_META_CACHE = {}


def _meta(influx, entity_id, field_pred):
    """(measurement, field) of the first measurement with a matching field."""
    if entity_id not in _META_CACHE:
        _META_CACHE[entity_id] = influx.inventory(entity_id) or []
    inv = _META_CACHE[entity_id]
    for m in inv:
        for f in m["fields"]:
            if field_pred(f):
                return m["measurement"], f
    return None


def _buckets(influx, entity_id, start, end):
    """{bucket_start_epoch: mean value} of 5-min buckets, last value carried forward."""
    meta = _meta(influx, entity_id, lambda f: f == "value")
    if meta is None:
        return {}
    meas, field = meta
    # carry the last value from before the window into it
    series = influx.query(
        f"SELECT mean({_ident(field)}) FROM {_ident(meas)} WHERE entity_id = {_q(_obj(entity_id))} "
        f"AND time >= {int(start.timestamp())}s AND time < {int(end.timestamp())}s "
        f"GROUP BY time(5m) fill(previous)")
    out = {}
    for s in series or []:
        for t, v in s.get("values", []):
            if v is not None:
                out[int(t)] = float(v)
    return out


def _slot_mean(buckets, slot_start):
    t0 = int(slot_start.timestamp())
    vals = [buckets.get(t0 + i * int(BUCKET.total_seconds())) for i in range(int(SLOT / BUCKET))]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if len(vals) >= MIN_BUCKETS else None


def _solcast_config_start(influx, entity_id, meas):
    """Time of the last Solcast configuration change seen in the history.

    The Solcast integration adds one attribute per rooftop site (named after
    the site's resource id). A site appearing or disappearing means the
    forecast was produced for a different plant -- older history must not be
    learned (it would teach a bias that no longer exists)."""
    series = influx.query(f"SHOW FIELD KEYS FROM {_ident(meas)}")
    keys = [row[0] for s in series or [] for row in s.get("values", []) if _SITE_FIELD_RE.match(str(row[0]))]
    spans = {}
    for k in keys:
        span = []
        for fn in ("first", "last"):
            res = influx.query(f"SELECT {fn}({_ident(k)}) FROM {_ident(meas)} WHERE entity_id = {_q(_obj(entity_id))}")
            vals = [row[0] for s in res or [] for row in s.get("values", [])]
            span.append(int(vals[0]) if vals else None)
        if span[0] is not None:
            spans[k] = span
    if not spans:
        return None, []
    newest = max(last for _first, last in spans.values())
    start = None
    for first, last in spans.values():
        t = first if last >= newest - 86400 else last      # site added -> its first value; removed -> its last value
        start = t if start is None else max(start, t)
    return datetime.fromtimestamp(start, timezone.utc), sorted(spans)


def _first_own_slot(influx):
    series = influx.query("SELECT first(target_mean_w) FROM ml_layer_pv_slot")
    if series is None:
        return None, False
    for s in series:
        for row in s.get("values", []):
            return datetime.fromtimestamp(int(row[0]), timezone.utc), True
    return None, True


def _history_samples(influx, cfg, station, since, until, stats):
    tz = ZoneInfo(cfg.timezone)
    meta = _meta(influx, cfg.pv_solcast_today, lambda f: "detailedforecast" in f.lower())
    if meta is None:
        stats["reason"] = "no Solcast detailedForecast history in InfluxDB"
        return []
    meas, field = meta
    samples = []
    day = since.astimezone(tz).date()
    last_day = until.astimezone(tz).date()
    while day <= last_day:
        d0 = datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(timezone.utc)
        d1 = d0 + timedelta(days=1)
        series = influx.query(
            f"SELECT {_ident(field)} FROM {_ident(meas)} WHERE entity_id = {_q(_obj(cfg.pv_solcast_today))} "
            f"AND time >= {int((d0 - timedelta(hours=12)).timestamp())}s AND time < {int(d1.timestamp())}s")
        if series is None:
            stats["reason"] = "InfluxDB query failed"
            return samples
        snaps = []
        for s in series:
            for t, text in s.get("values", []):
                rows = parse_solcast(parse_forecast_field(text, tz), None)
                if rows:
                    snaps.append((int(t), {int(r[0].timestamp()): r for r in rows}))
        snaps = sorted((sn for sn in snaps if sn[0] >= int(since.timestamp())), key=lambda x: x[0])
        if snaps:
            actual = _buckets(influx, cfg.pv_actual_power, d0, d1)
            fallback = _buckets(influx, cfg.pv_actual_power_fallback, d0, d1) if cfg.pv_actual_power_fallback else {}
            for k, v in fallback.items():
                actual.setdefault(k, v)
            irr = _buckets(influx, cfg.pv_irradiance, d0, d1) if cfg.pv_irradiance else {}
            curt = _buckets(influx, cfg.pv_curtailed, d0, d1) if cfg.pv_curtailed else {}
            starts = [d0 + i * SLOT for i in range(int((d1 - d0) / SLOT))]
            starts = [s for s in starts if s >= since and s + SLOT <= until]
            mids = [s + SLOT_MID for s in starts]
            ghi, elev, azi = solar_position(mids) if mids else ([], [], [])
            for s0, m, g, e, a in zip(starts, mids, ghi, elev, azi):
                if e < MIN_ELEVATION_LEARN:
                    continue
                end_ts = int((s0 + SLOT).timestamp())
                row = None
                for t, by_start in snaps:          # latest curve stored before the slot ended
                    if t > end_ts:
                        break
                    row = by_start.get(int(s0.timestamp()), row)
                if row is None:
                    log.debug("History: no forecast for %s", s0)
                    stats["no_forecast"] += 1
                    continue
                y = _slot_mean(actual, s0)
                if y is None or not 0 <= y <= PLAUSIBLE_MAX_W:
                    stats["no_actual"] += 1
                    continue
                c = _slot_mean(curt, s0) if curt else None
                if c is not None and c > 0:
                    stats["curtailed"] += 1
                    continue
                ir = _slot_mean(irr, s0) if irr else None
                if station is not None and station.ready and ir is not None and ir >= 0:
                    pot = station.potential(plant_features(ir, g, e, a, m))
                    if pot is not None and pot >= CURTAIL_MIN_POTENTIAL_W and y < CURTAIL_RATIO * pot:
                        stats["curtailed"] += 1
                        log.debug("History: curtailed (station) %s actual %.0f W potential %.0f W", s0, y, pot)
                        continue
                samples.append((s0, dayahead_features(m, row[1], row[2], row[3], g, e), y))
        day += timedelta(days=1)
    return samples


def _own_samples(influx, cutoff):
    series = influx.query(f"SELECT * FROM ml_layer_pv_slot WHERE time >= {int(cutoff.timestamp())}s")
    out = []
    for s in series or []:
        cols = s["columns"]
        for row in s.get("values", []):
            r = dict(zip(cols, row))
            need = ("target_mean_w", "solcast_p50_w", "solcast_p10_w", "solcast_p90_w",
                    "clear_sky_ghi_w_m2", "elevation_deg")
            if any(r.get(k) is None for k in need) or r["elevation_deg"] < MIN_ELEVATION_LEARN:
                continue
            s0 = datetime.fromtimestamp(int(r["time"]), timezone.utc)
            day = {"secondary_day_kwh": r.get("secondary_day_kwh")} if r.get("secondary_day_kwh") is not None else None
            x = dayahead_features(s0 + SLOT_MID, r["solcast_p50_w"], r["solcast_p10_w"], r["solcast_p90_w"],
                                  r["clear_sky_ghi_w_m2"], r["elevation_deg"], day)
            out.append((s0, x, float(r["target_mean_w"])))
    return out


def learn_from_history(influx, cfg, station, current, now=None):
    """Returns the day-ahead model to use (rebuilt from history, or `current`)."""
    if MARKER_PATH.exists():
        return current
    now = now or datetime.now(timezone.utc)
    stats = {"no_forecast": 0, "no_actual": 0, "curtailed": 0, "reason": None}
    cutoff, ok = _first_own_slot(influx)
    if not ok:
        log.warning("History: InfluxDB not readable, trying again at the next start")
        return current
    until = cutoff or now
    since = now - timedelta(days=MAX_DAYS)
    meta = _meta(influx, cfg.pv_solcast_today, lambda f: "detailedforecast" in f.lower())
    config_start, sites = _solcast_config_start(influx, cfg.pv_solcast_today, meta[0]) if meta else (None, [])
    if config_start is not None and config_start > since:
        since = config_start
    if cfg.influx_history_start is not None and cfg.influx_history_start > since:
        since = cfg.influx_history_start
    log.info("History: Solcast sites in history %s, last configuration change %s -> learning from %s",
             sites or "-", config_start.isoformat() if config_start else "-", since.isoformat())
    samples = _history_samples(influx, cfg, station, since, until, stats) if since < until else []
    own = _own_samples(influx, max(cutoff, since)) if cutoff else []
    if stats["reason"] == "InfluxDB query failed":
        log.warning("History: InfluxDB query failed, trying again at the next start")
        return current
    all_samples = sorted(samples + own, key=lambda s: s[0])
    result = {"date": now.isoformat(), "since": since.isoformat(), "history_slots": len(samples), "own_slots": len(own),
              "skipped_no_forecast": stats["no_forecast"], "skipped_no_actual": stats["no_actual"],
              "skipped_curtailed": stats["curtailed"], "reason": stats["reason"]}
    if len(samples) + len(own) == 0:
        log.info("History: nothing to learn (%s) - live learning continues", stats["reason"] or "no usable slots")
        MARKER_PATH.write_text(json.dumps(result))
        return current

    tmp = DAYAHEAD_MODEL_PATH.with_suffix(".history.tmp")
    tmp.unlink(missing_ok=True)
    fresh = DayAheadModel(path=tmp)
    for _s0, x, y in all_samples:
        fresh.learn_slot(x, y)
    tmp.unlink(missing_ok=True)
    if DAYAHEAD_MODEL_PATH.exists():
        if BACKUP_PATH.exists():          # keep the oldest (pre-history) model as the backup
            DAYAHEAD_MODEL_PATH.unlink()
        else:
            DAYAHEAD_MODEL_PATH.replace(BACKUP_PATH)
    fresh.path = DAYAHEAD_MODEL_PATH
    fresh.save()
    result.update({"first_slot": all_samples[0][0].isoformat(), "last_slot": all_samples[-1][0].isoformat(),
                   "mae_model_w": round(fresh.current_mae, 1), "mae_solcast_w": round(fresh.baseline_mae, 1),
                   "active": fresh.active})
    MARKER_PATH.write_text(json.dumps(result))
    log.info("History: day-ahead model rebuilt from %d history slots + %d own slots (%s .. %s); "
             "skipped %d without forecast, %d without actual, %d curtailed | rolling MAE %.1f W vs Solcast %.1f W, "
             "active=%s (old model kept as %s)",
             len(samples), len(own), all_samples[0][0].date(), all_samples[-1][0].date(),
             stats["no_forecast"], stats["no_actual"], stats["curtailed"],
             fresh.current_mae, fresh.baseline_mae, fresh.active, BACKUP_PATH.name)
    return fresh
