import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from config import Config
from ecowitt_model import StationModel, csi, local_hour, plant_features
from ha_client import HAClient, Publisher
from influx import InfluxWriter, line
from pv_model import (
    ENTITY_OUTPUT_CORRECTED,
    ENTITY_OUTPUT_CURVE,
    HORIZON_SLOTS,
    MIN_ELEVATION_LEARN,
    MIN_SLOT_COVERAGE,
    PLAUSIBLE_MAX_W,
    SLOT,
    SLOT_MID,
    DayAheadModel,
    NowcastModel,
    SlotAccumulator,
    blend_curve,
    daily_kwh,
    dayahead_features,
    nowcast_factor,
    parse_solcast,
    set_location,
    solar_position,
)
from building_thermal import EMHASS_SENSOR, BuildingThermalModel

ENTITY_TOMORROW_KWH = "sensor.pv_ml_forecast_tomorrow_kwh"
ENTITY_TODAY_REMAINING_KWH = "sensor.pv_ml_forecast_today_remaining_kwh"
WATCH_INTERVAL_S = 60

SOURCE_TEXT = "ha-ml-layer / day-ahead (Solcast + second provider) + station nowcast/fallback"


def _safe_float(state_dict):
    if state_dict is None:
        return None
    try:
        return float(state_dict["state"])
    except (KeyError, ValueError, TypeError):
        return None


def _plausible(value):
    return value is not None and 0 <= value <= PLAUSIBLE_MAX_W


def _attr(states, entity, key):
    return ((states.get(entity) or {}).get("attributes") or {}).get(key)


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def _sum_states(states, entities):
    """Sum of several sensors (e.g. one per roof plane); None unless all are valid."""
    if not entities:
        return None
    vals = [_safe_float(states.get(e)) for e in entities]
    return None if any(v is None for v in vals) else sum(vals)


def daily_energy(rows, now, slot0, tz, today, tomorrow, dayahead, day_ctx, station, tz_name):
    """kWh for tomorrow and for the rest of today.

    Solcast slots go through the day-ahead model (= raw Solcast while the model
    is gated off). Without Solcast slots for tomorrow, the station estimate
    (typical cloudiness only - weak) is used and flagged source=station."""
    out = {}
    for key, day in (("today_remaining", today), ("tomorrow", tomorrow)):
        sel = [r for r in rows if r[0].astimezone(tz).date() == day and r[0] + SLOT > now]
        if sel:
            mids = [r[0] + SLOT_MID for r in sel]
            ghi, elev, _azi = solar_position(mids)
            ml_kwh = sc_kwh = 0.0
            for (s0, p50, p10, p90), m, g, e in zip(sel, mids, ghi, elev):
                # the running slot only counts with its remaining share
                share = 1.0 if s0 >= now else max((s0 + SLOT - now) / SLOT, 0.0)
                v = dayahead.predict_or_solcast(dayahead_features(m, p50, p10, p90, g, e, day_ctx(s0)))
                ml_kwh += v * 0.5 / 1000.0 * share
                sc_kwh += p50 * 0.5 / 1000.0 * share
            out[key] = {"kwh": ml_kwh, "solcast_kwh": sc_kwh, "slots": len(sel),
                        "source": "ml" if dayahead.active else "solcast"}
            continue
        kwh, n = None, 0
        if station is not None and station.ready:
            day_start = datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(timezone.utc)
            slots = [day_start + i * SLOT for i in range(48)]
            slots = [s for s in slots if s + SLOT > now]
            mids = [s + SLOT_MID for s in slots]
            ghi, elev, azi = solar_position(mids) if mids else ([], [], [])
            vals = [station.fallback(m, g, e, a, local_hour(m, tz_name), None, 99.0)
                    for m, g, e, a in zip(mids, ghi, elev, azi)]
            if vals and all(v is not None for v in vals):
                kwh, n = sum(vals) * 0.5 / 1000.0, len(vals)
        out[key] = {"kwh": kwh, "solcast_kwh": None, "slots": n,
                    "source": "station" if kwh is not None else "none"}
    return out


def pv_cycle(ha, cfg, dayahead, nowcast, slot_acc, log, influx=None, snapshot=None, station=None):
    """One cycle (default 5 min).

    Primary: Solcast (+ optional second provider) corrected by the day-ahead
    model; nowcast with the live irradiance sensor for the next ~2 h.
    Station: learns irradiance -> PV power, stands in for the actual power
    while the inverter curtails, and fills every slot the forecast APIs do not
    cover. The current value and the 48-slot curve are published every cycle
    (EMHASS reads the curve; REST states do not survive an HA restart)."""
    tz = ZoneInfo(cfg.timezone)
    wanted = [cfg.pv_solcast_today, cfg.pv_solcast_tomorrow, cfg.pv_actual_power,
              cfg.pv_actual_power_fallback, cfg.pv_irradiance, cfg.pv_curtailed,
              cfg.pv_reference_irradiance] + cfg.pv_secondary_today + cfg.pv_secondary_tomorrow
    states = ha.get_states([e for e in wanted if e])
    now = datetime.now(timezone.utc)

    # --- forecast inputs ---------------------------------------------------
    rows = parse_solcast(
        _attr(states, cfg.pv_solcast_today, "detailedForecast"),
        _attr(states, cfg.pv_solcast_tomorrow, "detailedForecast"),
    )
    by_start = {int(r[0].timestamp()): r for r in rows}
    slot0 = int(now.timestamp() // SLOT.total_seconds() * SLOT.total_seconds())
    starts = [datetime.fromtimestamp(slot0 + i * SLOT.total_seconds(), timezone.utc) for i in range(HORIZON_SLOTS)]
    mids = [s + SLOT_MID for s in starts]
    ghi, elev, azi = solar_position(mids)
    g_now, e_now, a_now = solar_position([now])
    ghi_now, elev_now, azi_now = g_now[0], e_now[0], a_now[0]

    today = now.astimezone(tz).date()
    solcast_day = daily_kwh(rows, cfg.timezone)
    secondary_day = {today: _sum_states(states, cfg.pv_secondary_today),
                     today + timedelta(days=1): _sum_states(states, cfg.pv_secondary_tomorrow)}

    def day_ctx(ts):
        d = ts.astimezone(tz).date()
        return {"solcast_day_kwh": solcast_day.get(d), "secondary_day_kwh": secondary_day.get(d)}

    x_curve = []
    for s, m, g, e in zip(starts, mids, ghi, elev):
        r = by_start.get(int(s.timestamp()))
        x_curve.append(None if r is None else dayahead_features(m, r[1], r[2], r[3], g, e, day_ctx(s)))
    x_now = x_curve[0]

    # --- measurements ------------------------------------------------------
    actual = _safe_float(states.get(cfg.pv_actual_power))
    actual_source = cfg.pv_actual_power
    if actual is None and cfg.pv_actual_power_fallback:
        actual = _safe_float(states.get(cfg.pv_actual_power_fallback))
        actual_source = cfg.pv_actual_power_fallback
    if not _plausible(actual):
        log.warning("Actual PV missing/implausible (%s=%s, fallback %s) -- not learning this cycle.",
                    cfg.pv_actual_power, _safe_float(states.get(cfg.pv_actual_power)),
                    _safe_float(states.get(cfg.pv_actual_power_fallback)))
        actual = None
    irradiance = _safe_float(states.get(cfg.pv_irradiance)) if cfg.pv_irradiance else None
    if irradiance is not None and irradiance < 0:
        irradiance = None
    curtailed = bool(cfg.pv_curtailed) and (states.get(cfg.pv_curtailed) or {}).get("state") == "on"
    reference = _safe_float(states.get(cfg.pv_reference_irradiance)) if cfg.pv_reference_irradiance else None

    # --- station: plant model, typical cloudiness, reference check ----------
    x_plant_now = potential_now = csi_now = None
    if station is not None and irradiance is not None and elev_now > 0:
        x_plant_now = plant_features(irradiance, ghi_now, elev_now, azi_now, now)
        potential_now = station.potential(x_plant_now)
        if elev_now >= MIN_ELEVATION_LEARN:
            csi_now = csi(irradiance, ghi_now)
            station.update_csi(local_hour(now, cfg.timezone), csi_now)
            if actual is not None and not curtailed:
                station.learn(x_plant_now, actual)
        if reference is not None:
            station.update_reference(irradiance, reference, elev_now)
    if station is not None:
        station.save()

    # what the forecast models learn from: the actual power, or -- while the
    # inverter curtails -- the potential power seen by the station
    target, target_source = (potential_now, "station") if curtailed else (actual, "actual")

    # --- forecast models -------------------------------------------------------
    dayahead_now = dayahead.predict_or_solcast(x_now) if x_now is not None else None
    nowcast_now = x_nc = None
    if dayahead_now is not None and irradiance is not None and elev_now > 0:
        x_nc = nowcast.features(now, dayahead_now, irradiance, ghi_now, elev_now)
        nowcast_now = nowcast.predict(x_nc, dayahead_now)
    if target is not None and x_nc is not None and elev_now >= MIN_ELEVATION_LEARN:
        nowcast.learn(x_nc, target, dayahead_now)
        nowcast.save()

    prev_slot_start = slot_acc.start
    finished = slot_acc.add(starts[0], x_now, target)
    if finished is not None and finished[0] is not None:
        x_done, mean_done = finished
        if x_done["elevation"] >= MIN_ELEVATION_LEARN:
            dayahead.learn_slot(x_done, mean_done)
            dayahead.save()
        if influx is not None:
            influx.write([line("ml_layer_pv_slot", {
                "target_mean_w": _r(mean_done),
                "solcast_p50_w": _r(x_done["solcast_p50"]),
                "solcast_p10_w": _r(x_done["solcast_p10"]),
                "solcast_p90_w": _r(x_done["solcast_p90"]),
                "secondary_day_kwh": _r(x_done.get("secondary_day_kwh"), 2),
                "clear_sky_ghi_w_m2": _r(x_done["clear_sky_ghi"]),
                "elevation_deg": _r(x_done["elevation"], 2),
            }, ts=prev_slot_start)])

    # --- curve: API slots corrected, missing slots from the station ------------
    factor = nowcast_factor(nowcast_now, dayahead_now) if dayahead_now is not None else 1.0
    values, sources = [], []
    for s, m, g, e, a, x in zip(starts, mids, ghi, elev, azi, x_curve):
        if x is not None:
            values.append(dayahead.predict_or_solcast(x))
            sources.append("solcast")
        elif e <= 0:
            values.append(0.0)
            sources.append("night")
        else:
            dt_h = max((s - now).total_seconds() / 3600.0, 0.0)
            v = None if station is None else station.fallback(m, g, e, a, local_hour(m, cfg.timezone), csi_now, dt_h)
            values.append(v)
            sources.append("station")
    blended = blend_curve(starts, [v or 0.0 for v in values], now, factor)
    curve = [b if src == "solcast" else v for b, v, src in zip(blended, values, sources)]
    n_station = sources.count("station")
    n_solcast = sources.count("solcast")
    forecast_source = "station" if n_solcast == 0 and n_station else ("solcast+station" if n_station else "solcast")
    missing = sum(1 for v in curve if v is None)

    if x_now is not None:
        corrected_now = dayahead_now * factor if nowcast_now is not None else dayahead_now
    elif potential_now is not None:
        corrected_now = potential_now
    else:
        corrected_now = curve[0]

    def nmae(mae):
        return round(100.0 * mae / (cfg.pv_peak_power_kw * 1000.0), 2) if cfg.pv_peak_power_kw else None

    if corrected_now is not None:
        ha.set_state(ENTITY_OUTPUT_CORRECTED, round(corrected_now, 1), attributes={
            "unit_of_measurement": "W",
            "device_class": "power",
            "friendly_name": "PV Forecast (ML corrected)",
            "forecast_source": forecast_source,
            "fallback_slots": n_station,
            "current_mae_w": round(nowcast.current_mae if nowcast.active else dayahead.current_mae, 1),
            "mae_dayahead_w": round(dayahead.current_mae, 1),
            "mae_solcast_w": round(dayahead.baseline_mae, 1),
            "nmae_dayahead_pct": nmae(dayahead.current_mae),
            "nmae_solcast_pct": nmae(dayahead.baseline_mae),
            "mae_nowcast_w": round(nowcast.current_mae, 1),
            "mae_nowcast_baseline_w": round(nowcast.baseline_mae, 1),
            "dayahead_active": dayahead.active,
            "nowcast_active": nowcast.active,
            "samples_dayahead": dayahead.n_learned,
            "samples_nowcast": nowcast.n_learned,
            "dayahead_now_w": _r(dayahead_now),
            "solcast_p50_now_w": _r(x_now["solcast_p50"]) if x_now else None,
            "secondary_today_kwh": _r(secondary_day[today], 2),
            "nowcast_factor": round(factor, 3),
            "actual_source": actual_source,
            "learning_target": target_source,
            "curtailed": curtailed,
            "irradiance_w_m2": _r(irradiance),
            "station_potential_w": _r(potential_now),
            "station_ready": station.ready if station else False,
            "station_w_per_w_m2": _r(station.ratio, 3) if station else None,
            "station_plant_active": station.plant.active if station else False,
            "irradiance_vs_reference": _r(station.ref_ratio, 3) if station else None,
            "source": SOURCE_TEXT,
        })

    if missing:
        log.warning("No forecast for %d of %d slots (no Solcast data, station not ready) -- curve not published.",
                    missing, len(curve))
    else:
        ha.set_state(ENTITY_OUTPUT_CURVE, round(curve[0], 1), attributes={
            "unit_of_measurement": "W",
            "device_class": "power",
            "friendly_name": "PV Forecast Curve (ML corrected)",
            "forecasts": [{"date": s.isoformat(), "p_pv_forecast": round(v, 1)} for s, v in zip(starts, curve)],
            "forecast_source": forecast_source,
            "fallback_slots": n_station,
            "nowcast_factor": round(factor, 3),
            "source": SOURCE_TEXT,
        })

    log.info(
        "now: source=%s solcast=%s dayahead=%s nowcast=%s factor=%.2f actual=%s station=%s%s | "
        "MAE dayahead=%.1fW vs solcast=%.1fW (n=%d, active=%s) nowcast=%.1fW vs %.1fW (n=%d, active=%s) "
        "station n=%d ready=%s",
        forecast_source,
        "-" if x_now is None else f"{x_now['solcast_p50']:.0f}W",
        "-" if dayahead_now is None else f"{dayahead_now:.0f}W",
        "-" if nowcast_now is None else f"{nowcast_now:.0f}W", factor,
        "-" if actual is None else f"{actual:.0f}W",
        "-" if potential_now is None else f"{potential_now:.0f}W",
        " CURTAILED" if curtailed else "",
        dayahead.current_mae, dayahead.baseline_mae, dayahead.n_learned, dayahead.active,
        nowcast.current_mae, nowcast.baseline_mae, nowcast.n_learned, nowcast.active,
        station.n_ratio if station else 0, station.ready if station else False,
    )

    # --- daily energy for EMHASS soc_final / battery reserve -------------------
    tomorrow = today + timedelta(days=1)
    energy = daily_energy(rows, now, starts[0], tz, today, tomorrow, dayahead, day_ctx, station, cfg.timezone)
    for entity, key, name, sec in (
            (ENTITY_TOMORROW_KWH, "tomorrow", "PV Forecast Tomorrow (ML)", secondary_day.get(tomorrow)),
            (ENTITY_TODAY_REMAINING_KWH, "today_remaining", "PV Forecast Today Remaining (ML)", None)):
        e = energy[key]
        ha.set_state(entity, "unknown" if e["kwh"] is None else round(e["kwh"], 2), attributes={
            "unit_of_measurement": "kWh",
            "device_class": "energy",
            "friendly_name": name,
            "source": e["source"],
            "solcast_kwh": _r(e["solcast_kwh"], 2),
            "secondary_kwh": _r(sec, 2) if key == "tomorrow" else None,
            "slots": e["slots"],
            "model_active": dayahead.active,
        })

    if influx is None:
        return
    influx.write([line("ml_layer_pv", {
        "solcast_p50_w": _r(x_now["solcast_p50"]) if x_now else None,
        "solcast_p10_w": _r(x_now["solcast_p10"]) if x_now else None,
        "solcast_p90_w": _r(x_now["solcast_p90"]) if x_now else None,
        "secondary_today_kwh": _r(secondary_day[today], 2),
        "dayahead_w": _r(dayahead_now),
        "nowcast_w": _r(nowcast_now),
        "nowcast_factor": _r(factor, 3),
        "published_w": _r(corrected_now),
        "actual_w": _r(actual),
        "target_w": _r(target),
        "curtailed": curtailed,
        "irradiance_w_m2": _r(irradiance),
        "reference_irradiance_w_m2": _r(reference),
        "station_potential_w": _r(potential_now),
        "station_w_per_w_m2": _r(station.ratio, 4) if station else None,
        "irradiance_vs_reference": _r(station.ref_ratio, 4) if station else None,
        "clear_sky_ghi_w_m2": _r(ghi_now),
        "elevation_deg": _r(elev_now, 2),
        "azimuth_deg": _r(azi_now, 1),
        "mae_dayahead_w": _r(dayahead.current_mae),
        "mae_solcast_w": _r(dayahead.baseline_mae),
        "mae_nowcast_w": _r(nowcast.current_mae),
        "dayahead_active": dayahead.active,
        "nowcast_active": nowcast.active,
        "fallback_slots": n_station,
        "forecast_source": forecast_source,
        "tomorrow_kwh": _r(energy["tomorrow"]["kwh"], 2),
        "tomorrow_solcast_kwh": _r(energy["tomorrow"]["solcast_kwh"], 2),
        "today_remaining_kwh": _r(energy["today_remaining"]["kwh"], 2),
    }, ts=now)])

    # once per evening: what EMHASS plans with for tomorrow (raw forecasts + ML curve)
    local_now = now.astimezone(tz)
    if snapshot is not None and local_now.hour >= cfg.influx_snapshot_hour \
            and snapshot.get("date") != local_now.date():
        tomorrow = local_now.date() + timedelta(days=1)
        rows_tm = [r for r in rows if r[0].astimezone(tz).date() == tomorrow]
        if rows_tm:
            mids_tm = [r[0] + SLOT_MID for r in rows_tm]
            ghi_tm, elev_tm, _azi_tm = solar_position(mids_tm)
            lines = []
            for (s0, p50, p10, p90), m, g, e in zip(rows_tm, mids_tm, ghi_tm, elev_tm):
                ml = dayahead.predict_or_solcast(dayahead_features(m, p50, p10, p90, g, e, day_ctx(s0)))
                lines.append(line("ml_layer_dayahead", {
                    "solcast_p50_w": _r(p50), "solcast_p10_w": _r(p10), "solcast_p90_w": _r(p90),
                    "secondary_day_kwh": _r(secondary_day.get(tomorrow), 2),
                    "ml_dayahead_w": _r(ml),
                }, tags={"issued": local_now.date().isoformat()}, ts=s0))
            if influx.write(lines):
                snapshot["date"] = local_now.date()
                log.info("InfluxDB: day-ahead snapshot for %s written (%d slots)", tomorrow, len(lines))


def log_influx_inventory(influx, cfg, log):
    """Once at start: which history of the configured sensors is in InfluxDB
    (read-only). Basis for learning from history."""
    entities = [cfg.pv_solcast_today, cfg.pv_solcast_tomorrow, cfg.pv_actual_power,
                cfg.pv_actual_power_fallback, cfg.pv_irradiance, cfg.pv_curtailed,
                cfg.pv_reference_irradiance, cfg.thermal_outdoor_temp] \
        + cfg.pv_secondary_today + cfg.pv_secondary_tomorrow \
        + [f["temp"] for f in cfg.thermal_floors]
    for e in [e for e in dict.fromkeys(entities) if e]:
        inv = influx.inventory(e)
        if inv is None:
            log.warning("InfluxDB history check: query failed for %s (see InfluxDB warning above)", e)
            return
        if not inv:
            log.info("InfluxDB history: %s -> no data", e)
        for m in inv:
            days = (m["last"] - m["first"]) / 86400 if m.get("first") and m.get("last") else 0
            curve = any("detailedforecast" in f.lower() for f in m["fields"])
            log.info("InfluxDB history: %s -> '%s' %s .. %s (%.0f days), fields: %s%s",
                     e, m["measurement"],
                     datetime.fromtimestamp(m["first"], timezone.utc).date() if m.get("first") else "?",
                     datetime.fromtimestamp(m["last"], timezone.utc).date() if m.get("last") else "?",
                     days, ", ".join(m["fields"][:8]),
                     " | Solcast 30-min curve stored: YES" if curve else
                     (" | Solcast 30-min curve stored: no" if "solcast" in e else ""))


def run():
    cfg = Config()
    logging.basicConfig(level=cfg.log_level.upper())
    log = logging.getLogger("main")
    for problem in cfg.problems:
        log.error("Configuration: %s", problem)

    if not cfg.pv_enabled and not cfg.thermal_enabled:
        log.info("All models disabled (add-on option), exiting.")
        return

    ha = Publisher(HAClient(cfg.ha_url, cfg.token),
                   ENTITY_OUTPUT_CURVE if cfg.pv_enabled else EMHASS_SENSOR)
    if None in (cfg.latitude, cfg.longitude, cfg.timezone):
        while True:
            try:
                cfg.resolve_site(ha.get_config())
                break
            except Exception as exc:
                log.warning("Could not read location from Home Assistant (%s), retrying in 60 s", exc)
                time.sleep(60)
    set_location(cfg.latitude, cfg.longitude, cfg.timezone)
    dayahead = DayAheadModel() if cfg.pv_enabled else None
    nowcast = NowcastModel() if cfg.pv_enabled else None
    slot_acc = SlotAccumulator(
        max(1, round(MIN_SLOT_COVERAGE * SLOT.total_seconds() / cfg.poll_interval)))
    station = StationModel() if cfg.pv_enabled and cfg.pv_irradiance else None
    thermal = BuildingThermalModel(cfg, max_gap_s=2 * cfg.poll_interval + 60) if cfg.thermal_enabled else None

    influx = None
    if cfg.influx_enabled:
        influx = InfluxWriter(cfg.influx_host, cfg.influx_port, cfg.influx_database,
                              cfg.influx_username, cfg.influx_password, cfg.influx_ssl)
        log.info("InfluxDB enabled: %s", influx.describe())
        log_influx_inventory(influx, cfg, log)
    snapshot = {}

    log.info("ha-ml-layer started. Poll interval: %ss | pv=%s thermal=%s influx=%s station=%s "
             "second provider=%s curtailment=%s reference=%s | site %.3f/%.3f %s",
             cfg.poll_interval, cfg.pv_enabled, cfg.thermal_enabled, cfg.influx_enabled,
             station is not None, bool(cfg.pv_secondary_today), bool(cfg.pv_curtailed),
             bool(cfg.pv_reference_irradiance), cfg.latitude, cfg.longitude, cfg.timezone)

    next_cycle = time.monotonic()
    while True:
        if time.monotonic() >= next_cycle:
            next_cycle = time.monotonic() + cfg.poll_interval
            if thermal is not None:
                try:
                    thermal.update(ha)
                except Exception:
                    log.exception("Error in building_thermal cycle, retrying next interval.")

            if dayahead is not None:
                try:
                    pv_cycle(ha, cfg, dayahead, nowcast, slot_acc, log, influx, snapshot, station)
                except Exception:
                    log.exception("Error in PV cycle, retrying next interval.")
        else:
            # between cycles: bring the sensors back right after an HA restart
            try:
                ha.ensure()
            except Exception:
                log.exception("Error while checking the published sensors")
        time.sleep(max(1.0, min(WATCH_INTERVAL_S, next_cycle - time.monotonic())))


if __name__ == "__main__":
    run()