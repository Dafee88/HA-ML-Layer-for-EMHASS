import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from config import Config
from ha_client import HAClient
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
    current_slot_index,
    dayahead_features,
    nowcast_factor,
    parse_solcast,
    set_location,
    solar_geometry,
)
from building_thermal import BuildingThermalModel  # 28.09.2026


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


def pv_cycle(ha, cfg, dayahead, nowcast, slot_acc, log, influx=None, snapshot=None):
    """One 5-minute cycle: learn the nowcast on the current reading and the
    day-ahead model on the slot that just ended, then publish
    the current value and the full 48-slot curve (every cycle, so the curve is
    always < 1 h old for the EMHASS payload and reappears right after an HA
    restart)."""
    states = ha.get_states([e for e in (
        cfg.pv_solcast_today,
        cfg.pv_solcast_tomorrow,
        cfg.pv_actual_power,
        cfg.pv_actual_power_fallback,
        cfg.pv_irradiance,
    ) if e])
    now = datetime.now(timezone.utc)

    rows = parse_solcast(
        _attr(states, cfg.pv_solcast_today, "detailedForecast"),
        _attr(states, cfg.pv_solcast_tomorrow, "detailedForecast"),
    )
    idx = current_slot_index(rows, now)
    if idx is None:
        log.warning("Solcast detailedForecast has no slot for %s -- skipping cycle.", now.isoformat())
        return
    horizon = rows[idx:idx + HORIZON_SLOTS]

    starts = [r[0] for r in horizon]
    mids = [s + SLOT_MID for s in starts]
    ghi, elev = solar_geometry(mids)
    x_curve = [dayahead_features(m, p50, p10, p90, g, e)
               for (_s, p50, p10, p90), m, g, e in zip(horizon, mids, ghi, elev)]

    # current slot = first slot of the horizon; same features as the curve
    x_now = x_curve[0]
    ghi_now, elev_now = solar_geometry([now])
    ghi_now, elev_now = ghi_now[0], elev_now[0]

    actual = _safe_float(states.get(cfg.pv_actual_power))
    actual_source = cfg.pv_actual_power
    if actual is None and cfg.pv_actual_power_fallback:
        actual = _safe_float(states.get(cfg.pv_actual_power_fallback))
        actual_source = cfg.pv_actual_power_fallback
    irradiance = _safe_float(states.get(cfg.pv_irradiance)) if cfg.pv_irradiance else None

    dayahead_now = dayahead.predict_or_solcast(x_now)

    # --- nowcast (only with a live irradiance sensor) ---
    nowcast_now = None
    x_nc = None
    if irradiance is not None and irradiance >= 0 and elev_now > 0:
        x_nc = nowcast.features(now, dayahead_now, irradiance, ghi_now, elev_now)
        nowcast_now = nowcast.predict(x_nc, dayahead_now)

    # --- learning ---
    if not _plausible(actual):
        log.warning("Actual PV missing/implausible (%s=%s, fallback %s) -- not learning this cycle.",
                    cfg.pv_actual_power, _safe_float(states.get(cfg.pv_actual_power)),
                    _safe_float(states.get(cfg.pv_actual_power_fallback)))
        actual = None
    elif elev_now >= MIN_ELEVATION_LEARN and x_nc is not None:
        nowcast.learn(x_nc, actual, dayahead_now)
        nowcast.save()

    prev_slot_start = slot_acc.start
    finished = slot_acc.add(starts[0], x_now, actual)
    if finished is not None:
        x_done, mean_done = finished
        if x_done["elevation"] >= MIN_ELEVATION_LEARN:
            dayahead.learn_slot(x_done, mean_done)
            dayahead.save()
        if influx is not None:
            influx.write([line("ml_layer_pv_slot", {
                "actual_mean_w": _r(mean_done),
                "solcast_p50_w": _r(x_done["solcast_p50"]),
                "solcast_p10_w": _r(x_done["solcast_p10"]),
                "solcast_p90_w": _r(x_done["solcast_p90"]),
                "clear_sky_ghi_w_m2": _r(x_done["clear_sky_ghi"]),
                "elevation_deg": _r(x_done["elevation"], 2),
            }, ts=prev_slot_start)])

    # --- publish current value ---
    factor = nowcast_factor(nowcast_now, dayahead_now)
    corrected_now = dayahead_now * factor if nowcast_now is not None else dayahead_now
    ha.set_state(
        ENTITY_OUTPUT_CORRECTED,
        round(corrected_now, 1),
        attributes={
            "unit_of_measurement": "W",
            "device_class": "power",
            "friendly_name": "PV Forecast (ML corrected)",
            "current_mae_w": round(nowcast.current_mae if nowcast.active else dayahead.current_mae, 1),
            "mae_dayahead_w": round(dayahead.current_mae, 1),
            "mae_solcast_w": round(dayahead.baseline_mae, 1),
            "mae_nowcast_w": round(nowcast.current_mae, 1),
            "mae_nowcast_baseline_w": round(nowcast.baseline_mae, 1),
            "dayahead_active": dayahead.active,
            "nowcast_active": nowcast.active,
            "samples_dayahead": dayahead.n_learned,
            "samples_nowcast": nowcast.n_learned,
            "dayahead_now_w": round(dayahead_now, 1),
            "solcast_p50_now_w": round(x_now["solcast_p50"], 1),
            "nowcast_factor": round(factor, 3),
            "actual_source": actual_source,
            "irradiance_w_m2": round(irradiance, 1) if irradiance is not None else None,
            "source": "ha-ml-layer / day-ahead (Solcast) + irradiance nowcast",
        },
    )

    # --- publish curve ---
    dayahead_curve = [dayahead.predict_or_solcast(x) for x in x_curve]
    curve = blend_curve(starts, dayahead_curve, now, factor)
    ha.set_state(
        ENTITY_OUTPUT_CURVE,
        round(curve[0], 1),
        attributes={
            "unit_of_measurement": "W",
            "device_class": "power",
            "friendly_name": "PV Forecast Curve (ML corrected)",
            "forecasts": [
                {"date": s.isoformat(), "p_pv_forecast": round(v, 1)}
                for s, v in zip(starts, curve)
            ],
            "nowcast_factor": round(factor, 3),
            "source": "ha-ml-layer / day-ahead (Solcast) + irradiance nowcast",
        },
    )

    log.info(
        "now: solcast=%.0fW dayahead=%.0fW nowcast=%s factor=%.2f actual=%s | curve %d slots | "
        "rolling MAE dayahead=%.1fW vs solcast=%.1fW (n=%d, active=%s) "
        "nowcast=%.1fW vs dayahead=%.1fW (n=%d, active=%s)",
        x_now["solcast_p50"], dayahead_now,
        "-" if nowcast_now is None else f"{nowcast_now:.0f}W", factor,
        "-" if actual is None else f"{actual:.0f}W", len(curve),
        dayahead.current_mae, dayahead.baseline_mae, dayahead.n_learned, dayahead.active,
        nowcast.current_mae, nowcast.baseline_mae, nowcast.n_learned, nowcast.active,
    )

    if influx is None:
        return
    influx.write([line("ml_layer_pv", {
        "solcast_p50_w": _r(x_now["solcast_p50"]),
        "solcast_p10_w": _r(x_now["solcast_p10"]),
        "solcast_p90_w": _r(x_now["solcast_p90"]),
        "dayahead_w": _r(dayahead_now),
        "nowcast_w": _r(nowcast_now),
        "nowcast_factor": _r(factor, 3),
        "published_w": _r(corrected_now),
        "actual_w": _r(actual),
        "irradiance_w_m2": _r(irradiance),
        "clear_sky_ghi_w_m2": _r(ghi_now),
        "elevation_deg": _r(elev_now, 2),
        "mae_dayahead_w": _r(dayahead.current_mae),
        "mae_solcast_w": _r(dayahead.baseline_mae),
        "mae_nowcast_w": _r(nowcast.current_mae),
        "dayahead_active": dayahead.active,
        "nowcast_active": nowcast.active,
    }, ts=now)])

    # once per evening: what EMHASS plans with for tomorrow (raw Solcast + ML curve)
    local_now = now.astimezone(ZoneInfo(cfg.timezone))
    if snapshot is not None and local_now.hour >= cfg.influx_snapshot_hour \
            and snapshot.get("date") != local_now.date():
        tomorrow = local_now.date() + timedelta(days=1)
        rows_tm = [r for r in rows if r[0].astimezone(ZoneInfo(cfg.timezone)).date() == tomorrow]
        if rows_tm:
            mids_tm = [r[0] + SLOT_MID for r in rows_tm]
            ghi_tm, elev_tm = solar_geometry(mids_tm)
            lines = []
            for (s0, p50, p10, p90), m, g, e in zip(rows_tm, mids_tm, ghi_tm, elev_tm):
                ml = dayahead.predict_or_solcast(dayahead_features(m, p50, p10, p90, g, e))
                lines.append(line("ml_layer_dayahead", {
                    "solcast_p50_w": _r(p50), "solcast_p10_w": _r(p10), "solcast_p90_w": _r(p90),
                    "ml_dayahead_w": _r(ml),
                }, tags={"issued": local_now.date().isoformat()}, ts=s0))
            if influx.write(lines):
                snapshot["date"] = local_now.date()
                log.info("InfluxDB: day-ahead snapshot for %s written (%d slots)", tomorrow, len(lines))


def run():
    cfg = Config()
    logging.basicConfig(level=cfg.log_level.upper())
    log = logging.getLogger("main")
    for problem in cfg.problems:
        log.error("Configuration: %s", problem)

    if not cfg.pv_enabled and not cfg.thermal_enabled:
        log.info("All models disabled (add-on option), exiting.")
        return

    ha = HAClient(cfg.ha_url, cfg.token)
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
    thermal = BuildingThermalModel(cfg, max_gap_s=2 * cfg.poll_interval + 60) if cfg.thermal_enabled else None

    influx = None
    if cfg.influx_enabled:
        influx = InfluxWriter(cfg.influx_host, cfg.influx_port, cfg.influx_database,
                              cfg.influx_username, cfg.influx_password, cfg.influx_ssl)
        log.info("InfluxDB enabled: %s", influx.describe())
    snapshot = {}

    log.info("ha-ml-layer started. Poll interval: %ss | pv=%s thermal=%s influx=%s | site %.3f/%.3f %s",
             cfg.poll_interval, cfg.pv_enabled, cfg.thermal_enabled, cfg.influx_enabled,
             cfg.latitude, cfg.longitude, cfg.timezone)

    while True:
        if thermal is not None:
            try:
                thermal.update(ha)
            except Exception:
                log.exception("Error in building_thermal cycle, retrying next interval.")

        if dayahead is not None:
            try:
                pv_cycle(ha, cfg, dayahead, nowcast, slot_acc, log, influx, snapshot)
            except Exception:
                log.exception("Error in PV cycle, retrying next interval.")

        time.sleep(cfg.poll_interval)


if __name__ == "__main__":
    run()