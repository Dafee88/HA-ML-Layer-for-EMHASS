import logging
import time
from datetime import datetime, timezone

from config import Config
from ha_client import HAClient
from pv_model import (
    ENTITY_ACTUAL_POWER,
    ENTITY_ACTUAL_POWER_FALLBACK,
    ENTITY_IRRADIANCE,
    ENTITY_OUTPUT_CORRECTED,
    ENTITY_OUTPUT_CURVE,
    ENTITY_SOLCAST_TODAY,
    ENTITY_SOLCAST_TOMORROW,
    HORIZON_SLOTS,
    MIN_ELEVATION_LEARN,
    PLAUSIBLE_MAX_W,
    DayAheadModel,
    NowcastModel,
    blend_curve,
    current_slot_index,
    dayahead_features,
    nowcast_factor,
    parse_solcast,
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


def pv_cycle(ha, dayahead, nowcast, log):
    """One 5-minute cycle: learn both models on the current slot, then publish
    the current value and the full 48-slot curve (every cycle, so the curve is
    always < 1 h old for the EMHASS payload and reappears right after an HA
    restart)."""
    states = ha.get_states([
        ENTITY_SOLCAST_TODAY,
        ENTITY_SOLCAST_TOMORROW,
        ENTITY_ACTUAL_POWER,
        ENTITY_ACTUAL_POWER_FALLBACK,
        ENTITY_IRRADIANCE,
    ])
    now = datetime.now(timezone.utc)

    rows = parse_solcast(
        _attr(states, ENTITY_SOLCAST_TODAY, "detailedForecast"),
        _attr(states, ENTITY_SOLCAST_TOMORROW, "detailedForecast"),
    )
    idx = current_slot_index(rows, now)
    if idx is None:
        log.warning("Solcast detailedForecast has no slot for %s -- skipping cycle.", now.isoformat())
        return
    horizon = rows[idx:idx + HORIZON_SLOTS]

    starts = [r[0] for r in horizon]
    ghi, elev = solar_geometry(starts)
    x_curve = [dayahead_features(s, p50, p10, p90, g, e)
               for (s, p50, p10, p90), g, e in zip(horizon, ghi, elev)]

    # current slot = first slot of the horizon; same features as the curve
    x_now = x_curve[0]
    ghi_now, elev_now = solar_geometry([now])
    ghi_now, elev_now = ghi_now[0], elev_now[0]

    actual = _safe_float(states.get(ENTITY_ACTUAL_POWER))
    actual_source = ENTITY_ACTUAL_POWER
    if actual is None:
        actual = _safe_float(states.get(ENTITY_ACTUAL_POWER_FALLBACK))
        actual_source = ENTITY_ACTUAL_POWER_FALLBACK
    irradiance = _safe_float(states.get(ENTITY_IRRADIANCE))

    dayahead_now = dayahead.predict_or_solcast(x_now)

    # --- nowcast (only with live Ecowitt data) ---
    nowcast_now = None
    x_nc = None
    if irradiance is not None and irradiance >= 0 and elev_now > 0:
        x_nc = nowcast.features(dayahead_now, irradiance, ghi_now, elev_now)
        nowcast_now = nowcast.predict(x_nc)

    # --- learning ---
    if not _plausible(actual):
        log.warning("Actual PV missing/implausible (%s=%s, fallback %s) -- not learning this cycle.",
                    ENTITY_ACTUAL_POWER, _safe_float(states.get(ENTITY_ACTUAL_POWER)),
                    _safe_float(states.get(ENTITY_ACTUAL_POWER_FALLBACK)))
    elif elev_now >= MIN_ELEVATION_LEARN:
        dayahead.learn(x_now, actual)
        dayahead.save()
        if x_nc is not None:
            nowcast.learn(x_nc, actual)
            nowcast.save()

    # --- publish current value ---
    factor = nowcast_factor(nowcast_now, dayahead_now)
    corrected_now = dayahead_now * factor if nowcast_now is not None else dayahead_now
    ha.set_state(
        ENTITY_OUTPUT_CORRECTED,
        round(corrected_now, 1),
        attributes={
            "unit_of_measurement": "W",
            "device_class": "power",
            "friendly_name": "PV Forecast (ML-korrigiert)",
            "current_mae_w": round(nowcast.current_mae if nowcast.n_learned else dayahead.current_mae, 1),
            "mae_dayahead_w": round(dayahead.current_mae, 1),
            "mae_nowcast_w": round(nowcast.current_mae, 1),
            "samples_dayahead": dayahead.n_learned,
            "samples_nowcast": nowcast.n_learned,
            "dayahead_now_w": round(dayahead_now, 1),
            "solcast_p50_now_w": round(x_now["solcast_p50"], 1),
            "nowcast_factor": round(factor, 3),
            "actual_source": actual_source,
            "ecowitt_irradiance_w_m2": round(irradiance, 1) if irradiance is not None else None,
            "source": "ha-ml-layer / day-ahead (Solcast-only) + Ecowitt nowcast",
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
            "friendly_name": "PV Forecast Curve (ML-korrigiert)",
            "forecasts": [
                {"date": s.isoformat(), "p_pv_forecast": round(v, 1)}
                for s, v in zip(starts, curve)
            ],
            "nowcast_factor": round(factor, 3),
            "source": "ha-ml-layer / day-ahead (Solcast-only) + Ecowitt nowcast",
        },
    )

    log.info(
        "now: solcast=%.0fW dayahead=%.0fW nowcast=%s factor=%.2f actual=%s | curve %d slots | "
        "MAE dayahead=%.1fW (n=%d) nowcast=%.1fW (n=%d)",
        x_now["solcast_p50"], dayahead_now,
        "-" if nowcast_now is None else f"{nowcast_now:.0f}W", factor,
        "-" if actual is None else f"{actual:.0f}W", len(curve),
        dayahead.current_mae, dayahead.n_learned, nowcast.current_mae, nowcast.n_learned,
    )


def run():
    cfg = Config()
    logging.basicConfig(level=cfg.log_level.upper())
    log = logging.getLogger("main")

    pv_enabled = cfg.models.get("pv_forecast_correction", False)
    thermal_enabled = cfg.models.get("building_thermal", False)

    if not pv_enabled and not thermal_enabled:
        log.info("All models disabled (add-on option), exiting.")
        return

    ha = HAClient(cfg.ha_url, cfg.token)
    dayahead = DayAheadModel() if pv_enabled else None
    nowcast = NowcastModel() if pv_enabled else None
    thermal = BuildingThermalModel() if thermal_enabled else None

    log.info("ha-ml-layer started. Poll interval: %ss | pv=%s thermal=%s",
             cfg.poll_interval, pv_enabled, thermal_enabled)

    while True:
        if thermal is not None:
            try:
                thermal.update(ha)
            except Exception:
                log.exception("Error in building_thermal cycle, retrying next interval.")

        if dayahead is not None:
            try:
                pv_cycle(ha, dayahead, nowcast, log)
            except Exception:
                log.exception("Error in PV cycle, retrying next interval.")

        time.sleep(cfg.poll_interval)


if __name__ == "__main__":
    run()