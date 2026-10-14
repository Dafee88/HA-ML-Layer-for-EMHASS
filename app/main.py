import logging
import time
from datetime import datetime, timezone

from config import Config
from ha_client import HAClient
from pv_model import (
    ENTITY_ACTUAL_POWER,
    ENTITY_FORECAST_SOLAR,
    ENTITY_OUTPUT_CORRECTED,
    ENTITY_SOLCAST,
    ENTITY_IRRADIANCE,
    ENTITY_PRESSURE,
    PVForecastCorrectionModel,
    build_features,
)
from building_thermal import BuildingThermalModel  # NEU 28.09.2026

# --- NEU: Forecast-Kurve statt nur Momentanwert (18.09.2026) ----------------
ENTITY_PV_FORECAST_RAW = "sensor.p_pv_forecast"                         # EMHASS' eigener physikalischer Forecast (30-Min-Array in .forecasts)
ENTITY_SOLCAST_TODAY = "sensor.solcast_pv_forecast_forecast_today"       # .detailedForecast, 30-Min-Schritte, kWh/Halbstunde
ENTITY_SOLCAST_TOMORROW = "sensor.solcast_pv_forecast_forecast_tomorrow"
ENTITY_OUTPUT_CURVE = "sensor.pv_ml_corrected_forecast_curve"            # NEUER Sensor -- getrennt vom Nowcast-Sensor, damit sich beide nicht überschreiben

CURVE_UPDATE_EVERY_N_CYCLES = 6  # FIX 29.09.2026: poll_interval ist 300s -> 6 Zyklen = 30 Min (vorher 60 = 5h; EMHASS-Payload nutzt die Kurve nur, wenn < 1h alt)
# -----------------------------------------------------------------------------


def _safe_float(state_dict, default=None):
    if state_dict is None:
        return default
    try:
        return float(state_dict["state"])
    except (KeyError, ValueError, TypeError):
        return default


PLAUSIBLE_MAX_W = 15000


def _is_plausible(value):
    return value is not None and 0 <= value <= PLAUSIBLE_MAX_W


def _merge_forecast_arrays(pv_forecast_attr, solcast_today_attr, solcast_tomorrow_attr):
    """Baut für jeden 30-Min-Zeitstempel aus EMHASS' eigenem Forecast
    (sensor.p_pv_forecast.forecasts) ein Paar (forecast_solar, solcast), indem
    der jeweils passende Solcast-detailedForecast-Eintrag (heute+morgen
    zusammengeführt) gesucht wird. Solcast liefert kWh/30min -> *2000 für
    durchschnittliche W-Leistung im Intervall. Zeitschritte ohne passenden
    Solcast-Eintrag werden ausgelassen, statt geraten."""
    solcast_by_ts = {}
    for entry in (solcast_today_attr or []) + (solcast_tomorrow_attr or []):
        solcast_by_ts[entry["period_start"]] = entry["pv_estimate"] * 2000  # kWh/30min -> W

    merged_ts, merged_fs, merged_sc = [], [], []
    for entry in pv_forecast_attr or []:
        ts_str = entry["date"]
        fs_val = float(entry["p_pv_forecast"])
        sc_val = solcast_by_ts.get(ts_str)
        if sc_val is None:
            continue
        merged_ts.append(datetime.fromisoformat(ts_str))
        merged_fs.append(fs_val)
        merged_sc.append(sc_val)
    return merged_ts, merged_fs, merged_sc


def run():
    cfg = Config()
    logging.basicConfig(level=cfg.log_level.upper())
    log = logging.getLogger("main")

    pv_enabled = cfg.models.get("pv_forecast_correction", False)
    thermal_enabled = cfg.models.get("building_thermal", False)   # NEU 28.09.2026

    if not pv_enabled and not thermal_enabled:
        log.info("Alle Modelle deaktiviert (Add-on-Option), beende.")
        return

    ha = HAClient(cfg.ha_url, cfg.token)
    model = PVForecastCorrectionModel() if pv_enabled else None
    thermal = BuildingThermalModel() if thermal_enabled else None   # NEU 28.09.2026

    log.info("ha-ml-layer gestartet. Poll-Intervall: %ss | pv=%s thermal=%s",
             cfg.poll_interval, pv_enabled, thermal_enabled)

    cycle_count = 0

    while True:
        # --- NEU 28.09.2026: Gebaeude-Waermemodell, eigener try-Block ---
        if thermal is not None:
            try:
                thermal.update(ha)
            except Exception:
                log.exception("Fehler im building_thermal-Zyklus, naechster Versuch im naechsten Intervall.")

        if model is None:
            time.sleep(cfg.poll_interval)
            continue

        try:
            states = ha.get_states([
                ENTITY_FORECAST_SOLAR,
                ENTITY_SOLCAST,
                ENTITY_ACTUAL_POWER,
                ENTITY_IRRADIANCE,
                ENTITY_PRESSURE
            ])

            forecast_solar = _safe_float(states.get(ENTITY_FORECAST_SOLAR))
            solcast = _safe_float(states.get(ENTITY_SOLCAST))
            actual = _safe_float(states.get(ENTITY_ACTUAL_POWER))
            ecowitt_irradiance = _safe_float(states.get(ENTITY_IRRADIANCE))
            ecowitt_pressure = _safe_float(states.get(ENTITY_PRESSURE))

            if None in (forecast_solar, solcast, actual):
                log.warning(
                    "Fehlende Kernwerte (forecast_solar=%s, solcast=%s, actual=%s) -- "
                    "Entity-IDs in pv_model.py verifiziert? Überspringe diesen Zyklus.",
                    forecast_solar, solcast, actual,
                )
            elif not all(_is_plausible(v) for v in (forecast_solar, solcast, actual)):
                log.warning(
                    "UNPLAUSIBLER Wert erkannt (forecast_solar=%s, solcast=%s, actual=%s), "
                    "> %sW oder negativ -- vermutlich Sensor-/API-Glitch. NICHT gelernt, "
                    "Zyklus übersprungen, um das Modell nicht zu vergiften.",
                    forecast_solar, solcast, actual, PLAUSIBLE_MAX_W,
                )
            else:
                if ecowitt_irradiance is None:
                    log.debug("Ecowitt-Irradiance nicht verfügbar, verwende None für build_features")
                if ecowitt_pressure is None:
                    log.debug("Ecowitt-Pressure nicht verfügbar, verwende None für build_features")

                now = datetime.now(timezone.utc)
                x = build_features(now, forecast_solar, solcast, ecowitt_irradiance, ecowitt_pressure)
                predicted = model.predict(x)
                model.learn(x, actual)
                model.save()

                if predicted is not None:
                    ha.set_state(
                        ENTITY_OUTPUT_CORRECTED,
                        round(predicted, 1),
                        attributes={
                            "unit_of_measurement": "W",
                            "friendly_name": "PV Forecast (ML-korrigiert)",
                            "current_mae_w": round(model.current_mae, 1),
                            "source": "ha-ml-layer / river ARFRegressor + Ecowitt + CSI",
                            "ecowitt_irradiance_w_m2": round(ecowitt_irradiance, 1) if ecowitt_irradiance else None,
                            "ecowitt_pressure_hpa": round(ecowitt_pressure, 1) if ecowitt_pressure else None,
                        },
                    )
                    log.info(
                        "Vorhersage=%.0fW, Ist=%.0fW, laufender MAE=%.1fW | "
                        "Ecowitt: irrad=%.0f W/m² pressure=%.1f hPa",
                        predicted, actual, model.current_mae,
                        ecowitt_irradiance if ecowitt_irradiance else -1,
                        ecowitt_pressure if ecowitt_pressure else -1,
                    )

            # --- NEU: Forecast-Kurve, nur alle CURVE_UPDATE_EVERY_N_CYCLES Zyklen ---
            cycle_count += 1
            if cycle_count % CURVE_UPDATE_EVERY_N_CYCLES == 0:
                try:
                    curve_states = ha.get_states([
                        ENTITY_PV_FORECAST_RAW,
                        ENTITY_SOLCAST_TODAY,
                        ENTITY_SOLCAST_TOMORROW,
                    ])
                    pv_fc = (curve_states.get(ENTITY_PV_FORECAST_RAW) or {}).get("attributes", {}).get("forecasts")
                    sc_today = (curve_states.get(ENTITY_SOLCAST_TODAY) or {}).get("attributes", {}).get("detailedForecast")
                    sc_tomorrow = (curve_states.get(ENTITY_SOLCAST_TOMORROW) or {}).get("attributes", {}).get("detailedForecast")

                    ts_list, fs_list, sc_list = _merge_forecast_arrays(pv_fc, sc_today, sc_tomorrow)
                    if ts_list:
                        curve = model.predict_curve(ts_list, fs_list, sc_list)
                        ha.set_state(
                            ENTITY_OUTPUT_CURVE,
                            round(curve[0][1], 1) if curve else 0,
                            attributes={
                                "unit_of_measurement": "W",
                                "friendly_name": "PV Forecast Curve (ML-korrigiert)",
                                "forecasts": [
                                    {"date": ts.isoformat(), "p_pv_forecast": round(val, 1)}
                                    for ts, val in curve
                                ],
                                "source": "ha-ml-layer / river ARFRegressor (forecast mode)",
                            },
                        )
                        log.info("Forecast-Kurve aktualisiert: %d Zeitschritte", len(curve))
                    else:
                        log.warning(
                            "Forecast-Kurve: keine überlappenden Zeitstempel zwischen "
                            "EMHASS- und Solcast-Forecast gefunden."
                        )
                except Exception:
                    log.exception("Fehler bei Forecast-Kurven-Aktualisierung, nächster Versuch beim nächsten Intervall.")

        except Exception:
            log.exception("Fehler im Poll-Zyklus, versuche es im nächsten Intervall erneut.")

        time.sleep(cfg.poll_interval)


if __name__ == "__main__":
    run()