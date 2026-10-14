import logging
import math
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import pvlib
from river import compose, forest, metrics

log = logging.getLogger("pv_model")

MODEL_PATH = Path("/data/pv_forecast_model.pkl")

# --- Verifiziert via ha_search am 21.08.2026 ---------------------------------
ENTITY_FORECAST_SOLAR = "sensor.p_pv_forecast"                         # Forecast.Solar, Momentanleistung aktuelle Stunde (W)
ENTITY_SOLCAST = "sensor.solcast_pv_forecast_forecast_this_hour"       # Solcast, Energie aktuelle Stunde (Wh) -- ACHTUNG: andere
                                                                        # physikalische Einheit als die anderen beiden (W)! Numerisch
                                                                        # ähnliche Größenordnung (Wh/Stunde ~ mittlere Leistung), aber
                                                                        # nicht exakt gleich. Modell lernt trotzdem eine sinnvolle
                                                                        # Korrektur, da es nur auf Konsistenz der Zahl ankommt -- sauberer
                                                                        # wäre aber Division durch 1h/Normierung, falls sich das Ergebnis
                                                                        # nach den ersten Tagen als systematisch verzerrt zeigt.
ENTITY_ACTUAL_POWER = "sensor.pv_power"                                # Ist-Leistung, aktuell (W) -- bereits im Fast-Charging-Test verifiziert
ENTITY_OUTPUT_CORRECTED = "sensor.pv_ml_corrected_forecast"            # neuer, extern gesetzter Sensor (Output)

# --- Ergänzt am 10.09.2026: Ecowitt WS69/GW3000, echte Wettermessung --------
ENTITY_IRRADIANCE = "sensor.ecowitt_solar_radiation_1f335"             # W/m², echte Messung statt nur Prognose (Entity-ID kann sich bei Hardware-Reconnect ändern!)
ENTITY_PRESSURE = "sensor.ecowitt_pressure_relative"                   # hPa, relativer Luftdruck
# -----------------------------------------------------------------------------

# --- NEU: Forecast-Kurve statt nur Momentanwert (18.09.2026) ----------------
# Für zukünftige Zeitschritte gibt es keine Ecowitt-Live-Messung (irradiance/
# pressure). Ersatz: clear_sky_ghi (pvlib, für jeden Zeitpunkt berechenbar) als
# Irradiance-Platzhalter, Standardatmosphäre (1013.25 hPa) für Druck. Gleiches,
# bereits trainiertes Modell -- kein Re-Training, nur andere Inputs.
DEFAULT_LATITUDE = 48.2
DEFAULT_LONGITUDE = 16.1
DEFAULT_TZ = "Europe/Vienna"
STANDARD_PRESSURE_HPA = 1013.25


def _time_features(ts: datetime) -> dict:
    """Zyklische Zeit-Features statt roher Integer -- vermeidet den künstlichen
    Sprung Dezember->Januar / 23 Uhr->0 Uhr, den ein Baummodell sonst falsch
    als 'weit entfernt' interpretieren würde."""
    hour_angle = 2 * math.pi * ts.hour / 24
    month_angle = 2 * math.pi * ts.month / 12
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "month_sin": math.sin(month_angle),
        "month_cos": math.cos(month_angle),
        "weekday": ts.weekday(),
    }


def _clear_sky_ghi(ts: datetime) -> float:
    """Gemeinsame Clear-Sky-Berechnung für den Live- und den Forecast-Pfad."""
    try:
        location = pvlib.location.Location(
            latitude=DEFAULT_LATITUDE, longitude=DEFAULT_LONGITUDE, tz=DEFAULT_TZ
        )
        times = pd.DatetimeIndex([ts])
        clearsky = location.get_clearsky(times, model="ineichen")
        return float(clearsky["ghi"].values[0]) if len(clearsky) > 0 else 100.0
    except Exception as e:
        log.debug("Clear-sky Berechnung fehlgeschlagen: %s", e)
        return 100.0


def build_features(ts: datetime, forecast_solar: float, solcast: float,
                    irradiance: float, pressure: float) -> dict:
    x = _time_features(ts)
    x["forecast_solar"] = forecast_solar
    x["solcast"] = solcast
    x["forecast_mean"] = (forecast_solar + solcast) / 2
    x["forecast_spread"] = abs(forecast_solar - solcast)
    x["irradiance"] = irradiance
    x["pressure"] = pressure

    # ===== CSI-Features (27.09.2026) =====
    clear_sky_ghi = _clear_sky_ghi(ts)

    # CSI nur wenn irradiance vorhanden
    if irradiance is not None and irradiance >= 0:
        measured_csi = irradiance / max(clear_sky_ghi, 1.0)
        x["measured_csi"] = min(measured_csi, 1.5)  # cap bei 1.5
    else:
        x["measured_csi"] = 0.0

    x["clear_sky_ghi"] = clear_sky_ghi

    return x


def build_features_forecast(ts: datetime, forecast_solar: float, solcast: float) -> dict:
    """Wie build_features(), aber ohne Live-Messwerte -- für Zukunfts-Zeitschritte.
    irradiance wird durch clear_sky_ghi selbst ersetzt (neutrale Annahme: 'erwarteter'
    Wert ohne Wolken-Info), pressure durch die Standardatmosphäre. measured_csi wird
    dadurch automatisch 1.0 (irradiance == clear_sky_ghi), was als neutrales
    Cloud-Korrektursignal für die Zukunft interpretiert wird."""
    x = _time_features(ts)
    x["forecast_solar"] = forecast_solar
    x["solcast"] = solcast
    x["forecast_mean"] = (forecast_solar + solcast) / 2
    x["forecast_spread"] = abs(forecast_solar - solcast)

    clear_sky_ghi = _clear_sky_ghi(ts)

    x["irradiance"] = clear_sky_ghi
    x["pressure"] = STANDARD_PRESSURE_HPA
    x["measured_csi"] = 1.0
    x["clear_sky_ghi"] = clear_sky_ghi

    return x


class PVForecastCorrectionModel:
    """Online-Regressor: lernt datenpunktweise, wie die tatsächliche PV-Leistung
    von den beiden Prognose-Providern (+ Zeitkontext) abweicht. Kein Batch-
    Neu-Training -- jeder neue Ist-Wert aktualisiert das Modell sofort
    (learn_one), passend zum ausdrücklichen Wunsch nach kontinuierlichem
    Lernen statt reiner Ist-Soll-Protokollierung (siehe Doku Abschnitt 2.3)."""

    def __init__(self):
        self.model = compose.Pipeline(
            compose.Select(
                "hour_sin", "hour_cos", "month_sin", "month_cos",
                "weekday", "forecast_solar", "solcast",
                "forecast_mean", "forecast_spread",
                "irradiance", "pressure",
                "measured_csi", "clear_sky_ghi",
            ),
            forest.ARFRegressor(n_models=10, seed=42),
        )
        self.mae = metrics.MAE()
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    state = pickle.load(f)
                    self.model = state["model"]
                    self.mae = state["mae"]
                log.info("Modell geladen, bisheriger MAE: %.1f W", self.mae.get())
            except Exception:
                log.exception("Konnte gespeichertes Modell nicht laden, starte frisch.")

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": self.model, "mae": self.mae}, f)

    def predict(self, x: dict) -> float:
        return self.model.predict_one(x)

    def learn(self, x: dict, y_actual: float):
        y_pred = self.model.predict_one(x)
        if y_pred is not None:
            self.mae.update(y_actual, y_pred)
        self.model.learn_one(x, y_actual)

    def predict_curve(self, timestamps, forecast_solar_values, solcast_values):
        """Wendet das JETZT bereits trainierte Modell auf eine Reihe zukünftiger
        Zeitschritte an. Kein learn() hier -- Lernen bleibt exklusiv der
        Ist-Wert-Schleife in main.py vorbehalten (echte Ist-Werte, nicht
        Prognosen). Gibt eine Liste von (timestamp, korrigierter_wert)-Tupeln
        zurück, in derselben Reihenfolge wie die Eingabe-Listen."""
        out = []
        for ts, fs, sc in zip(timestamps, forecast_solar_values, solcast_values):
            x = build_features_forecast(ts, fs, sc)
            pred = self.predict(x)
            out.append((ts, pred if pred is not None else fs))
        return out

    @property
    def current_mae(self) -> float:
        return self.mae.get()