# Changelog

## 0.6.0 — 2026-10-21

Das Gebäudemodell lernt nach dem Update neu (`/data/building_thermal_v2.json`);
die alten Werte aus 60-min-Fenstern waren durch die Sensorauflösung zu ungenau.

### Für EMHASS
- Neu: `sensor.pv_ml_forecast_tomorrow_kwh` und `sensor.pv_ml_forecast_today_remaining_kwh`
  (für `soc_final` / Akku-Reserve), mit `source` (`ml`, `solcast`, `station`, `none`)
  und `solcast_kwh` zum Vergleich.
- Neu: `sensor.building_thermal_emhass` mit `cooling_constant` und `heating_rate`
  in EMHASS-Einheiten und `valid`-Flag.
- Nach einem Home-Assistant-Neustart werden alle Sensoren binnen einer Minute neu
  geschrieben (vorher bis zu einem Poll-Intervall, Standard 5 min).

### Gebäudemodell
- Abkühlkonstante aus Abkühlphasen ≥ 3 h mit Ausgleichsgerade, 3 h Sperre nach
  dem Abschalten der Heizung, Qualitätsprüfung (Temperaturänderung, R²).
- Heizrate aus einer Tages-Energiebilanz (Ruhephase zu Ruhephase), normiert auf
  die Nennleistung (`thermal_nominal_power_w`); sonnige Tage werden übersprungen.
- Heizsignal auch über einen Leistungssensor (`heating_power_sensor`, Schwelle
  `thermal_heating_power_threshold_w`).
- Simulation (Haus mit k = 0,005/h, 2,0 K/h bei 1500 W, Fußbodenheizung mit 1,5 h
  Verzögerung, 0,1-K-Sensor): gelernt 0,00484/h und 1,94 K/h. Das alte Verfahren
  über die Heizphase hätte 1,55 K/h (−23 %) ergeben.

### Sonstiges
- Log-Meldungen des HA-Clients auf Englisch.

## 0.5.2 — 2026-10-21

### InfluxDB
- Beim Start (mit `influxdb_enabled`) prüft das Add-on nur lesend, welche Historie
  der eingetragenen Sensoren in InfluxDB liegt, und schreibt pro Sensor eine Zeile
  ins Log: Measurement, erster/letzter Wert, Felder und ob die Solcast-30-min-Kurve
  mitgespeichert wird. Grundlage für das spätere Lernen aus der Historie.
- Fehlermeldungen heißen jetzt „InfluxDB request failed“ (galt auch für Abfragen).
- Getestet gegen eine echte InfluxDB 1.x mit Daten im Format der HA-Integration.

## 0.5.1 — 2026-10-20

### Dokumentation
- Add-on-Seite im Store (`README.md`) listet alle vorher zu installierenden
  Integrationen und Add-ons mit Links (Pflicht und optional).
- Neuer Tab „Documentation“ (`DOCS.md`): Voraussetzungen, Konfiguration,
  Ausgaben, Beispiel-`rest_command` für EMHASS, Beispiel-Abregelungs-Sensor, InfluxDB.
- README auf GitHub mit derselben Liste; Feldbeschreibungen nennen die
  passenden Solcast- und Forecast.Solar-Sensoren.
- Am Code hat sich nichts geändert.

## 0.5.0 — 2026-10-20

PV-Modelle lernen nach dem Update neu (Modelldateien `*_v4.pkl`).

### Wetterstation
- Anlagenmodell Einstrahlung → PV-Leistung (mit Sonnenazimut für mehrere
  Dachrichtungen), gesperrt bis es einen einfachen W-pro-W/m²-Faktor schlägt.
- Ersatz-Istwert bei Abregelung (`pv_curtailed_entity`): die Modelle lernen dann
  die mögliche statt der abgeregelten Leistung.
- Fallback: Slots ohne API-Daten werden aus der Station berechnet (Bewölkung jetzt
  → typische Bewölkung der Uhrzeit). Kurve bleibt bei API-Ausfall vollständig.
- Typische Bewölkung je Uhrzeit wird gelernt; optionaler Referenzsensor
  (`pv_reference_irradiance`) für den Kalibrierfaktor.

### Prognose
- Zweite Prognose (z. B. Forecast.Solar) als Tageswerte im Day-Ahead-Modell,
  mehrere Dachflächen per Komma.
- Festes 48-Slot-Raster; fehlt der aktuelle Solcast-Slot, läuft der Zyklus weiter.
- nMAE in % der Anlagenleistung (`pv_peak_power_kw`).
- Neue Attribute und InfluxDB-Felder: `forecast_source`, `fallback_slots`,
  `station_*`, `curtailed`, `learning_target`, `irradiance_vs_reference`, nMAE.

Simulation (16 Tage, zwei Dachrichtungen, Sensor −18 %, Abregelung, 2 Tage API-Ausfall):
Fehler 3 h voraus 208 W normal, 321 W im Ausfall (vorher keine Kurve);
mit Abregelungs-Sensor 208 W statt 407 W.

## 0.4.0 — 2026-10-20

Für jede Anlage nutzbar: keine anlagenspezifischen Werte mehr im Repo.

- Vorbelegte Sensoren sind neutrale Beispielnamen (Muster wie bei EMHASS:
  `sensor.power_photovoltaics`, `sensor.outdoor_temperature`, eine Beispiel-Etage
  `ground_floor`). Bestehende, gespeicherte Konfigurationen bleiben unverändert.
- `latitude`, `longitude`, `timezone` sind optional; leer = Werte aus der
  Home-Assistant-Konfiguration (`/api/config`).
- Einstrahlungs- und Fallback-Sensoren sind standardmäßig leer (optional).
- Sensor-Attribut `ecowitt_irradiance_w_m2` heißt jetzt `irradiance_w_m2`;
  Anzeigenamen der Ausgabe-Sensoren auf Englisch.
- Icon wieder nur das Hirn-Symbol; das Logo mit Schrift bleibt auf der Add-on-Seite.

## 0.3.2 — 2026-10-20
- Add-on heißt jetzt „HA ML Layer for EMHASS“.
- Neues Icon: Schriftzug mit Hirn-Symbol (Quelle `assets/icon.svg`).

## 0.3.1 — 2026-10-20
- Icon (`icon.png`, 128 × 128) und Logo (`logo.png`, 250 × 100) für den
  Add-on-Store; Logo auch oben in der README. Quellen in `assets/`.

## 0.3.0 — 2026-10-20

### Konfiguration
- Alle Eingangs-Sensoren, Standort und Etagen des Wärmemodells werden im
  Konfigurations-Tab des Add-ons eingetragen (vorher fest im Code). Beschriftungen
  und Erklärungen in `translations/en.yaml`.
- Ungültige Eingaben werden beim Start im Log gemeldet.
- Altes Optionsformat (`models`-Liste) wird weiter verstanden.
- `input_select.heating_stage` ist jetzt optional (`thermal_heating_stage`, Standard leer);
  die Etage gilt als beheizt, wenn der Zustand ihrem Namen entspricht.

### InfluxDB
- Optionales Schreiben in InfluxDB 1.x: `ml_layer_pv` (jeder Zyklus),
  `ml_layer_pv_slot` (je 30-min-Slot), `ml_layer_dayahead` (abendlicher
  Schnappschuss der Prognose für morgen) als Datenbasis für Backtests.

## 0.2.0 — 2026-10-20

PV-Modelle lernen nach dem Update neu (Modelldateien `*_v3.pkl`).

### Verteilung
- Repo ist jetzt ein Home-Assistant-Add-on-Repository (`repository.yaml`,
  Add-on im Ordner `ha_ml_layer/`); Installation und Updates über den Add-on-Store.

### PV-Prognose
- Modelle lernen das Residual auf Solcast bzw. den Day-Ahead-Wert statt
  absoluter Watt.
- Day-Ahead-Modell lernt einmal pro 30-min-Slot auf dem Mittel der Ist-Werte
  statt alle 5 min auf Momentanwerten.
- Sonnengeometrie in der Slot-Mitte statt am Slot-Beginn.
- `StandardScaler` vor dem `ARFRegressor`: Die linearen Blattmodelle
  divergierten ohne Skalierung (live gemessener MAE ~50 kW).
- Modelle werden nur verwendet, solange ihr rollierender MAE besser ist als der
  von Solcast bzw. dem Day-Ahead-Wert.
- Neue Sensor-Attribute: `mae_solcast_w`, `mae_nowcast_baseline_w`,
  `dayahead_active`, `nowcast_active`.
- Nowcast verwirft CSI-Werte, die älter als 20 min sind.
- Modelle werden atomar gespeichert.

### Gebäude-Wärmemodell
- Lernfenster mit Polling-Lücke oder Einstrahlung > 50 W/m² werden verworfen.
- Zustand wird atomar gespeichert.

### Dokumentation
- README an den aktuellen Stand angepasst.

## 0.1.0

- Erste Version: PV-Korrektur (Day-Ahead + Einstrahlungs-Nowcast) mit River und
  Gebäude-Wärmemodell (Newton-Abkühlung).
