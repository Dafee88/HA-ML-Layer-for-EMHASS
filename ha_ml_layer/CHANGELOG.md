# Changelog

## 0.7.0 — 2026-10-27

### Neu: Prognose-Prüfung pro Tag
- `sensor.pv_ml_forecast_accuracy`: Am Abend (Snapshot-Stunde, Standard 20 Uhr)
  wird die Prognose für morgen notiert (ML und Solcast, kWh) – das, womit EMHASS
  plant. Nach Mitternacht wird sie mit der gemessenen PV-Energie verglichen.
  Zustand = ML-Abweichung des letzten Tages in %, Attribute u. a. Ist/ML/Solcast
  in kWh, Abregelungs- und Minderertrags-Stunden, 7-Tage-Mittel beider Fehler,
  Verlauf. Tage mit < 90 % Messwerten werden nicht bewertet.

### Neu: Minderertrag erkennen (Schnee, Verschattung, Störung)
- Liegt die PV-Leistung ohne Abregelung unter 50 % des Stations-Potenzials
  (mind. 300 W Potenzial), lernen die Prognosemodelle diese Werte nicht – sonst
  würde z. B. Schnee auf den Modulen als „Solcast zu hoch“ gelernt.
- Beginnt so eine Phase bei ≤ 4 °C Außentemperatur (unbekannt zählt als kalt),
  lernt auch das Stationsmodell nicht (möglicher Schnee); warme Phasen lernt das
  Anlagenmodell (Verschattung). Nach 60 min Warnung im Log; Attribute `low_output`,
  `low_output_snow_possible`, `low_output_since`.
- Simulation (8 Novembertage, 2 Schneetage mit 5 % Ertrag): Stationsfaktor und
  Modelle bleiben an den Schneetagen unverändert, danach normales Lernen; normale
  Tage (Juni-Simulation) liefern exakt dieselben Ergebnisse wie 0.6.3.

## 0.6.3 — 2026-10-26

### Fix: Zeitumstellung
- In der Nacht der Zeitumstellung (25.10.2026) brach der PV-Zyklus rund 70 min lang
  bei der kWh-Berechnung ab („Tz-aware datetime.datetime cannot be converted …“):
  Die Solcast-Slots desselben Tages hatten gemischte Offsets (+02:00 / +01:00).
  Die 48-Slot-Kurve für EMHASS wurde weiter geschrieben; es fehlten in dieser Zeit
  die kWh-Sensoren und die InfluxDB-Zyklusdaten.
- Sonnenstand/Clear-Sky rechnen jetzt immer in UTC. Getestet mit den Slots
  02:30 +02:00 und 02:00 +01:00 vom 25.10.; Simulation unverändert.

## 0.6.2 — 2026-10-21

### Fix: Historie nur aus der aktuellen Solcast-Einrichtung
- Problem in 0.6.1 (live gesehen): Die Historie enthielt Wochen, in denen Solcast
  anders eingerichtet war (eine statt zwei Dachflächen, Tagesprognosen bis 50 kWh
  bei real ~17 kWh). Das Modell lernte daraus „Solcast liegt ×2,5 zu hoch“ und
  drückte die aktuelle, korrekte Prognose zu stark (Rolling-MAE Solcast 1309 W auf
  den alten Daten).
- Jetzt erkennt das Add-on Änderungen der Solcast-Dachflächen (pro Fläche ein
  Attribut mit der Resource-ID; Fläche hinzugefügt oder entfernt) und lernt nur
  die Historie danach.
- Neue Option `influxdb_history_start` (JJJJ-MM-TT) für andere Änderungen
  (neue Module, geänderte Geometrie/Leistung bei solcast.com).
- Das Modell wird nach dem Update einmal neu aufgebaut (Marker v2); als Backup
  bleibt das ursprüngliche Modell vor dem ersten Nachlernen erhalten.
- Test (echte InfluxDB 1.x): Flächenwechsel korrekt erkannt, nur Slots danach
  gelernt.

## 0.6.1 — 2026-10-21

### Lernen aus der InfluxDB-Historie
- Neu (Option `influxdb_learn_from_history`, Standard an): Beim ersten Start baut
  das Add-on das Day-ahead-Modell einmalig aus der InfluxDB-Historie neu auf –
  Solcast-30-min-Kurven (`detailedForecast`, wie von der HA-InfluxDB-Integration
  gespeichert) und PV-Leistung der letzten bis zu 90 Tage, danach die eigenen
  Slot-Datensätze (`ml_layer_pv_slot`), in zeitlicher Reihenfolge. Damit muss das
  Modell nicht wochenlang live Slots sammeln.
- Je Slot wird die letzte vor Slot-Ende gespeicherte Solcast-Kurve verwendet –
  dieselben Werte, mit denen auch live gelernt wird.
- Abgeregelte Slots werden übersprungen: Abregelungs-Indikator an, oder (mit
  Wetterstation) Ist-Leistung unter 60 % der Stations-Potenzialleistung.
- Die MAE-Schranke gilt unverändert: das Modell ersetzt Solcast nur, wenn es
  nachweislich besser ist. Altes Modell bleibt als `*.pre_history.pkl` erhalten,
  Ergebnis in `/data/dayahead_history_v1.json` und im Log (`History: ...`).
- Test gegen eine echte InfluxDB 1.x mit Daten im Format der HA-Integration
  (Python-Darstellung mit `datetime`/`ZoneInfo`, wie live in InfluxDB gesehen):
  30 Tage, 541 Historien- + 57 eigene Slots, 35 abgeregelte Slots übersprungen.

### Sonstiges
- Add-on-Beschreibung auf Englisch.

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
