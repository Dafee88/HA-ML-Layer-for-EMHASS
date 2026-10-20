# Changelog

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

- Erste Version: PV-Korrektur (Day-Ahead + Ecowitt-Nowcast) mit River und
  Gebäude-Wärmemodell (Newton-Abkühlung).
