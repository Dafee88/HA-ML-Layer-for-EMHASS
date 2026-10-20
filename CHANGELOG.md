# Changelog

## 0.2.0 — 2026-10-20

PV-Modelle lernen nach dem Update neu (Modelldateien `*_v3.pkl`).

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
