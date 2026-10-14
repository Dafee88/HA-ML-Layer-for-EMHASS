# HA ML Layer — Skelett (Version 0.1.0, unverifiziert)

Zentrale Online-Learning-Schicht für das Energiemanagement-System, gemäß
Architekturentscheidung aus der Systemdokumentation (Abschnitt 2.3):
River statt scikit-learn, eigener Docker-Add-on nach EMHASS-Vorbild.

## Was das hier ist
Ein **Skelett**, kein fertiges, getestetes Add-on. Es demonstriert die
Architektur (Online-Learning mit River, Polling über HA-REST-API,
externer Ergebnis-Sensor) konkret lauffähig, aber mit drei offenen Punkten,
die vor dem echten Deployment auf dem NUC geklärt werden müssen:

1. **Entity-IDs verifiziert (21.08.2026).** `sensor.p_pv_forecast` (Forecast.Solar,
   Momentanleistung) und `sensor.solcast_pv_forecast_forecast_this_hour`
   (Solcast, Momentanleistung) liefern beide Watt-Werte für die aktuelle
   Stunde, `sensor.pv_power` die Ist-Leistung — bereits im Fast-Charging-Test
   verifiziert. Damit lernt das Modell direkt auf Momentanleistung statt nur
   auf Tagessummen (siehe Punkt 2, das war vorher ein offener Punkt, ist es
   jetzt nicht mehr).
2. **Granularität ist jetzt hoch genug für echtes Online-Learning:** Poll alle
   5 Minuten liefert ~100+ Trainings-Datenpunkte pro Tag statt nur einem
   (Tagessumme), was das "nur ~10-14 Tage Historie"-Problem aus Doku Abschnitt
   4.5 entschärft — mehr Datenpunkte pro Tag kompensiert die kurze Zeitspanne.
3. **InfluxDB-Anbindung fehlt noch komplett.** Das Skelett liest nur über
   die HA-REST-API (aktuelle States), nicht aus InfluxDB. Für Backtesting
   mit echter Historie (relevant für das 2-Jahres-Retention-Ziel) müsste
   ein separates Init-Skript die InfluxDB-Query-API (Port 8086) anfragen —
   **zusätzlich blockiert**, weil das dokumentierte Add-on-Passwort nicht
   gegen die echte InfluxDB-Instanz funktioniert (Systemdoku Abschnitt 6,
   Punkt 2) — noch ungeklärt.

## Deployment (grober Ablauf)
1. Diesen Ordner nach `/addons/ha_ml_layer/` auf dem NUC kopieren
   (lokales Add-on-Repository)
2. In Home Assistant: Einstellungen → Add-ons → Add-on-Store → oben rechts
   neu laden → "HA ML Layer" sollte unter "Lokale Add-ons" erscheinen
3. Vor dem Start: Entity-IDs in `app/pv_model.py` korrigieren
4. Add-on installieren und starten, Logs beobachten
5. Neuen Sensor `sensor.pv_ml_corrected_forecast` im Dashboard beobachten,
   `current_mae_w`-Attribut zeigt den laufenden mittleren Fehler

## Nicht enthalten (bewusst, für spätere Schritte)
- Rückschreiben in EMHASS-Konfiguration (aktuell nur Anzeige-Sensor,
  passend zum Governance-Prinzip "LLM/Modell schlägt vor, ändert nicht
  autonom EMHASS-Kernkonfig")
- Gebäude-Wärmemodell (RC-Modell) — als zweites Modell im selben Dienst
  vorgesehen (Abschnitt 2.3), noch nicht begonnen
- Persistenz-Volume-Konfiguration in `config.yaml` für `/data/` — Standard-
  Add-on-Verhalten sollte das automatisch abdecken, aber nicht getestet

