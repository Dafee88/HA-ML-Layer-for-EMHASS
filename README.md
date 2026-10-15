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


## Analyse & Verbesserungen (15.10.2026)

### Umgesetzt
**PV-Prognose** (`app/pv_model.py`, `app/main.py`; Modelldateien jetzt `*_v2.pkl`,
d. h. das Lernen startet neu — die alten v1-Modelle haben ein anderes Ziel und
sind inkompatibel):
1. **Residual statt Absolutwert.** Beide Modelle lernen `Ist − Basis`
   (Day-Ahead: Solcast P50, Nowcast: Day-Ahead-Wert). Baum-Ensembles (ARF)
   können nicht extrapolieren und erzeugen bei Absolutwerten Treppenkurven;
   ein Residual behält die Kurvenform von Solcast und fällt sauber darauf zurück.
2. **Lernen auf dem Slot-Mittelwert.** Solcast liefert 30-min-Mittelleistung,
   EMHASS verbraucht 30-min-Werte — vorher lernte das Day-Ahead-Modell alle
   5 min auf Momentanwerten, wobei die 6 Samples eines Slots *identische*
   Features hatten (6-fach gewichtetes Rauschen). Jetzt: `SlotAccumulator`
   mittelt die Ist-Werte und lernt einmal pro Slot (≥ 60 % Abdeckung nötig).
3. **Sonnengeometrie in Slot-Mitte** statt Slot-Beginn (Sonnenaufgangs-Slot
   wurde vorher fälschlich auf 0 W gesetzt).
4. **Gating gegen die Baseline.** Ein Modell wird nur benutzt, solange sein
   rollierender MAE (Day-Ahead ~300 Slots, Nowcast ~600 Samples) *besser* ist
   als der der Baseline auf denselben Samples. Neue Attribute:
   `mae_solcast_w`, `mae_nowcast_baseline_w`, `dayahead_active`,
   `nowcast_active` — damit ist erstmals sichtbar, ob das ML Solcast schlägt.
5. **CSI-Historie** des Nowcasts wird nach Lücken > 20 min verworfen (vorher
   flossen Werte vom Vorabend in den ersten Morgenwert ein).
6. **Atomares Speichern** (tmp + `os.replace`) — ein Absturz beim Schreiben
   löscht kein gelerntes Modell mehr.

**Gebäude-Wärmemodell** (`app/building_thermal.py`):
1. Fenster wird verworfen, wenn das Polling eine Lücke hatte (Add-on-Neustart,
   HA down) — vorher konnte ein Fenster über Stunden mit unbeobachtetem
   Heizungswechsel laufen.
2. Fenster wird verworfen, wenn die Einstrahlung (Ecowitt) > 50 W/m² war —
   Solargewinne würden k zu klein und h zu groß schätzen.
3. Atomares Speichern.

### Offene Empfehlungen (nicht umgesetzt)
- **Vorlaufzeit-Mismatch (größter verbleibender Hebel):** Das Day-Ahead-Modell
  lernt mit dem *zuletzt* aktualisierten Solcast-Wert eines Slots (Vorlauf
  ~0 h), angewendet wird es aber auf Werte mit bis zu 24 h Vorlauf. Besser:
  Feature-Snapshots der Kurve zum Prognosezeitpunkt speichern (z. B. zum
  EMHASS-Day-Ahead-Lauf) und auf genau diesen mit Feature `lead_h` lernen.
- **Abregelung herausfiltern:** Wenn die Anlage abgeregelt wird (Batterie voll
  + Einspeiselimit, negative Preise), lernt das Modell die Abregelung als
  „schlechtes Wetter". Samples mit aktiver Begrenzung sollten übersprungen
  werden (Entity für Export-Limit/Wechselrichter-Status nötig).
- **Wärmemodell:** Fußbodenheizung hat Stunden Nachlauf — nach dem Ausschalten
  eher 2–3 h statt 60 min verwerfen; Steigung per linearer Regression über das
  ganze Fenster statt aus zwei Endpunkten (0,1-K-Sensorauflösung ergibt sonst
  ±0,1 K/h Rauschen, bei k·ΔT ≈ 0,2–0,5 K/h erheblich).
- **InfluxDB-Backfill** (siehe oben, Punkt 3) würde den Kaltstart von ~3–4
  Tagen auf null reduzieren.
