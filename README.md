# HA ML Layer für EMHASS (Version 0.2.0)

Home-Assistant-Add-on mit einer Online-Learning-Schicht (River) für das
Energiemanagement mit EMHASS. Es läuft als eigener Docker-Dienst, fragt
Home Assistant alle 5 Minuten über die REST-API ab, lernt fortlaufend aus
Messwerten und schreibt seine Ergebnisse als Sensoren zurück.

Enthalten sind zwei Modelle:

| Modell | Add-on-Option | Ausgabe |
|---|---|---|
| PV-Prognose-Korrektur (Day-Ahead + Nowcast) | `pv_forecast_correction` | `sensor.pv_ml_corrected_forecast`, `sensor.pv_ml_corrected_forecast_curve` |
| Gebäude-Wärmemodell (pro Etage) | `building_thermal` | `sensor.building_thermal_downstairs`, `sensor.building_thermal_upstairs` |

Änderungen je Version: siehe [CHANGELOG.md](CHANGELOG.md).

---

## PV-Prognose-Korrektur (`app/pv_model.py`)

### Eingänge
| Entity | Bedeutung |
|---|---|
| `sensor.solcast_pv_forecast_forecast_today` / `_tomorrow` | Solcast `detailedForecast`: P50/P10/P90 in kW, 30-min-Mittelleistung |
| `sensor.goodwe_pv_power_effective` (Fallback `sensor.pv_power`) | Ist-Leistung in W |
| `sensor.ecowitt_solar_radiation_1f335` | Einstrahlung in W/m² (Entity-ID kann sich nach einem Gateway-Reconnect ändern!) |

Bewusst **nicht** verwendet: `sensor.p_pv_forecast`. Das ist die Ausgabe von
EMHASS selbst; sobald EMHASS die ML-Kurve bekommt, würde das Modell von seiner
eigenen Prognose lernen.

Standort (Breite 48,2 / Länge 16,1, `Europe/Vienna`) und Entity-IDs stehen als
Konstanten oben in `app/pv_model.py`.

### Zwei Modelle
1. **Day-Ahead-Modell.** Korrigiert Solcast für die nächsten 48 × 30 min.
   Es nutzt nur Features, die es auch für die Zukunft gibt: Solcast
   P50/P10/P90, Clear-Sky-Einstrahlung, Sonnenhöhe in der Slot-Mitte,
   Tages- und Jahreszeit. Training und Prognose verwenden denselben
   Feature-Builder.
2. **Nowcast-Modell.** Korrigiert den *aktuellen* Wert mit der live gemessenen
   Ecowitt-Einstrahlung (Clear-Sky-Index und seine Schwankung über 30 min).
   Seine Wirkung auf die Kurve klingt mit `exp(−Δt / 1 h)` ab, weil ein Sensor
   keine Wolken sieht, die noch nicht da sind.

### Lernverfahren
- **Residual statt Absolutwert:** Beide Modelle lernen `Ist − Basis`
  (Day-Ahead: Solcast P50, Nowcast: Day-Ahead-Wert). Die Kurvenform von
  Solcast bleibt erhalten.
- **Ein Lernschritt pro Slot:** Das Day-Ahead-Modell lernt einmal je 30-min-Slot
  auf dem Mittel der Ist-Werte (mindestens 60 % Abdeckung), also auf genau der
  Größe, die Solcast prognostiziert und EMHASS verwendet.
- **Skalierte Features:** `StandardScaler` vor dem `ARFRegressor`. Ohne
  Skalierung divergieren dessen lineare Blattmodelle bei Watt-Werten.
- **Gating gegen die Basis:** Ein Modell wird erst nach einer Mindestzahl von
  Samples verwendet (Day-Ahead 60 Slots ≈ 3–4 Tage, Nowcast 150 × 5 min) und
  nur, solange sein rollierender MAE kleiner ist als der der Basis auf denselben
  Samples. Sonst wird die Basis unverändert durchgereicht. Die ML-Schicht
  macht die Prognose also nicht dauerhaft schlechter als Solcast.
- Gelernt wird nur bei Sonnenhöhe ≥ 5° und plausiblen Ist-Werten (0–15 kW).

### Ausgaben
`sensor.pv_ml_corrected_forecast`: aktueller korrigierter Wert in W. Attribute:

| Attribut | Bedeutung |
|---|---|
| `dayahead_active`, `nowcast_active` | `true`, wenn das Modell gerade verwendet wird (besser als seine Basis) |
| `mae_dayahead_w` / `mae_solcast_w` | rollierender MAE des Day-Ahead-Modells / von Solcast auf denselben Slots |
| `mae_nowcast_w` / `mae_nowcast_baseline_w` | dasselbe für den Nowcast gegen den Day-Ahead-Wert |
| `samples_dayahead`, `samples_nowcast` | Anzahl gelernter Samples |
| `dayahead_now_w`, `solcast_p50_now_w`, `nowcast_factor` | Zwischenwerte des aktuellen Zyklus |

`sensor.pv_ml_corrected_forecast_curve`: Attribut `forecasts` mit 48 Einträgen
`{"date": …, "p_pv_forecast": …}`. Diese Kurve lesen die EMHASS-`rest_command`s.

Gelernte Modelle liegen in `/data/pv_dayahead_model_v3.pkl` und
`/data/pv_nowcast_model_v3.pkl`.

---

## Gebäude-Wärmemodell (`app/building_thermal.py`)

Lernt pro Etage aus Messwerten:
- **Abkühlkonstante k** [1/h] bei Heizung AUS: `dT/dt = −k · (T_innen − T_außen)`
- **Heizrate h** [K/h] bei Heizung EIN: `dT/dt = h − k · (T_innen − T_außen)`

| Etage | Temperatur | Heizung EIN |
|---|---|---|
| unten | `sensor.eve_room_e9ff_temperature` | `switch.shelly1pmminig3_5432044e9dfc` = `on` |
| oben | `sensor.eve_room_e49e_temperature` | `input_select.heating_stage` = `upstairs` oder Tado `hvac_action` = `heating` |

Außentemperatur: Ecowitt, Fallback Vitocal (MQTT).

Ein Lernfenster dauert mindestens 60 min bei unverändertem Heizzustand. Es wird
verworfen, wenn
- das Polling eine Lücke hatte (Add-on-Neustart, HA nicht erreichbar),
- die Einstrahlung über 50 W/m² lag (Solargewinne verfälschen k und h),
- beim Abkühlen die Differenz innen/außen unter 3 K lag.

Werte werden per gleitendem Mittel (α = 0,1) geglättet und in
`/data/building_thermal.json` gespeichert. Der Sensorzustand ist `calibrated`
ab 10 Abkühl- und 5 Heiz-Samples, vorher `learning`.

---

## Installation und Updates (Add-on-Store)
1. Einstellungen → Add-ons → Add-on-Store → ⋮ oben rechts → **Repositories**.
2. `https://github.com/Dafee88/HA-ML-Layer-for-EMHASS` hinzufügen.
3. Add-on **HA ML Layer** installieren. Es wird auf dem Gerät aus dem
   `Dockerfile` gebaut (kein vorgefertigtes Image).
4. Modelle in den Optionen aktivieren, starten, Logs beobachten.
5. Bei einer anderen Anlage: Entity-IDs und Standort in
   `ha_ml_layer/app/pv_model.py` und `ha_ml_layer/app/building_thermal.py` anpassen.

Updates: Home Assistant zeigt ein Update an, sobald `version` in
`ha_ml_layer/config.yaml` auf `main` erhöht wurde. Jede Code-Änderung braucht
deshalb eine neue Versionsnummer, sonst kommt sie nicht an.

Umstieg vom lokalen Add-on (`/addons/ha_ml_layer/`): Das Store-Add-on ist für
Home Assistant ein eigenes Add-on mit eigenem `/data`. Lokales Add-on stoppen
und deinstallieren, sonst schreiben beide dieselben Sensoren. Die Modelle
lernen danach neu.

### Konfiguration (Tab „Konfiguration“ des Add-ons)
Alle Sensoren werden im Formular eingetragen, nichts mehr im Code. Jedes Feld hat
dort eine Beschreibung. Nach dem Speichern das Add-on neu starten. Fehlerhafte
Eingaben (z. B. ungültige Entity-ID) stehen beim Start als `Configuration:` im Log.

| Bereich | Optionen |
|---|---|
| Allgemein | `poll_interval_seconds` (60–3600, empfohlen 300), `log_level`, `pv_enabled`, `thermal_enabled` |
| Standort | `latitude`, `longitude`, `timezone` (für Sonnenstand und Clear-Sky) |
| PV | `pv_solcast_today`, `pv_solcast_tomorrow` (Solcast-Sensoren mit `detailedForecast`), `pv_actual_power` (W), optional `pv_actual_power_fallback`, `pv_irradiance` (W/m², aktiviert den Nowcast) |
| Wärmemodell | `thermal_outdoor_temp`, optional `thermal_outdoor_temp_fallback`, optional `thermal_heating_stage`, `thermal_floors` (pro Etage `name`, `temperature_sensor` und `heating_switch` oder `climate`) |
| InfluxDB | `influxdb_enabled`, `influxdb_host` (InfluxDB-Add-on: `a0d7b954-influxdb`), `influxdb_port`, `influxdb_ssl`, `influxdb_database`, `influxdb_username`, `influxdb_password`, `influxdb_dayahead_snapshot_hour` |

Ausgabe-Sensoren sind fest, weil EMHASS sie liest: `sensor.pv_ml_corrected_forecast`,
`sensor.pv_ml_corrected_forecast_curve`, `sensor.building_thermal_<etage>`.

### InfluxDB (optional)
Schreibt über die HTTP-Schnittstelle von InfluxDB 1.x in eine bestehende Datenbank:

| Measurement | Wann | Inhalt |
|---|---|---|
| `ml_layer_pv` | jeder Zyklus | Solcast P50/P10/P90 des aktuellen Slots, Day-Ahead-, Nowcast- und veröffentlichter Wert, Ist-Leistung, Einstrahlung, Clear-Sky, Sonnenhöhe, MAE, aktiv-Flags |
| `ml_layer_pv_slot` | je fertigem 30-min-Slot | Mittel der Ist-Leistung und die Solcast-Werte, auf denen das Day-Ahead-Modell gelernt hat |
| `ml_layer_dayahead` | einmal täglich ab `influxdb_dayahead_snapshot_hour` | die 48 Slots von morgen: Solcast P50/P10/P90 und ML-Kurve, Tag `issued` |

Damit liegen die Daten für einen echten Backtest vor (der Home-Assistant-Recorder
speichert die Solcast-Kurven nicht). Ein Schreibfehler hält das Add-on nicht an, er
steht höchstens alle 10 Minuten einmal im Log.

### Update auf 0.2.0
Die PV-Modelle lernen ab dem Update neu (`*_v3.pkl`). Alte `*.pkl` in `/data/`
werden nicht mehr gelesen und können gelöscht werden. Etwa 3–4 Tage lang wird
Solcast unverändert durchgereicht, danach übernimmt das Modell, sobald
`dayahead_active` auf `true` steht.

---

## Grenzen und offene Punkte
- **Vorlaufzeit:** Das Day-Ahead-Modell lernt mit dem zuletzt aktualisierten
  Solcast-Wert eines Slots (Vorlauf fast 0 h), wird aber auf Werte mit bis zu
  24 h Vorlauf angewendet. Abhilfe: Feature-Snapshots zum Prognosezeitpunkt
  speichern und mit einem Feature `lead_h` lernen.
- **Abregelung:** Bei voller Batterie, Einspeiselimit oder negativen Preisen
  lernt das Modell die Abregelung als „schlechtes Wetter". Dafür fehlt noch
  eine Entity mit dem Begrenzungsstatus des Wechselrichters.
- **Wärmemodell:** Die Fußbodenheizung wirkt stundenlang nach. Besser wären
  2–3 h Sperrzeit nach dem Umschalten und eine Regressionsgerade über das ganze
  Fenster statt zweier Endpunkte (Sensorauflösung 0,1 K).
- **Keine InfluxDB-Anbindung:** Es wird nur über die HA-REST-API gelesen. Ein
  Backfill aus InfluxDB würde den Kaltstart verkürzen. Blockiert, weil das
  dokumentierte Add-on-Passwort nicht gegen die InfluxDB-Instanz funktioniert.
- **Nur Anzeige:** Die Modelle ändern keine EMHASS-Konfiguration selbst. Die
  Sensoren werden per `POST /api/states` gesetzt und landen daher nicht in der
  Langzeitstatistik.
