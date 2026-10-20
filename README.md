<p align="center"><img src="assets/logo.png" alt="HA ML Layer for EMHASS" width="500"></p>

# HA ML Layer für EMHASS

Home-Assistant-Add-on mit einer Online-Learning-Schicht (River) für das
Energiemanagement mit EMHASS. Es läuft als eigener Docker-Dienst, fragt
Home Assistant alle 5 Minuten über die REST-API ab, lernt fortlaufend aus
Messwerten und schreibt seine Ergebnisse als Sensoren zurück.

Enthalten sind zwei Modelle:

| Modell | Add-on-Option | Ausgabe |
|---|---|---|
| PV-Prognose-Korrektur (Day-Ahead + Nowcast) | `pv_enabled` | `sensor.pv_ml_corrected_forecast`, `sensor.pv_ml_corrected_forecast_curve` |
| Gebäude-Wärmemodell (pro Etage) | `thermal_enabled` | `sensor.building_thermal_<etage>` |

Voraussetzungen: die [Solcast-Integration](https://github.com/BJReplay/ha-solcast-solar) (liefert die 30-min-Kurve im Attribut `detailedForecast`) und ein Sensor für die aktuelle PV-Leistung in W. Optional: ein Einstrahlungssensor (W/m², z. B. Wetterstation), Raum- und Außentemperaturen für das Wärmemodell, InfluxDB 1.x.

Änderungen je Version: siehe [CHANGELOG.md](CHANGELOG.md).

---

## PV-Prognose-Korrektur (`ha_ml_layer/app/pv_model.py`)

### Eingänge
| Option | Beispiel | Bedeutung |
|---|---|---|
| `pv_solcast_today` / `pv_solcast_tomorrow` | `sensor.solcast_pv_forecast_forecast_today` / `_tomorrow` | Solcast `detailedForecast`: P50/P10/P90 in kW, 30-min-Mittelleistung |
| `pv_actual_power` (optional `pv_actual_power_fallback`) | `sensor.power_photovoltaics` | Ist-Leistung in W |
| `pv_irradiance` (optional) | `sensor.solar_radiation` | Einstrahlung in W/m², schaltet den Nowcast frei |

Nicht als Eingang verwenden: `sensor.p_pv_forecast`. Das ist die Ausgabe von
EMHASS selbst; sobald EMHASS die ML-Kurve bekommt, würde das Modell von seiner
eigenen Prognose lernen.

Standort und Zeitzone kommen aus den Optionen `latitude`, `longitude`, `timezone`;
bleiben sie leer, übernimmt das Add-on die Werte aus der Home-Assistant-Konfiguration.

### Zwei Modelle
1. **Day-Ahead-Modell.** Korrigiert Solcast für die nächsten 48 × 30 min.
   Es nutzt nur Features, die es auch für die Zukunft gibt: Solcast
   P50/P10/P90, Clear-Sky-Einstrahlung, Sonnenhöhe in der Slot-Mitte,
   Tages- und Jahreszeit. Training und Prognose verwenden denselben
   Feature-Builder.
2. **Nowcast-Modell.** Korrigiert den *aktuellen* Wert mit der live gemessenen
   Einstrahlung (Clear-Sky-Index und seine Schwankung über 30 min).
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

## Gebäude-Wärmemodell (`ha_ml_layer/app/building_thermal.py`)

Lernt pro Etage aus Messwerten:
- **Abkühlkonstante k** [1/h] bei Heizung AUS: `dT/dt = −k · (T_innen − T_außen)`
- **Heizrate h** [K/h] bei Heizung EIN: `dT/dt = h − k · (T_innen − T_außen)`

Etagen werden in der Option `thermal_floors` eingetragen, z. B.:

```yaml
thermal_floors:
  - name: ground_floor
    temperature_sensor: sensor.ground_floor_temperature
    heating_switch: switch.ground_floor_heating      # Heizung EIN = 'on'
  - name: first_floor
    temperature_sensor: sensor.first_floor_temperature
    climate: climate.first_floor                      # Heizung EIN = hvac_action 'heating'
```

Optional gilt eine Etage auch als beheizt, wenn `thermal_heating_stage` (ein
`input_select`) ihren Namen als Zustand hat. Außentemperatur: `thermal_outdoor_temp`,
optional `thermal_outdoor_temp_fallback`.

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
3. Add-on **HA ML Layer for EMHASS** installieren. Es wird auf dem Gerät aus dem
   `Dockerfile` gebaut (kein vorgefertigtes Image).
4. Im Tab **Konfiguration** die eigenen Sensoren eintragen (die vorbelegten
   Namen sind nur Beispiele), Modelle aktivieren, starten, Log prüfen.

Updates: Home Assistant zeigt ein Update an, sobald `version` in
`ha_ml_layer/config.yaml` auf `main` erhöht wurde. Jede Code-Änderung braucht
deshalb eine neue Versionsnummer, sonst kommt sie nicht an.

Wer das Add-on vorher als lokales Add-on (`/addons/ha_ml_layer/`) betrieben hat:
Das Store-Add-on ist ein eigenes Add-on mit eigenem `/data`. Das lokale Add-on
stoppen und deinstallieren, sonst schreiben beide dieselben Sensoren. Die Modelle
lernen danach neu.

### Konfiguration (Tab „Konfiguration“ des Add-ons)
Alle Sensoren werden im Formular eingetragen. Die Vorbelegungen sind Beispielnamen
(Muster wie bei EMHASS, z. B. `sensor.power_photovoltaics`) und müssen durch die
eigenen Entity-IDs ersetzt werden. Leere optionale Felder blendet Home Assistant
aus; sichtbar werden sie mit „Show unused optional configuration options“. Nach dem Speichern das Add-on neu starten. Fehlerhafte
Eingaben (z. B. ungültige Entity-ID) stehen beim Start als `Configuration:` im Log.

| Bereich | Optionen |
|---|---|
| Allgemein | `poll_interval_seconds` (60–3600, empfohlen 300), `log_level`, `pv_enabled`, `thermal_enabled` |
| Standort | optional `latitude`, `longitude`, `timezone` (leer = aus der Home-Assistant-Konfiguration) |
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

### Update von 0.1.x
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
- **Wärmemodell:** Träge Heizungen wirken stundenlang nach. Besser wären
  2–3 h Sperrzeit nach dem Umschalten (z. B. bei Fußbodenheizung) und eine Regressionsgerade über das ganze
  Fenster statt zweier Endpunkte (Sensorauflösung 0,1 K).
- **InfluxDB nur schreibend:** Gelesen wird über die HA-REST-API. Ein Backfill
  aus InfluxDB würde den Kaltstart verkürzen.
- **Nur Anzeige:** Die Modelle ändern keine EMHASS-Konfiguration selbst. Die
  Sensoren werden per `POST /api/states` gesetzt und landen daher nicht in der
  Langzeitstatistik.
