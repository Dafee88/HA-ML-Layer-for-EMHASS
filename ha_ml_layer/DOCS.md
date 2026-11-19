# HA ML Layer for EMHASS – Documentation

## 1. Prerequisites

Install and set up these before starting the add-on.

### Required
1. **Home Assistant OS or Supervised** – add-ons need the Supervisor.
   <https://www.home-assistant.io/installation/>
2. **HACS** – needed for the Solcast integration. <https://hacs.xyz/>
3. **Solcast PV Forecast** (HACS) with a free Solcast hobbyist account.
   Integration: <https://github.com/BJReplay/ha-solcast-solar> – account: <https://solcast.com/>.
   The add-on reads the sensors *Forecast today* and *Forecast tomorrow*
   (attribute `detailedForecast`, 30-min values).
4. **Current PV power in W** – any sensor, usually from your inverter's integration.
5. **EMHASS add-on** – <https://github.com/davidusb-geek/emhass-add-on>,
   documentation <https://emhass.readthedocs.io>. It needs a `rest_command`
   that passes the corrected curve (see section 4).

### Optional
- **Weather station with solar radiation (W/m²)**, for example Ecowitt:
  [Ecowitt (core, local push)](https://www.home-assistant.io/integrations/ecowitt/) or
  [Ecowitt Local (HACS, local polling)](https://github.com/alexlenk/ecowitt_local).
  Mount it free of shade. Enables the real-time nowcast, the stand-in for
  curtailed power and the offline fallback.
- **Forecast.Solar** – second forecast: <https://www.home-assistant.io/integrations/forecast_solar/>.
  Several roof planes are combined by the integration into one set of sensors.
- **InfluxDB 1.x** – add-on <https://github.com/hassio-addons/addon-influxdb>.
  The ML layer writes to it directly. The InfluxDB *integration*
  (<https://www.home-assistant.io/integrations/influxdb/>) is only needed if
  Home Assistant itself should also store its history there.
- **Air pressure and humidity of the weather station** (optional) – inputs of
  the local weather outlook (section 3), works without internet.
- **Reference station** for the calibration check of your own sensor, e.g.
  GeoSphere Austria: <https://www.home-assistant.io/integrations/zamg/>.
- **Curtailment sensor** – see section 5.

## 2. Configuration

Open the **Configuration** tab. The pre-filled entity ids are examples only –
replace them with your own. Empty optional fields are hidden; turn on
*Show unused optional configuration options* to see them. Save and restart
the add-on. Invalid entries are logged at start as `Configuration: ...`.

Minimum for the PV model: `pv_solcast_today`, `pv_solcast_tomorrow`,
`pv_actual_power`. Location and time zone are taken from Home Assistant
unless you set them.

## 3. Outputs

- `sensor.pv_ml_corrected_forecast` – current corrected value (W) with
  diagnostics as attributes (`forecast_source`, `dayahead_active`,
  `mae_dayahead_w` vs `mae_solcast_w`, `station_*`, ...).
- `sensor.pv_ml_corrected_forecast_curve` – attribute `forecasts`: 48 × 30 min
  `{"date", "p_pv_forecast"}`. This is what EMHASS should use.

Until a model has learned enough and beats its baseline, raw Solcast is passed
through unchanged.
- `sensor.pv_ml_forecast_tomorrow_kwh` and `sensor.pv_ml_forecast_today_remaining_kwh`
  – energy in kWh (for EMHASS `soc_final` or a battery reserve). Attribute
  `source`: `ml` (model active), `solcast` (raw Solcast), `station` (no Solcast
  data – weather-station estimate from typical cloudiness only, use with care),
  `none` (state `unknown`). `solcast_kwh` for comparison.
- `sensor.building_thermal_emhass` – `cooling_constant`, `heating_rate` and
  `thermal_inertia` in EMHASS units, attribute `valid` (see section 7).
- `sensor.pv_ml_forecast_accuracy` ("PV Forecast Error (ML vs actual)") – daily
  check: the forecast for a day as known at the snapshot hour of the evening
  before (ML and raw Solcast, kWh) against the measured PV energy. State =
  energy-weighted ML error of the last 7 scored days in %
  (sum of |forecast - actual| / sum of actual). Attributes: `error_solcast_pct`
  (same for raw Solcast), mean absolute errors in kWh, `bias_ml_kwh` /
  `bias_solcast_kwh` (mean signed error, positive = forecast too high), the
  values of the last scored day (`yesterday_*`) and the daily `history`.
  Days with less than 90 % readings are not scored.

- `sensor.pv_local_forecast_curve` ("PV Forecast Curve (local station)", needs
  `pv_irradiance`) – the same 48 × 30 min curve from your weather station
  only, computed every cycle in parallel to the ML curve, plus
  `today_remaining_kwh`, `tomorrow_kwh`, `calibration` and `error_local_pct`.
  Its evening forecast for tomorrow is scored like ML and Solcast
  (`error_local_pct`, `bias_local_kwh` on `sensor.pv_ml_forecast_accuracy`).
  The ratio actual / raw forecast of the last 14 scored days (from 5 days,
  limited to 0.5-1.5) calibrates it (`local_calibration`). The calibrated
  station forecast is also what fills every slot without Solcast data in the
  ML curve - the fallback EMHASS gets without internet.
- `sensor.pv_ml_local_weather` ("Local Weather Outlook (station)", needs
  `pv_irradiance`) – tomorrow's cloudiness from your own station only, so it
  also works without internet (power cut, API down). State `sunny` / `partly
  cloudy` / `cloudy` (daily clear-sky index >= 0.7 / >= 0.4 / below) or
  `unknown`. Every evening at the snapshot hour it predicts the next day's
  daily clear-sky index (measured / clear-sky irradiance, sun >= 10°) with a
  ridge regression on today's index and, if configured, the pressure change
  over 3 h and 12 h, the pressure against its 7-day mean (`weather_pressure`)
  and the humidity (`weather_humidity`). The next evening the prediction is
  scored against the measured day, together with climatology (mean of the
  last 30 days) and persistence (today = tomorrow). Only a method that beats
  climatology over at least 10 scored days is used (`forecast_method`,
  `fallback_active`); then the station fallback for that day scales its
  typical cloudiness by predicted / climatology (`fallback_factor`, also on
  `sensor.pv_ml_forecast_*_kwh` as `local_weather_factor`). While Solcast has
  data nothing changes. Also `pressure_hpa`, `pressure_tendency_3h_hpa` and
  `pressure_tendency` (Met Office terms: steady, rising/falling slowly,
  rising/falling, quickly, very rapidly), the MAE of each method and the
  number of days. With InfluxDB the days are rebuilt from history at the first
  start (and when the inputs change). State file `/data/local_weather_v1.json`.

  **Pre-training with a reference station (optional,
  `weather_reference_history`):** your own station has weeks of history, a
  nearby national weather service station has years. Download its hourly data
  once as CSV (GeoSphere Austria data hub, dataset `klima-v2-1h`, parameters
  `cglo,p,rf,tl`; time stamp = end of the hour), copy the files to e.g.
  `/config/ha_ml_layer/` and set the option to a path or glob such as
  `/config/ha_ml_layer/station_*.csv`. At start the same kind of model is
  fitted on these years (features: today's index, pressure change and anomaly,
  humidity, evening cooling Tmax - T now, daily temperature range, season; the
  newest 120 days are held out). Your station feeds it through a linear
  mapping per feature learned on the days both stations have data (at least
  20). It is scored like the other methods (against the reference station's
  measured day if `pv_reference_irradiance` is set, otherwise against your own)
  and only used if it beats climatology (`forecast_method: reference prior`).
  Attributes `csi_reference_forecast`, `mae_csi_reference_prior`,
  `mae_csi_reference_climatology`, `reference_evaluated_days`,
  `reference_mapping`, `reference_prior`. Cache `/data/reference_prior_v1.json`
  (rebuilt when the files change). Temperature inputs come from
  `thermal_outdoor_temp`. Nothing is downloaded.

- `sensor.pv_ml_irradiance_corrected` ("Irradiance (station, corrected)", needs
  `pv_irradiance` and `pv_reference_irradiance`) – your station's irradiance
  corrected against the reference station. Many hobby stations (e.g. Ecowitt)
  measure lux and divide by a fixed factor, which fits sun and clouds
  differently; a gain in the station's settings only moves the level. The
  add-on learns the ratio reference / own per hour in a table by clearness
  (own clear-sky index) and sun elevation (median of the last 60 hours per
  cell, else the overall gain). Each hour is scored before it is learned;
  the state uses the method with the lowest error (`method`: raw, gain or
  table; `mae_raw_w_m2`, `mae_gain_w_m2`, `mae_table_w_m2`, `learned_hours`,
  `scored_hours`, `gain`, `table`, `raw_w_m2`, `reference_w_m2`). Without
  internet the learned table keeps working. With InfluxDB it learns the last
  60 days at the first start. The forecast models keep using the raw value
  (they have learned its scale). State file `/data/irradiance_correction_v1.json`.

**Low output (snow, shading, faults):** with a weather station, PV power below
50 % of the station's potential (potential at least 300 W) while not curtailed
is not learned by the forecast models (attribute `low_output`). If such an
episode starts at or below 4 °C outdoors (`thermal_outdoor_temp`, unknown counts
as cold) it may be snow on the panels and the station model does not learn
either; warm episodes are learned by the station's plant model (shading). After
60 min a warning is logged.

After a Home Assistant restart all sensors are written again within one minute.

## 4. EMHASS: pass the curve

Example for `configuration.yaml` (EMHASS add-on on the same host, 30-min time
step, 24 h horizon = 48 values). If the curve is older than one hour, the key
is left out and EMHASS uses its own forecast method:

```yaml
rest_command:
  emhass_dayahead_optim:
    url: http://localhost:5000/action/dayahead-optim
    method: post
    content_type: application/json
    payload: >-
      {% set s = states.sensor.pv_ml_corrected_forecast_curve %}
      {% set c = state_attr('sensor.pv_ml_corrected_forecast_curve', 'forecasts') %}
      {% if s and c and (now() - s.last_updated).total_seconds() < 3600 %}
      {"pv_power_forecast": {{ c | map(attribute='p_pv_forecast') | list | tojson }}}
      {% else %}{}{% endif %}
```

Combine it with your other runtime parameters (prices, loads) as described in
the EMHASS documentation on passing data.

## 5. Curtailment sensor (recommended with a battery or export limit)

When the battery is full or exports are limited, the inverter throttles PV.
Without a hint the models learn this as "weak sun". Point `pv_curtailed_entity`
to a sensor that is `on` while curtailment is possible, for example (adapt the
entity ids and thresholds):

```yaml
template:
  - binary_sensor:
      - name: "PV curtailed"
        state: >
          {{ states('sensor.battery_state_of_charge') | float(0) >= 97
             and states('sensor.power_photovoltaics') | float(0) > 50 }}
```

With a weather station, the models then learn from the station's potential
power instead.

## 6. InfluxDB

Enable `influxdb_enabled` and enter host (InfluxDB add-on: `a0d7b954-influxdb`),
database, user and password. Measurements: `ml_layer_pv` (every cycle),
`ml_layer_pv_slot` (per 30-min slot), `ml_layer_dayahead` (evening snapshot of
tomorrow's forecast). HTTP 401 in the log means the login is wrong.

At start the add-on logs, read-only, which history of your configured sensors
exists in InfluxDB (`InfluxDB history: ...` lines: measurement, first/last
value, fields, and whether Solcast's 30-min curve is stored).

**Learning from history** (`influxdb_learn_from_history`, on by default): if
Home Assistant's InfluxDB integration stores your Solcast sensor (including the
`detailedForecast` attribute) and your PV power, the add-on rebuilds the
day-ahead model once at start from up to 90 days of history plus its own slot
records (`ml_layer_pv_slot`), in time order. Curtailed slots are skipped: where
the curtailment indicator was on, and - with a weather station - where the
actual power was below 60 % of the station's potential power. The model is
still only used once it beats raw Solcast. Result in the log (`History: ...`)
and in `/data/dayahead_history_v1.json`; the previous model is kept as
`pv_dayahead_model_v4.pre_history.pkl`.

Only history from the current plant/forecast setup is learned: a change of the
Solcast rooftop sites (a site added or removed) is detected automatically, and
`influxdb_history_start` (YYYY-MM-DD) lets you set the date of any other change
(e.g. new panels, a changed site geometry or capacity at solcast.com). To run it
again, delete `/data/dayahead_history_v2.json` or reinstall the add-on.

## 7. Building thermal model for EMHASS

The add-on fits the parameters of EMHASS' thermal model
(<https://emhass.readthedocs.io/en/stable/thermal_model.html>) the way EMHASS
uses them: on the error of 24 h simulations. Each simulation starts at a
measured room temperature and then only sees the measured outdoor temperature,
sun and heating power. Fitted (mean of the floors = `sensor.building_thermal_emhass`,
each floor = `sensor.building_thermal_<floor>`):

- `cooling_constant` (per hour per degree difference)
- `heating_rate` (K/h at `thermal_nominal_power_w`, the P_nom of the heating load in EMHASS)
- `thermal_inertia` (h) – delay of the heating (underfloor heating: hours),
  EMHASS option of the same name
- `internal_gain` (K/h) and `solar_gain` (K/h per 100 W/m² of `pv_irradiance`):
  people, devices, neighbouring flats, sun. EMHASS' equation has no such terms;
  `outdoor_temp_offset` = `internal_gain / cooling_constant` is the equivalent
  shift of the outdoor temperature (add it to `outdoor_temperature_forecast`
  for the exact model). Without these terms the gains end up in the heating
  rate, which is then several times too high.

Data: 30-min slots, rebuilt at start from Home Assistant's InfluxDB history
(last 35 days, needs `influxdb_enabled` and a `heating_power_sensor`), then
live. InfluxDB stores no "unavailable", so a value is carried forward for at
most 12 h for a temperature sensor that has a fallback (other sensors: until their next change). Periods in which Home Assistant's recorder has an input sensor `unavailable`/`unknown` are removed (as far back as the recorder keeps history, default 10 days) - otherwise a heating whose data link was down would look like "last value, e.g. 0 W" for that time. Slots in which the live poll had no usable data are remembered and not filled from the history either. Optional per floor: `temperature_sensor_fallback` (e.g. the
thermostat's own temperature sensor) - used while `temperature_sensor` is
unavailable, corrected by the learned difference between both sensors
(attribute `fallback_offset` of `sensor.building_thermal_<floor>`). The fit runs at start and every 24 h (a few seconds).

`valid` only if, on the last 2 days (not used for fitting), the 24 h error
(`rmse_24h`) is not worse than the trivial forecast "the temperature stays as it
is" (`rmse_24h_persistence`) + 0.1 K. After a data gap the most recent complete
24 h windows before it are used as test days, as long as they ended at most 4 days
ago - one sensor outage does not take the model away for two days. Gaps of up to 2 h (Home Assistant restart, short sensor drop-out) are bridged by linear interpolation for the fit; longer outages stay excluded (attribute `gap_stats` of `sensor.building_thermal_emhass`). Slots that
could not be built because an input was unavailable are remembered (InfluxDB
alone would fill them again once the recorder has purged that period). Before that: at least 4 training days and
2 kWh of heating energy (attribute `info` says what is missing). Use the values
only when valid, e.g. inside your `def_load_config` / `thermal_config`:

```jinja
{% set t = states.sensor.building_thermal_emhass %}
{% set ok = t is not none and t.attributes.valid %}
"heating_rate": {{ t.attributes.heating_rate if ok else 2.0 }},
"cooling_constant": {{ t.attributes.cooling_constant if ok else 0.005 }},
"thermal_inertia": {{ t.attributes.thermal_inertia if ok else 0.0 }},
```

(2.0 / 0.005 stand for your current fixed values. `thermal_inertia` needs an
EMHASS version that has it, and `prior_heat` under MPC – see the EMHASS docs.)

