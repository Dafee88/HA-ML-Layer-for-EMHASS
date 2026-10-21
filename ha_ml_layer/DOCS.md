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
- `sensor.building_thermal_emhass` – `cooling_constant` and `heating_rate` in
  EMHASS units, attribute `valid` (see section 7).

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
`pv_dayahead_model_v4.pre_history.pkl`. To run it again, delete the marker file
or reinstall the add-on.

## 7. Building thermal model for EMHASS

The add-on learns the parameters of EMHASS' thermal model
(<https://emhass.readthedocs.io/en/stable/thermal_model.html>):

- `cooling_constant` (per hour per degree difference) from heating-off phases of
  at least 3 h (3 h lock-out after the heating switched off), with a straight-line
  fit through all readings – short windows are useless with 0.1 K sensors.
- `heating_rate` (K/h at nominal power) from a daily energy balance between two
  heating-off rest periods; days with noticeable sun are skipped. Needs
  `thermal_nominal_power_w` (the P_nom of the heating load in EMHASS) and,
  ideally, `heating_power_sensor` per floor.

`sensor.building_thermal_emhass` becomes `valid` after at least 3 usable cooling
phases and 3 heating balances per floor. Use the values only when valid, e.g.
inside your `def_load_config` / `thermal_config`:

```jinja
{% set t = states.sensor.building_thermal_emhass %}
{% set ok = t is not none and t.attributes.valid %}
"heating_rate": {{ t.attributes.heating_rate if ok else 2.0 }},
"cooling_constant": {{ t.attributes.cooling_constant if ok else 0.005 }},
```

(2.0 / 0.005 stand for your current fixed values.)

