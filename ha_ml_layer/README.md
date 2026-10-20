# HA ML Layer for EMHASS

Online-learning layer (River) that corrects your Solcast PV forecast for
[EMHASS](https://emhass.readthedocs.io), checks it in real time against a local
weather station, and fills in for the forecast APIs when they are offline.

## Install these first

**Required**

| What | Why | Link |
|---|---|---|
| Home Assistant OS or Supervised | add-ons need the Supervisor | [Installation](https://www.home-assistant.io/installation/) |
| HACS | to install the Solcast integration | [hacs.xyz](https://hacs.xyz/) |
| Solcast PV Forecast integration + free Solcast hobbyist account | PV forecast with the 30-min curve (`detailedForecast`) | [ha-solcast-solar](https://github.com/BJReplay/ha-solcast-solar), account: [solcast.com](https://solcast.com/) |
| A sensor with your current PV power in W | the models learn from it | your inverter's integration |
| EMHASS add-on | uses the corrected curve | [emhass-add-on](https://github.com/davidusb-geek/emhass-add-on) |

**Optional**

| What | Enables | Link |
|---|---|---|
| Weather station with a solar radiation sensor (W/m²), e.g. Ecowitt | real-time check, stand-in while the inverter curtails, offline fallback | [Ecowitt (core)](https://www.home-assistant.io/integrations/ecowitt/) or [Ecowitt Local (HACS)](https://github.com/alexlenk/ecowitt_local) |
| Forecast.Solar | second forecast as extra model input | [Forecast.Solar](https://www.home-assistant.io/integrations/forecast_solar/) |
| InfluxDB 1.x add-on | stores forecasts and results for backtests | [InfluxDB add-on](https://github.com/hassio-addons/addon-influxdb) |
| Reference irradiance station nearby | checks your sensor's calibration | e.g. [GeoSphere Austria](https://www.home-assistant.io/integrations/zamg/) |
| Template sensor for curtailment | correct learning on days with a full battery / export limit | [Template](https://www.home-assistant.io/integrations/template/) |

Setup, options and the EMHASS `rest_command`: see the **Documentation** tab.
