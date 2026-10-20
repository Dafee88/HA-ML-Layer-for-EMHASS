import logging

import requests

log = logging.getLogger("ha_client")


class HAClient:
    """Dünner Wrapper um die HA REST API (über den Supervisor-Proxy erreichbar,
    wenn das Add-on hassio_api/homeassistant_api: true gesetzt hat)."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_state(self, entity_id: str) -> dict | None:
        r = requests.get(f"{self.base_url}/api/states/{entity_id}", headers=self.headers, timeout=10)
        if r.status_code == 404:
            log.warning("Entity nicht gefunden: %s", entity_id)
            return None
        r.raise_for_status()
        return r.json()

    def get_config(self) -> dict:
        """Core configuration (latitude, longitude, time_zone, ...)."""
        r = requests.get(f"{self.base_url}/api/config", headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_states(self, entity_ids: list[str]) -> dict[str, dict]:
        result = {}
        for eid in entity_ids:
            state = self.get_state(eid)
            if state is not None:
                result[eid] = state
        return result

    def set_state(self, entity_id: str, state, attributes: dict | None = None):
        """Schreibt einen externen Sensor. WICHTIG: Das ist state-setzend
        (kein input_number-Write), landet also NICHT im Recorder als
        'echte' Integration, sondern als externer Zustand -- völlig ausreichend
        für Anzeige/Weiterverarbeitung durch EMHASS/Templates, aber falls
        Langzeitstatistik gewünscht ist, sollte stattdessen ein
        input_number.set_value auf einen dafür angelegten Helper verwendet werden."""
        payload = {"state": state, "attributes": attributes or {}}
        r = requests.post(
            f"{self.base_url}/api/states/{entity_id}",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

