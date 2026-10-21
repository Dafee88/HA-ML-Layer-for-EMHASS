import logging

import requests

log = logging.getLogger("ha_client")


class HAClient:
    """Thin wrapper around the HA REST API (reachable through the Supervisor
    proxy when the add-on sets hassio_api/homeassistant_api: true)."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def get_state(self, entity_id: str, quiet: bool = False) -> dict | None:
        r = requests.get(f"{self.base_url}/api/states/{entity_id}", headers=self.headers, timeout=10)
        if r.status_code == 404:
            if not quiet:
                log.warning("Entity not found: %s", entity_id)
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


class Publisher:
    """Wraps HAClient: remembers the last state written per entity and writes
    them all again when Home Assistant lost them. States set through the REST
    API are not restored after an HA restart; ensure() is called every minute
    and checks one watched entity."""

    def __init__(self, ha: HAClient, watch_entity: str):
        self.ha = ha
        self.watch_entity = watch_entity
        self.cache = {}
        self.republished = 0

    def __getattr__(self, name):
        return getattr(self.ha, name)

    def set_state(self, entity_id, state, attributes=None):
        result = self.ha.set_state(entity_id, state, attributes)
        self.cache[entity_id] = (state, attributes)
        return result

    def ensure(self):
        if self.watch_entity not in self.cache:
            return False
        try:
            present = self.ha.get_state(self.watch_entity, quiet=True) is not None
        except requests.RequestException:
            return False            # HA (still) down - try again next minute
        if present:
            return False
        n = 0
        for entity_id, (state, attributes) in list(self.cache.items()):
            try:
                self.ha.set_state(entity_id, state, attributes)
                n += 1
            except requests.RequestException:
                log.warning("Republish of %s failed, retrying next minute", entity_id)
                return False
        self.republished += 1
        log.info("%s was missing (Home Assistant restart?) - republished %d sensors", self.watch_entity, n)
        return True

