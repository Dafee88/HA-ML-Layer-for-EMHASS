import json
import os

OPTIONS_FILE = os.environ.get("OPTIONS_FILE", "/data/options.json")


def load_options() -> dict:
    with open(OPTIONS_FILE, "r") as f:
        return json.load(f)


class Config:
    def __init__(self):
        opts = load_options()
        self.ha_url = opts.get("ha_url", "http://supervisor/core")
        self.poll_interval = int(opts.get("poll_interval_seconds", 300))
        self.log_level = opts.get("log_level", "info")
        self.models = {m["name"]: m["enabled"] for m in opts.get("models", [])}
        # Supervisor injects this automatically when hassio_api/homeassistant_api: true
        self.token = os.environ.get("SUPERVISOR_TOKEN", "")

