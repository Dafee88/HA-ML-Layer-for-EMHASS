#!/usr/bin/env bash
set -e

echo "[ha-ml-layer] Starting..."

# Supervisor mounts add-on options here; SUPERVISOR_TOKEN is injected
# automatically when hassio_api/homeassistant_api: true is set.
export OPTIONS_FILE="/data/options.json"

exec python3 -u /app/main.py
