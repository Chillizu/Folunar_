#!/bin/bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-/app/config.yaml}"
PORT="${PORT:-8000}"

# If no config is provided, copy the example for a sane default
if [ ! -f "$CONFIG_FILE" ]; then
  cp /app/config.example.yaml "$CONFIG_FILE"
fi

exec uvicorn main:create_app --factory --host 0.0.0.0 --port "$PORT"
