#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "Virtualenv python not found at $PYTHON" >&2
  exit 1
fi

"$PYTHON" -m pip install -U pip
"$PYTHON" -m pip install -r requirements.txt

if [ ! -f "config.yaml" ]; then
  cp config.example.yaml config.yaml
fi

mkdir -p logs data data/sandbox

"$PYTHON" main.py
