#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

[ -f "config.yaml" ] || cp config.example.yaml config.yaml
mkdir -p logs data data/sandbox

echo "Starting AgentContainer (API + sandbox) via docker compose..."
docker compose up -d --build
