#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PIPER_PYTHON:-python3.11}"
"$PYTHON" -m venv "$ROOT/deployment/.venv"
"$ROOT/deployment/.venv/bin/python" -m pip install -r "$ROOT/deployment/requirements-control.txt" \
  -e "$ROOT/deployment/vendor/pyAgxArm" \
  -e "$ROOT/deployment/vendor/openpi/packages/openpi-client"
if [[ ! -f "$ROOT/deployment/config.yaml" ]]; then
  cp "$ROOT/deployment/config.example.yaml" "$ROOT/deployment/config.yaml"
fi
mkdir -p "$ROOT/deployment/logs"
echo 'Control environment prepared. Configure hardware IDs and restore private runtime assets before running.'
