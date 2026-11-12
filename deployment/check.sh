#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
exec "$ROOT/.venv/bin/python" -u -m rtc_piper.cli check "$@"
