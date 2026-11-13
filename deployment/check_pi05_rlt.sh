#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/.."
export PIPER_DEPLOY_ROOT="$ROOT"
exec "$ROOT/.venv/bin/python" -u -m rlt_piper.cli check "$@"
