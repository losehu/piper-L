#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="${1:-$ROOT/corrector_assets/pi05_2500_k5}"
mkdir -p "$ROOT/logs" "$ROOT/cache/corrector_jax"
exec 9>"$ROOT/logs/corrector_policy.lock"
flock -n 9 || exit 1
export PYTHONPATH="$ROOT:$ROOT/..:$ROOT/vendor/openpi/src:$ROOT/vendor/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME="$ROOT/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="$ROOT/cache/corrector_jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
exec "$ROOT/vendor/openpi/.venv/bin/python" -u -m corrector.serve --root "$ROOT" --assets "$ASSETS" --port 8002
