#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/logs" "$ROOT/cache/jax"
exec 9>"$ROOT/logs/policy.lock"
flock -n 9 || { echo 'This policy service is already running.'; exit 1; }
export PYTHONPATH="$ROOT:$ROOT/vendor/openpi/src:$ROOT/vendor/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME="$ROOT/cache/openpi"
export JAX_COMPILATION_CACHE_DIR="$ROOT/cache/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
cd "$ROOT"
exec /usr/bin/gnome-session-inhibit --inhibit suspend:idle --reason 'RTC Piper model active' \
  "$ROOT/vendor/openpi/.venv/bin/python" -u -m rtc_piper.serve_policy --root "$ROOT" --port 8000
