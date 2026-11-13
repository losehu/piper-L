#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/.."
export PIPER_DEPLOY_ROOT="$ROOT"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
exec /usr/bin/gnome-session-inhibit --inhibit suspend:idle --reason 'Piper RLT console' \
  /usr/bin/python3 -u -m rlt_piper.ui "$@"
