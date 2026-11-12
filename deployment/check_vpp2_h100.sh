#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VPP2_REMOTE_PORT="${VPP2_REMOTE_PORT:-21969}"
export VPP2_REMOTE_CODEC="${VPP2_REMOTE_CODEC:-webp_prepared}"
exec "$ROOT/check.sh" --model vpp2 --compile "$@"
