#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-deployment}" in
  deployment|training) SIDE="${1:-deployment}" ;;
  *) echo 'Usage: setup_model.sh [deployment|training]' >&2; exit 2 ;;
esac
cd "$ROOT/$SIDE/vendor/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
echo "$SIDE OpenPI environment prepared. No dataset, base model or fine-tuned weights were downloaded."
