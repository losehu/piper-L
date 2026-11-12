#!/usr/bin/env bash
# Run on every GPU-bearing role, sharing data and output_dir.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == help ]]; then
  echo 'Usage: bash scripts/robodojo/train_video.sh [--dry-run] [key=value ...]'
  echo 'Stage-2 Video -> RoboDojo history-conditioned Video-10k; global batch 832.'
  exit 0
fi
export NNODES="${NNODES:-${WORLD_SIZE:-4}}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -z "${NODE_RANK:-}" && -n "${RANK:-}" ]]; then export NODE_RANK="$RANK"; fi
export VPP2_TRAIN_MODULE=vpp2.cli
export VPP2_ACCELERATE_CONFIG=configs/robodojo/accelerate.yaml
export REQUIRE_RDMA="${REQUIRE_RDMA:-1}"
exec bash scripts/train.sh --config configs/robodojo/train_video_10k.yaml "$@"
