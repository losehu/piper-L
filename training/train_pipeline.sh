#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_ROOT="${PIPER_RUN_ROOT:-$REPO_ROOT/artifacts/training}"
OPENPI_DIR="$SCRIPT_DIR/vendor/openpi"
PYTHON="${PIPER_TRAIN_PYTHON:-$OPENPI_DIR/.venv/bin/python}"
cd "$OPENPI_DIR"
export PIPER_RUN_ROOT="$TASK_ROOT"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$TASK_ROOT/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$TASK_ROOT/cache/openpi}"
export PYTHONPATH="$SCRIPT_DIR:$OPENPI_DIR:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/jax_cache"
mkdir -p "$TASK_ROOT/jax_cache" "$TASK_ROOT/logs"
exec 9>"$TASK_ROOT/pipeline.lock"
flock -n 9 || exit 1
trap 'printf "failed: exit=%s\n" "$?" > "$TASK_ROOT/pipeline_status.txt"' ERR
printf 'verifying data\n' > "$TASK_ROOT/pipeline_status.txt"
(cd "$HF_LEROBOT_HOME" && sha256sum -c SHA256SUMS > "$TASK_ROOT/logs/data_integrity.log")
printf 'computing train-only normalization\n' > "$TASK_ROOT/pipeline_status.txt"
if [[ " $* " != *" --resume "* ]]; then
"$PYTHON" -u "$SCRIPT_DIR/run_piper.py" stats > "$TASK_ROOT/logs/norm_stats.log" 2>&1
printf 'checking data and transforms\n' > "$TASK_ROOT/pipeline_status.txt"
"$PYTHON" -u "$SCRIPT_DIR/run_piper.py" check > "$TASK_ROOT/logs/pipeline_check.log" 2>&1
printf 'smoke test: 10 training steps\n' > "$TASK_ROOT/pipeline_status.txt"
"$PYTHON" -u "$SCRIPT_DIR/run_piper.py" smoke > "$TASK_ROOT/logs/smoke.log" 2>&1
fi
printf 'training: 2500 steps, 8 H100, batch64, 48 episodes; 6 held out\n' > "$TASK_ROOT/pipeline_status.txt"
"$PYTHON" -u "$SCRIPT_DIR/run_piper.py" train "$@" > "$TASK_ROOT/logs/train.log" 2>&1
printf 'training complete; physical robot evaluation remains\n' > "$TASK_ROOT/pipeline_status.txt"
