# Optional realtime-vla backend for Piper π0.5

The normal JAX entrypoints remain the default. This option changes model
computation to the vendored [realtime-vla](https://github.com/dexmal/realtime-vla)
Triton implementation. It does not change the 2500/2499 checkpoint identity,
30Hz action time axis, 120Hz command loop, RTC, 12-frame/50ms smoothing settings,
firmware command limits, camera timestamps or independent damping stop.

On the controller host, choose **one** command:

```bash
# π0.5 + existing RTC and smoothing
./run.sh --model pi05 --backend realtime-vla
```

```bash
# π0.5 + existing RTC/smoothing + full VLA-Corrector
./run_pi05_corrector.sh --mode full --backend realtime-vla
```

Both perform automatic preparation and then move the robot. Space/Esc/Ctrl+C
request the existing firmware damping stop. Without `--backend`, the old JAX
behavior remains selected. Explicit `--backend jax` also selects that behavior.
The backend option applies to π0.5; it does not select a backend for VPP2 or RLT.
RLT's token feature and learner path is separate and is not changed here.

Saved-observation checks do not open CAN or cameras:

```bash
./check.sh --model pi05 --backend realtime-vla
./check_pi05_corrector.sh --mode full --backend realtime-vla
```

## Dependencies and derived assets

The optional `.realtime-vla-venv` uses Python3.11, Torch2.8.0+cu128,
Triton3.4.0 and torchvision0.23.0. CUDA128/sm120 support matters on the5090;
the old JAX environment's Torch2.7.1+cu126 wheel does not provide that support.
The original JAX and control environments are not upgraded. The separate
environment reuses CPU dependencies from `vendor/openpi/.venv` through an
explicit `.pth` file; the model source and client package are supplied on
`PYTHONPATH`. Torch/CUDA packages live in the optional environment itself.

Example setup for a new controller, using a separate environment:

```bash
vendor/openpi/.venv/bin/python -m venv .realtime-vla-venv
.realtime-vla-venv/bin/pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.realtime-vla-venv/bin/python - <<'PY'
from pathlib import Path
root=Path.cwd()
site=root/'.realtime-vla-venv/lib/python3.11/site-packages'
(site/'piper_openpi_cpu_dependencies.pth').write_text(
    str(root/'vendor/openpi/.venv/lib/python3.11/site-packages')+'\n')
PY
```

Export and validation run on saved data only; use these after installing the
code and optional environment. Each command must finish successfully before
the next one:

```bash
export PYTHONPATH="$PWD:$PWD/vendor/openpi/src:$PWD/vendor/openpi/packages/openpi-client/src"
export OPENPI_DATA_HOME="$PWD/cache/openpi"
JAX_PLATFORMS=cpu vendor/openpi/.venv/bin/python -m realtime_vla_runtime.export --root "$PWD" --step 2500
XLA_PYTHON_CLIENT_PREALLOCATE=false vendor/openpi/.venv/bin/python -m realtime_vla_runtime.validate reference --root "$PWD" --step 2500
JAX_PLATFORMS=cpu .realtime-vla-venv/bin/python -m realtime_vla_runtime.validate triton --root "$PWD" --step 2500
```

Derived weights are kept in `realtime_vla_assets/pi05_2500/`. The exporter refuses
to overwrite an existing derived model. The source checkpoint and norm stats
remain unchanged. The same preparation supports1500 by changing `--step`, then
selecting `--pi05-step 1500` at the ordinary entrypoint; Corrector stays2500.
Missing/failed offline validation is rejected before an existing model is
stopped and before robot preparation. Metadata verifies the selected backend;
an acceleration failure does not silently fall back to JAX.

Plain and Corrector services use loopback ports8004 and8005. Model selection
releases only owned, idle services while the caller holds the common follower
lock; connected or unknown processes remain untouched. State and log files
start with `logs/realtime_vla_`.

## Numerical and performance scope

The2026-11-16 saved-observation comparison of2500 on the5090 D v2 covered two
observations with three fixed noises each. The adapted Triton path measured
38.90ms median /40.28ms P95, versus78.71ms median for the JAX reference. This is
local preprocessing + inference + postprocessing, not full robot-loop latency,
and is a small sample. It does not establish sustained30FPS inference.

At the same input/noise, joint-action difference was0.2515° P95 /0.7805° max;
gripper difference was0.3185mm P95 /0.5401mm max. Vision-prefix relative L2
difference was0.58–0.62%. The Triton backend is numerically approximate, not
bit-identical to JAX. Offline numerical acceptance checks use1° joint P95,
3° joint max,2mm gripper P95,5mm gripper max and3% latent relative L2. These are
validation criteria, not additional robot command limits or grasp acceptance.

Corrector keeps the original trained MLP and monitoring protocol. Its OGG
gradient runs through the frozen small MLP only, at the same two flow steps;
it does not backpropagate through Triton kernels. Zero guidance must reproduce
this backend's own baseline exactly. Backend approximation can affect monitor
scores, so real grasp success, monitor false-trigger behavior and physical
recovery quality still require hardware evaluation.

The full Corrector service passed a60-second software replay with real GPU
policy/monitor RPC and a fake120Hz arm:1144 accepted chunks,599 monitor calls,
7121 software joint commands,9.14ms maximum control gap,50.94ms policy median
and58.14ms P95, with no worker errors or queue exhaustion. The replay repeated
saved images with synthetic capture anchors; it triggered no visual events and
does not validate physical recovery. Separate startup checks confirmed that
zero OGG is exact and nonzero OGG changes actions (maximum joint difference
0.002264rad, gripper difference0.000171m on the fixed saved observation).

No training data, observations, generated weights or environments are committed.
The upstream license, pinned commit and adaptations are recorded under
`deployment/vendor/realtime_vla/`.
