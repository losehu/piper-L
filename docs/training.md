# 八卡训练与权重导出

## 数据与超参数

本轮使用 54 段筛选后的示范，按整段划分为 48 段训练、6 段验证。验证原编号是 19、30、42、59、72、83；第 76 段及它拆出的第二段 87 都在训练集。划分与裁剪元数据见 `training/dataset_manifest.json`。

数据不包含在仓库中，需要自行放置。固定留出集验证使用同一采集会话的数据，不代表跨日期、跨场景泛化能力。

| 配置 | 本轮设置 |
| --- | --- |
| 基础模型 | 官方 `pi05_base`，全量微调 |
| GPU | 8×H100；FSDP 2 × 数据并行 4 |
| Global batch | 64 |
| 更新次数 | 2500；最终目录为 `2499` |
| 学习率 | warmup 50，峰值 1e-5，余弦降到 1e-6 |
| EMA | 0.999 |
| 动作块 | 50 帧，30Hz，物理维度 7 |
| 图像 | 外部/腕部两路 RGB，224×224 letterbox |
| 验证 | 固定 48 帧、训练集归一化、固定 flow 噪声种子、EMA |

训练时六轴动作减去该观测时刻的实测关节状态，夹爪保持绝对开度；推理后反变换为绝对目标。没有 ALOHA 双臂或夹爪单位转换。

## 1. 环境和目录

从仓库根目录执行：

```bash
bash tools/setup_model.sh training
uv pip install --python training/vendor/openpi/.venv/bin/python \
  -r training/requirements-data.txt

export PIPER_RUN_ROOT=/path/to/private_training_run
export PIPER_SELECTED_DIR=/path/to/piper_screened
export HF_LEROBOT_HOME="$PIPER_RUN_ROOT/lerobot"
export PIPER_BASE_PARAMS=/path/to/pi05_base/params
```

基础模型需自行按 OpenPI 的模型使用条款准备，不随源码下载。运行目录与数据目录应放在仓库外，或使用被忽略的 `artifacts/`。

## 2. 数据转换

```bash
training/vendor/openpi/.venv/bin/python training/convert_on_arm.py
```

脚本读取筛选后 HDF5 和 manifest，生成 `local/piper_grasp_train` / `local/piper_grasp_val` 两套 LeRobot v2.1 数据。它保留真实实测状态和发送动作的弧度/米单位，使用 RGB、相同 letterbox，完整解码检查生成视频的帧数。

如使用新数据，先提供自己的 manifest，并设置 `PIPER_SPLIT_FILE=/path/to/manifest.json`，不要复用旧示范的裁剪区间或哈希。

## 3. 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash training/train_pipeline.sh
```

流水线依次执行数据完整性、仅训练集归一化、数据/动作变换检查、10 步训练 smoke，以及 2500 次更新的正式训练。`piper_data_loader.py` 为本数据选择 PyAV 视频后端。

恢复已存在的正式运行：

```bash
bash training/train_pipeline.sh --resume
```

产物位于 `$PIPER_RUN_ROOT` 的 `checkpoints/`、`assets/`、`logs/`，全部不上传 Git。固定留出集损失不能当作真机成功率。本轮曾以 1500 作为离线损失候选，当前部署由用户选择最终 `2499`。

## 4. BF16 推理导出

```bash
export PYTHONPATH="$PWD/training/vendor/openpi/src"
training/vendor/openpi/.venv/bin/python tools/export_bf16.py \
  --source "$PIPER_RUN_ROOT/checkpoints/pi05_piper_grasp/screened54_train48_v1/2499" \
  --output /path/to/private_export_2500
```

输出只包含推理参数和归一化资产；不包含优化器状态。加入 `--base /path/to/existing_bf16_checkpoint` 可以同时生成精确 XOR 增量包，减少跨机器传输量。

恢复增量包有两种入口：

```bash
# 常规：校验包和目标参数
training/vendor/openpi/.venv/bin/python tools/apply_bf16_delta.py \
  --base /path/to/matching_bf16_base \
  --patch /path/to/private_delta \
  --output /path/to/private_restored_checkpoint

# 低内存路径：原位 XOR，不做额外哈希或逐叶相等复查
training/vendor/openpi/.venv/bin/python tools/apply_bf16_delta_fast.py \
  --base /path/to/matching_bf16_base \
  --patch /path/to/private_delta \
  --output /path/to/private_restored_checkpoint
```

两者都只处理 CPU 文件，不启动机械臂；快速路径需要使用与增量包对应的 BF16 基础权重。最终部署目录按 [部署说明](deployment.md) 放置。
