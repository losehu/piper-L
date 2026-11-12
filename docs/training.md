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

## VLA-Corrector 八卡训练

复用既有π0.5/OpenPI环境、最终2500更新BF16检查点和转换好的LeRobot数据。不重新训练π0.5，也不将6段留出数据加入训练。使用与部署相同的冻结视觉编码器提取42,314帧：48段/37,581帧训练，6段/4,733帧留出；两路有效相机输出512×2048 token，缓存为float16，兼容BF16编码器输出。

在H100上的完整仓库根目录运行，替换下面私有路径：

```bash
export PYTHONPATH="$PWD:$PWD/deployment:/path/to/openpi/src:/path/to/openpi/packages/openpi-client/src"
/path/to/openpi/.venv/bin/python -u training/corrector_pipeline.py \
  --run-root /path/to/new_corrector_run \
  --checkpoint /path/to/pi05_2500_bf16 \
  --data-root /path/to/lerobot \
  --openpi-root /path/to/openpi --k-step 5
```

流水线先在CPU导出当前检查点的视觉子树，再将完整episode分配到8张GPU提取；已有完整episode缓存可复用。随后用8进程DDP训练原作者的残差MLP，batch512，每卡microbatch8、累积8次，BF16自动混合精度，AdamW学习率3e-4、weight decay1e-4、余弦退火、30轮、early-stop patience5。目标为潜特征残差的余弦方向误差。没有逐帧随机拆分训练/验证，也不跨episode构造时间配对。

共用服务器时，提取进程的JAX显存上限设为每卡8%，训练PyTorch上限为10%；不终止其他GPU进程。每卡CPU亲和性最多8个逻辑核。共享任务竞争仍可能影响速度；八卡DDP性能由最慢rank决定。

训练后自动将最佳MLP导出为JAX可加载的 `deploy/mlp.npz`，验证PyTorch/JAX前向及动作梯度一致性，再按10Hz监控采样路径回放留出轨迹，校准正常误触发阈值。`training/corrector_finish.py` 可单独完成已训练目录的导出/校准。该留出集同时用于选模和校准，因此不作为独立测试集宣称抓取提升。

当前实际运行完成30轮，最佳第25轮，方向验证损失0.4516338671；权重保存在私有运行目录 `run_k5/best.pt`，部署使用 `deploy/`。参数量42,510,336，k=5即166.7ms。训练数据、视觉缓存、权重及归一化资产不随Git发布。
