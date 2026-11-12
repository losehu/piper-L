# 本机部署

## 1. 生成并修改本地配置

在仓库根目录执行：

```bash
bash tools/setup_control.sh
bash tools/setup_model.sh deployment
cp -n deployment/config.example.yaml deployment/config.yaml
```

修改 `deployment/config.yaml` 的 `adapter_serial` 和两个相机序列号。`cam0` 是外部相机，`cam1` 是腕部相机，RGB 640×480、30Hz。

控制端固定使用 Piper-L V189、`can1`、1Mbps；应提前配置好 CAN 网卡。USB-CAN 编号用于避免误控其他适配器。`lock_file` 必须与其他控制/数采程序共用同一个锁文件，部署到已有机器时保留原来的锁路径，避免两套程序同时控制。示例使用当前用户的 `~/piper_calibration/.can1.lock`。

当前默认配置选择 `model_step=2500`、`checkpoint_index=2499`。服务端和客户端都检查模型身份，避免旧服务仍使用 1500 权重。

## 2. 放置私有模型与运行资产

```text
deployment/
  config.yaml                         # 本机配置，Git 忽略
  checkpoints/pi05_piper_2500_bf16/
    params/                           # OpenPI BF16 推理权重
    assets/local/piper_grasp_train/    # 训练归一化资产
  training_start_states.json          # 示范准备姿态，Git 忽略
  offline_sample.npz                  # 模型热身观测，Git 忽略
```

已有部署可直接复制上述私有文件；无需重新下载训练数据。若从私有筛选后的 HDF5 生成运行资产，可使用数据环境执行：

```bash
training/vendor/openpi/.venv/bin/python tools/export_runtime_assets.py \
  --selected-dir /path/to/piper_screened \
  --output deployment
```

`--selected-dir` 下应有 manifest 中的 `raw/episode_*.hdf5`。生成器不覆盖既有运行资产。

## 3. 启动

在本机桌面的交互终端运行：

```bash
cd deployment
./run.sh
```

此命令会自动加载本机模型、完成 CAN/相机预检、恢复固件急停、保持当前姿态使能、以 5°/s 对齐示范起点，然后进入高跟随策略执行。准备阶段不动夹爪。启动前应承托机械臂并清空对齐通路。

全局键盘依赖可访问的 X11 会话。可通过环境变量设置 `DISPLAY`、`XAUTHORITY`；默认使用当前 Linux 用户的桌面路径。没有可用的全局急停键盘时拒绝进入动作阶段。

Space、Esc 或 Ctrl+C 会锁定停止并由独立进程发送 `0x150/01` 阻尼停止；不发送 Disable。停止后的新一轮动作需要重新启动进程。

## 4. 参数与诊断

- `execute_horizon=18`：30Hz 时间轴上的有效前缀。
- `chunk_smoothing_method=cubic_velocity_bridge`、`chunk_smoothing_steps=10`：保留旧轨迹当前端点和前进速度，再收敛到新预测；有效重叠不足时缩短衔接。
- `command_filter_tau_s=0.04`：持续运行的二阶输出滤波，匀速段额外延迟约 80ms，不在换块时重置。
- `policy_joint_speed_deg_s`：60/60/60/90/90/90；夹爪速度 0.1m/s、力 0.5N。
- 两路相机配对窗口 25ms；暂时缺帧时重试，不伪造时间戳或延长旧动作有效期。

`logs/last_policy_run.json` 保存本轮权重身份、实际参数、相机等待和停止原因。`policy_chunks_*.npz` 保存最近的原始预测供离线分析。这些运行数据全部被 Git 忽略。

以下命令均不会使能或移动机械臂，但会占用相应资源，应避开正在执行的其他控制程序：

```bash
# 只读 CAN 限位和相机反馈检查
.venv/bin/python -m rtc_piper.cli sensors --seconds 3

# 可选：保存观测的模型推理检查，使用 GPU
./check.sh
```

模型服务本身只监听 `127.0.0.1:8000`，不直接操作 CAN 或相机。
