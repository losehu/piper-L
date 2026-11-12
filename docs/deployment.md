# 命令行部署

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
# 等同于 ./run.sh --model pi05
# 使用 VPP2：
./run.sh --model vpp2
# 使用 VPP2 + 动作去噪编译：
./run.sh --model vpp2 --compile
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

π0.5 模型服务只监听 `127.0.0.1:8000`，不直接操作 CAN 或相机。

## 5. VPP2 本地同步模式

把 `config.yaml` 的 `vpp2.root` 指向已准备好的 VPP2 私有资产目录；也可用 `VPP2_DEPLOY_ROOT` 指定。其目录结构为：

```text
vpp2-assets/
  .venv/bin/python                     # 独立 Python3.11 / torch2.7.1+cu128
  config/train_config.yaml
  weights/best_action.pt               # 已选2250检查点
  weights/video_int8.safetensors
  weights/video_int8_manifest.json
  weights/Wan2.1-I2V-14B-480P/
    Wan2.1_VAE.pth
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
  assets/dataset_stats.json
  assets/text_context.pt
```

独立环境依赖见 `deployment/vpp2_runtime/requirements.txt`，VPP2 源码固定在 `deployment/vendor/vpp2`。该模式使用已有2250权重、训练归一化和文本缓存，模型计算全部在机械臂端。设置 `vpp2.inference=local_sync`，并清除 `VPP2_REMOTE_PORT` 后可选择该模式。

默认 `vpp2.denoising_steps=5`，显式设成 `10` 可复现旧采样设置；只影响 VPP2。`vpp2.torch_compile` 默认 false，`--compile` 可启用。编译目标是 `MoT.forward_action_with_video_cache`，关闭 CUDA graphs 以控制24GB显存占用；没有编译整个视频主干。首次编译在读取保存的观测热身时完成，成功前不会进入机械臂使能阶段。

VPP2 服务监听 `127.0.0.1:8768`。启动器在持有共享 CAN 文件锁后，只释放本部署拥有的空闲模型服务，避免两种模型同时占用显存；其他程序或有连接的模型不被结束。编译设置或采样步数变化时重新加载本部署的 VPP2 服务。服务和响应均核对2250身份、采样步数、动作单位、32×7形状及请求编号。

VPP2 接收原始640×480双目RGB，复现训练时224×168缩放和上下黑边，再横向拼接。服务已把关节 delta 还原成绝对弧度，控制端不再次加当前状态；夹爪为绝对米。视频主干权重使用 INT8 存储并临时还原 BF16 运算，动作专家 BF16，CLIP/VAE 分阶段上显卡。

本地模式采用同步 `plan_execute` 调度：以30Hz完整执行32帧、120Hz插值与滤波，结束后保持块末目标；等待姿态稳定至少300ms，再采集停稳之后的双相机和状态观测，发起下一次推理，返回后执行下一块。动作执行期间不请求下一块，也不复用停稳前的观测。首次动作同样先停稳取观测。

推理请求由工作线程等待响应，用于让120Hz位置保持命令和独立键盘急停继续运行，动作块之间仍严格串行。超时、固件故障或控制心跳异常仍停止。VPP2 的该调度不改变 π0.5 的异步 RTC、块长度或平滑参数。VPP2 闭环抓取尚未验证。

只做保存观测推理、不打开 CAN 或相机：

```bash
./check.sh --model vpp2 --compile
```

该检查会加载模型并占用GPU和共享文件锁，不能与其他控制程序同时运行。

## 6. VPP2 单卡 H100 异步 RTC（当前部署）

H100 保留原2250动作权重和32帧/30Hz预测窗口，5步去噪。视频和动作主干以BF16常驻GPU，无INT8权重还原或CPU/GPU逐请求搬运；CLIP FP16、VAE FP32遵循原参考实现。动作去噪与视频条件前向使用 `torch.compile` 和 CUDA graphs。启动完成前执行热身及 A→B→A 输入新鲜性检查。

在 H100 上准备与本地模式相同的私有资产，其中视频权重改为原始 `weights/video_step010000.pt`；`vpp2_runtime/server.py` 的 BF16 分支直接装载它。将本仓库 `deployment/vpp2_network`、`deployment/vpp2_runtime` 和 `deployment/vendor/vpp2/src` 放在服务器部署目录，模型 Python 环境使用 PyTorch2.7.1。服务器只监听回环端口19868，不连接机械臂。

在能通过 SSH 同时连接 H100 和机械臂的工作站执行：

```bash
cp -n deployment/h100_runtime.example.json deployment/h100_runtime.json
# 填写 SSH 别名、服务器部署/环境/资产/源码/热身观测路径
python3 tools/vpp2_h100_bridge.py --config deployment/h100_runtime.json
```

桥接器启动或复用本项目拥有的单卡服务，建立工作站本地21968与机械臂回环21969两段加密转发。保持桥接器运行；工作站或隧道退出会中断远程推理。桥接器不使能、不启动机械臂，不终止其他GPU任务。可使用 `--stop` 关闭本桥接器及隧道，H100模型仍保温。

机械臂配置的 `vpp2` 子项设置如下，其他配置保留：

```json
{"inference":"h100_async_rtc","remote_port":21969,"max_inflight":2,
 "denoising_steps":5,"torch_compile":true}
```

在机械臂机器的交互终端运行：

```bash
cd /home/arm/piper_pi05_rtc_cmd_20261112
./check_vpp2_h100.sh  # 仅保存观测推理，不连接机械臂或相机
./run_vpp2_h100.sh    # 自动准备后开始动作；Space/Esc/Ctrl+C阻尼停止
```

新入口也可通过环境变量选择远程模式，原 `./run.sh` 仍默认π0.5。远程不可用时拒绝启动，不自动回退本地模型。

传输使用 HTTP/1.1 长连接、二进制请求及无损 WebP。机械臂端先执行与旧路径逐像素相同的 JPEG95→224×168缩放→上下黑边处理，再无损上传；每个请求核对像素哈希、编号、2250身份、5步计数、动作单位与32×7形状。

两路请求按最近8次往返时间中位数的一半持续错峰，单个固定GPU线程依次计算。首次预测在保持姿态期间完成，随后每块使用真实采集时间，不重写时间戳；已过去的前缀被跳过，有效期止于采集时刻加32/30秒。新旧有效轨迹通过三次位置/速度修正衔接，并复用持续运行的输出滤波。这里是部署端重叠续块，没有增加VPP2训练时未用的去噪前缀约束。过期、网络错误或预测断流触发独立阻尼停止，不补播过期动作。

无需机械臂的真实网络时序回放：

```bash
PYTHONPATH=.:vendor/pyAgxArm:vendor/openpi/packages/openpi-client/src \
 .venv/bin/python -m vpp2_network.replay_rtc --sample offline_sample.npz \
 --config config.yaml --output logs/h100_rtc_replay.json \
 --requests 160 --camera-age 0.0666666667 --inflight 2
```

回放重复保存的图像/状态，使用模拟采集时间并额外预留两帧相机延迟；报告不代表真实相机或抓取成功率。代码不打开CAN，但取得共享文件锁以避开真实控制。
