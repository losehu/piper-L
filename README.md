# Piper-L：π0.5 训练与 π0.5 / VPP2 部署

单臂 Piper-L 的数据整理、八卡 π0.5 微调，以及双 RealSense 相机的本机命令行部署代码。启动参数可选择 π0.5 或已训练的 VPP2，两者共用已使用的 V189 机械臂接口、相机、限位裁剪、输出滤波和固件阻尼停止。

π0.5 当前部署选择 **2500 次更新后的最终检查点**，训练目录编号为 `2499`，采用 30Hz 动作时间轴、120Hz 插值、异步推理、12 帧三次位置/速度衔接和 50ms 二阶输出滤波。推理与机械臂控制在同一台机器完成。π0.5 **默认使用 JAX**，也可通过 `--backend realtime-vla` 选择 Triton 加速后端；两者保留相同的 RTC 和平滑配置。

VPP2 选择 **2250 检查点**、**5 步去噪**、32 帧动作。当前部署使用单卡 H100：视频/动作主干 BF16、编码器常驻 GPU，动作去噪和视频条件前向启用 `torch.compile` 与 CUDA graphs；CLIP FP16、VAE FP32 沿用参考实现。机械臂端通过两条错峰 HTTP 长连接异步请求，按实际图像采集时间跳过已过期前缀，在120Hz控制循环中持续衔接。原24GB显卡的本地 INT8 同步模式仍可显式选择。π0.5 的模型和控制参数保持不变。

π0.5 另有可选 **VLA-Corrector** 入口：冻结现有2500权重，用原48段训练数据训练42.5M残差MLP，6段留出数据用于选模与正常轨迹校准。支持只监控、截断重规划、截断＋OGG三种模式；Corrector与推理均在机械臂机器的GPU执行。训练和提取使用8张H100，代码位于 `corrector/` 和 `training/corrector_*.py`。原入口默认保持原π0.5行为。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`deployment/`](deployment/) | 当前命令行控制器、模型服务、相机同步、阻尼急停、回归测试 |
| [`training/`](training/) | 实际八卡训练配置、数据转换、训练归一化、固定留出集验证 |
| [`corrector/`](corrector/) | 视觉监控、恢复引导及独立控制入口，默认 JAX，可选 Triton 推理后端 |
| [`deployment/realtime_vla_runtime/`](deployment/realtime_vla_runtime/) | realtime-vla 后端、派生权重转换、JAX/Triton 离线对比 |
| [`preprocessing/`](preprocessing/) | 本次数据的首尾裁剪、第 76 段拆分、质量统计与筛选脚本 |
| [`tools/`](tools/) | 环境安装、BF16 权重导出/增量恢复、私有运行资产生成 |
| [`docs/`](docs/) | 训练和部署说明、上游版本与第三方许可证 |
| `deployment/vendor/`、`training/vendor/` | 实际使用的第三方源码快照及依赖锁文件 |

## 不随 Git 上传的内容

原始 HDF5、LeRobot 数据、图像/视频、模型权重、归一化资产、真实启动姿态、热身观测、环境、日志和机器私有配置都由 `.gitignore` 排除。`training/dataset_manifest.json` 只保存片段编号、裁剪区间、划分和文件哈希，不包含观测或动作数组。

测试使用 `deployment/tests/fixtures/start_states.synthetic.json` 中的合成姿态，不依赖私有训练数据。

## 环境与安装

实际部署环境为 Ubuntu 20.04、Python 3.11、RTX 5090 D v2；训练使用 8 张 H100。依赖固定在源码快照的 `pyproject.toml` / `uv.lock`，控制环境版本见 `deployment/requirements-control.txt`。需要可用的 NVIDIA 驱动、CUDA/JAX 环境；控制端还使用 SocketCAN、RealSense 和桌面 X11 全局键盘。

```bash
git clone https://github.com/losehu/piper-L.git
cd piper-L

# 控制环境：不加载模型、不打开机械臂
bash tools/setup_control.sh

# 模型环境，需先安装 uv
bash tools/setup_model.sh deployment
```

配置、私有运行资产准备以及启动方式见 [部署说明](docs/deployment.md)。八卡训练步骤见 [训练说明](docs/training.md)。

## 当前部署入口

已部署的 `192.168.50.102`，在该机器终端进入：

```bash
cd /home/arm/piper_pi05_rtc_cmd_20261112
```

新克隆仓库需先按[部署说明](docs/deployment.md)准备环境、硬件配置和私有资产，再进入仓库的 `deployment/` 目录。以下命令均在相应部署目录执行，**每次只选择一条启动命令**。

### JAX 与 realtime-vla 怎么选

| 要运行的组合 | 推理后端 | 启动命令 |
| --- | --- | --- |
| π0.5 + RTC + 平滑 | **JAX（默认）** | `./run.sh --model pi05 --backend jax` |
| π0.5 + RTC + 平滑 + **完整 Corrector** | **JAX（不用 realtime-vla）** | `./run_pi05_corrector.sh --mode full --backend jax` |
| π0.5 + RTC + 平滑 | realtime-vla / Triton | `./run.sh --model pi05 --backend realtime-vla` |
| π0.5 + RTC + 平滑 + **完整 Corrector** | realtime-vla / Triton | `./run_pi05_corrector.sh --mode full --backend realtime-vla` |

**不用 realtime-vla，就显式写 `--backend jax`。** 原配置下，省略整个 `--backend` 参数也默认使用 JAX。切换后端只改变模型计算方式；RTC、12帧衔接、50ms滤波和独立阻尼急停保持原设置。`--backend` 适用于上述 π0.5 / Corrector 入口，不是 VPP2 或 RLT 的后端开关。

realtime-vla 使用独立的 CUDA 12.8 Torch/Triton 环境和从原权重转换的派生资产，不修改原 JAX 权重或环境；需要先完成导出与离线数值检查。环境准备、资产路径、数值误差及计时范围见 [realtime-vla 详细说明](docs/realtime_vla.md)。

Corrector 的 `--mode` 可选：

| 模式 | 行为 |
| --- | --- |
| `monitor` | 运行 π0.5 并监控，不截断动作、不施加恢复引导 |
| `truncate` | 监控并在触发时截断旧 chunk、重新规划 |
| `full` | 监控 + 截断重规划 + OGG 恢复引导 |

普通 `run.sh --model pi05` 不启用 Corrector。完整纠错使用 `run_pi05_corrector.sh --mode full`。

### 其他入口

| 用途 | 命令 |
| --- | --- |
| 1500 step π0.5，保留 RTC 和平滑 | `./run_pi05_1500.sh` |
| 2500 step π0.5，关闭衔接和滤波，保留 RTC | `./run_pi05_no_smoothing.sh` |
| 2500 step π0.5，无平滑，执行完整50帧后再推理 | `./run_pi05_no_smoothing_no_rtc.sh` |
| VPP2 单卡 H100 远程异步 RTC | `./run_vpp2_h100.sh` |

VPP2 需要先在连接 H100 的工作站运行桥接器，详见[部署说明](docs/deployment.md)。

**启动命令在预检通过后会自动恢复、使能、低速对齐并开始动作，不再要求输入 RUN。** 全局 Space / Esc 或运行终端 Ctrl+C 触发独立进程的固件阻尼停止，不发送 Disable 卸力。固件恢复可能短暂失能，启动前应承托机械臂并清空对齐通路。

只想做保存观测的推理检查时，用 `check` 入口；这些命令不打开 CAN、相机或机械臂控制：

| 检查内容 | 命令 |
| --- | --- |
| 原 JAX π0.5 | `./check.sh --model pi05 --backend jax` |
| JAX + 完整 Corrector | `./check_pi05_corrector.sh --mode full --backend jax` |
| realtime-vla π0.5 | `./check.sh --model pi05 --backend realtime-vla` |
| realtime-vla + 完整 Corrector | `./check_pi05_corrector.sh --mode full --backend realtime-vla` |

## π0.5 模型与控制约定

| 项目 | 约定 |
| --- | --- |
| 观测 | 6 个实测关节弧度 + 夹爪实测开度（米） |
| 相机 | `cam0` 外部 RGB，`cam1` 腕部 RGB |
| 训练动作 | 6 轴相对当前观测状态的 delta，夹爪保持绝对开度 |
| 物理输出 | 50×7：绝对关节弧度 + 绝对夹爪开度（米） |
| 动作时间 | 30Hz；120Hz 控制循环插值，不把动作加速四倍 |
| 有效块 | 18 帧；每次预测返回后立即异步请求下一块 |
| 交接 | 12 帧三次位置/速度修正，按实际有效重叠缩短，不延长旧块 |
| 最终输出 | 独立限速参考 + 持续运行的临界阻尼二阶滤波，τ=50ms |
| 角度范围 | 每轮启动读取固件限位；所有目标按完整固件范围裁剪 |
| 急停 | 独立键盘/心跳守护，直接发送固件阻尼停止帧 |

未将示范角度范围用作软件行程边界，也不因少量编码器越过指令边界而误停。固件错误、驱动故障、无效数据、失联和控制流超时检查仍然保留。

Kai0 的重叠替换思路用于部署端；三次速度衔接是本项目适配。这里没有移植 Kai0 的模型融合、优势训练或推理前缀引导。

## 无硬件回归

```bash
cd deployment
cp -n config.example.yaml config.yaml
PYTHONPATH=..:. .venv/bin/python -m unittest -q \
  test_runtime test_models test_vpp2_rtc test_pi05_options \
  vpp2_network.test_transport test_corrector test_realtime_vla
```

测试覆盖 CAN 编码、禁止卸力、固件限位裁剪、相机异步重试、动作过期、插值并发快照、速度衔接、输出滤波和权重身份选择，不启动真实机械臂。训练损失或离线回放结果不等于真机抓取成功率；最终 2500 权重与三次速度衔接组合的闭环效果需实机评估。

上游版本与许可见 [第三方说明](docs/third_party.md) 和 [版本清单](docs/upstreams.json)。
