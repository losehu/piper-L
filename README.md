# Piper-L：π0.5 训练与本机 RTC 部署

单臂 Piper-L 的数据整理、八卡 π0.5 微调，以及双 RealSense 相机的本机命令行部署代码。此仓库整理自 2026-11-12 实际使用的训练、部署源码。

当前部署选择 **2500 次更新后的最终检查点**，训练目录编号为 `2499`，采用 30Hz 动作时间轴、120Hz 插值、异步推理、10 帧三次位置/速度衔接和 40ms 二阶输出滤波。推理与机械臂控制在同一台机器完成。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`deployment/`](deployment/) | 当前命令行控制器、模型服务、相机同步、阻尼急停、回归测试 |
| [`training/`](training/) | 实际八卡训练配置、数据转换、训练归一化、固定留出集验证 |
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

```bash
cd deployment
# 配置硬件编号，放好权重与私有运行资产后：
./run.sh
```

**`run.sh` 在预检通过后会自动恢复、使能、低速对齐并开始动作，不再要求输入 RUN。** 全局 Space / Esc 或运行终端 Ctrl+C 触发独立进程的固件阻尼停止，不发送 Disable 卸力。固件恢复可能短暂失能，启动前应承托机械臂并清空对齐通路。

## 模型与控制约定

| 项目 | 约定 |
| --- | --- |
| 观测 | 6 个实测关节弧度 + 夹爪实测开度（米） |
| 相机 | `cam0` 外部 RGB，`cam1` 腕部 RGB |
| 训练动作 | 6 轴相对当前观测状态的 delta，夹爪保持绝对开度 |
| 物理输出 | 50×7：绝对关节弧度 + 绝对夹爪开度（米） |
| 动作时间 | 30Hz；120Hz 控制循环插值，不把动作加速四倍 |
| 有效块 | 18 帧；每次预测返回后立即异步请求下一块 |
| 交接 | 10 帧三次位置/速度修正，按实际有效重叠缩短，不延长旧块 |
| 最终输出 | 独立限速参考 + 持续运行的临界阻尼二阶滤波，τ=40ms |
| 角度范围 | 每轮启动读取固件限位；所有目标按完整固件范围裁剪 |
| 急停 | 独立键盘/心跳守护，直接发送固件阻尼停止帧 |

未将示范角度范围用作软件行程边界，也不因少量编码器越过指令边界而误停。固件错误、驱动故障、无效数据、失联和控制流超时检查仍然保留。

Kai0 的重叠替换思路用于部署端；三次速度衔接是本项目适配。这里没有移植 Kai0 的模型融合、优势训练或推理前缀引导。

## 无硬件回归

```bash
cd deployment
cp -n config.example.yaml config.yaml
.venv/bin/python -m unittest -q test_runtime
```

测试覆盖 CAN 编码、禁止卸力、固件限位裁剪、相机异步重试、动作过期、插值并发快照、速度衔接、输出滤波和权重身份选择，不启动真实机械臂。训练损失或离线回放结果不等于真机抓取成功率；最终 2500 权重与三次速度衔接组合的闭环效果需实机评估。

上游版本与许可见 [第三方说明](docs/third_party.md) 和 [版本清单](docs/upstreams.json)。
