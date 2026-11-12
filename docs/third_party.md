# 上游源码与许可证

第三方文件遵循原作者许可证，版本记录在 `upstreams.json`，原始许可文件保存在各目录中。

| 组件 | 来源 | 许可证 |
| --- | --- | --- |
| OpenPI | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0；另保留 `LICENSE_GEMMA.txt` |
| pyAgxArm | [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) | LGPL-3.0-only |
| RTC-Anything | [PEILAB-PhysAI/RTC-Anything](https://github.com/PEILAB-PhysAI/RTC-Anything) | MIT |
| Kai0 参考文件 | [OpenDriveLab/kai0](https://github.com/OpenDriveLab/kai0) | Apache-2.0 |
| VPP2 | [roboterax/video-prediction-policy-2](https://github.com/roboterax/video-prediction-policy-2) | MIT |

源码快照保留训练/部署时使用的实现、依赖描述和锁文件，省略 `.git`、虚拟环境、缓存、构建产物及上游媒体资产。OpenPI 的训练与部署快照内容一致，分别保留以便在两种机器上建立独立环境。

本项目直接使用 RTC-Anything 的时间块缓存和官方 pyAgxArm 的 V189 协议实现。`rtc_piper/smoothing.py` 保留 Kai0 重叠平滑参考说明，并增加项目自己的三次位置/速度衔接；原参考源码、许可证和来源记录位于 `deployment/vendor/kai0_reference/`。

VPP2 源码快照来自已准备的部署目录，包含本次 Piper 输入/动作适配所用实现。`deployment/vpp2_runtime/` 提供 INT8/BF16 装载及5步采样，`deployment/vpp2_network/` 提供单卡H100常驻、编译与CUDA graphs、无损图像传输及异步续块；机械臂控制复用本项目的 V189 实现。

本仓库不包含预训练模型、微调权重或原始训练数据。模型及其基础组件的使用条款由相应发布方提供。
