# 上游源码与许可证

第三方文件遵循原作者许可证，版本记录在 `upstreams.json`，原始许可文件保存在各目录中。

| 组件 | 来源 | 许可证 |
| --- | --- | --- |
| OpenPI | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0；另保留 `LICENSE_GEMMA.txt` |
| pyAgxArm | [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) | LGPL-3.0-only |
| RTC-Anything | [PEILAB-PhysAI/RTC-Anything](https://github.com/PEILAB-PhysAI/RTC-Anything) | MIT |
| Kai0 参考文件 | [OpenDriveLab/kai0](https://github.com/OpenDriveLab/kai0) | Apache-2.0 |

源码快照保留训练/部署时使用的实现、依赖描述和锁文件，省略 `.git`、虚拟环境、缓存、构建产物及上游媒体资产。OpenPI 的训练与部署快照内容一致，分别保留以便在两种机器上建立独立环境。

本项目直接使用 RTC-Anything 的时间块缓存和官方 pyAgxArm 的 V189 协议实现。`rtc_piper/smoothing.py` 保留 Kai0 重叠平滑参考说明，并增加项目自己的三次位置/速度衔接；原参考源码、许可证和来源记录位于 `deployment/vendor/kai0_reference/`。

本仓库不包含预训练模型、微调权重或原始训练数据。模型及其基础组件的使用条款由相应发布方提供。
