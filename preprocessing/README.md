# 本次采集数据整理脚本

这些脚本复现本轮数据处理流程，依赖私有数据和记录中的片段编号。它们不是对任意数据自动判定抓取成功率的通用工具。

运行前设置独立工作目录和原始采集副本：

```bash
export PIPER_DATA_WORKSPACE=/path/to/private_preprocessing_workspace
export PIPER_SOURCE_DIR=/path/to/copied/piper_datasets
export PIPER_CAPTURE_SSH=your_capture_host_alias
export PIPER_CAPTURE_DATA_ROOT=/path/on/capture_host/piper_datasets
```

处理顺序：

1. `trim_accepted.py`：核对本地副本与采集机哈希，筛选 manifest 中通过的 65 段，去首尾静止，保留中间停顿。
2. 备份裁剪后的第 76 段到 `$PIPER_DATA_WORKSPACE/work/episode76_before_split/episode_000076.hdf5`，运行 `split_episode76.py` 拆成 76/87 两段。
3. 备份拆出的 87 到同一备份目录的 `episode_000087_before_retrim.hdf5`，运行 `retrim87.py`，额外去掉第二段开头 53 帧。
4. `export_videos.py`：生成双相机并排视频供人工复核。
5. `screen_quality.py`：计算全样本数值指标与抽帧面板。
6. `package_screening.py`：应用本轮人工复核名单，生成 54 段候选子集及报告。

输出放在该工作目录的 `outputs/piper_65_trimmed`、`outputs/quality_review`、`outputs/piper_screened`。只运行需要的阶段，并保留脚本要求的来源备份。原脚本的固定 76/87 分段区间、人工介入/待复核名单只适用于本轮数据。

数据、视频、抽帧和质量日志都不进入 Git。训练阶段直接使用整理好的 54 段 HDF5 及 `training/dataset_manifest.json`；不要求在训练服务器重复执行人工筛选流程。
