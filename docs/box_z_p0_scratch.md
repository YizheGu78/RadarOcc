# P0 Z0–Z3 全网络随机初始化实验（20%）

本实验独立于之前“加载单帧 checkpoint、冻结原 RadarOcc”的实验。
原 RadarOcc 和新增时序模块一起训练，P0位置、框缓存、映射公式与相对残差适配器保持原方案。
改为随机起步并不能保证映射之间出现差异，也不代表适配器饱和已经修复。

## 启动

```bash
cd /home/user1/projects/RadarOcc
bash run_box_z_p0_20pct_scratch.sh
```

默认根目录 `work_dirs/box_z_p0_20pct_scratch_seed42`，每组15轮、seed42。
单帧对照、Z1、Z2、Z3、Z0依次运行。五组使用相同20%训练子集与验证子集、相同初值和训练预算。
每组独立建立原网络参数与优化器，原网络不冻结；不存在加载已训练初始化权重的步骤。
融合输出投影仍从零开始，Z3高度Linear仍为零初始化，保持初始Z3=Z2的设计。

可以自定义输出目录、每组轮数与种子，参数顺序与旧预训练脚本不同：

```bash
bash run_box_z_p0_20pct_scratch.sh \
  work_dirs/box_z_p0_20pct_scratch_seed42 15 42
```

不要把checkpoint路径作为第一个参数；第一个参数是新的实验根目录。
缓存仍复用 `data/box_z_pairs/train20` 和 `val20`，不需要重新提框。
数据、代码、初始方式、预算的记录在根目录 `experiment.json`，发生变化时要求新目录。
中断后原命令会从各组自己的 `latest.pth` 恢复，已完成组会复用。

## 与原实验的区别

| 项目 | 之前 | 本次 |
|---|---|---|
| 原 RadarOcc 初始化 | 单帧预训练checkpoint | 随机初始化 |
| 原 RadarOcc 参数 | 冻结 | 更新 |
| 新增模块参数 | 更新 | 更新 |
| 单帧基线 | 对初始化checkpoint直接验证 | 同预算从头训练 |
| 默认轮数 | 10 | 15 |
| 学习率 | 1e-4 | 3e-4，沿用原单帧从头训练设置 |
| P0位置与Z公式 | 原方案 | 同一方案 |

新配置显式设置 `load_from=None`、`resume_from=None`、`freeze_original=False`。
每组按相同seed重新构造模型；预检对五组共有参数做SHA256核对。
Z3独有的16个高度映射参数不参与共有参数的hash，其零初始化另由融合模块保证。

## 运行前验证

脚本自动进行真实帧预检，工程模型完成两步更新后丢弃。
原网络参数必须全部可训练、实际产生梯度并更新；Z3必须在零输出投影开始更新后收到数据梯度。
正式训练会重新构造随机模型，预检不提供初始化权重。

只运行缓存检查和预检：

```bash
BOX_Z_CHECK_ONLY=1 bash run_box_z_p0_20pct_scratch.sh
```

## 结果

- `comparison.md/csv/json`：五组各自验证集最佳结果和轮次。
- `epoch_iou.csv`：所有组逐轮的15项IoU，含12.8/25.6/51.2 m评价范围。
- `sampling_diagnostics.csv`：Z0–Z3逐轮采样、Gate、相对贡献与Z3参数统计。
- `各组/val_metrics.jsonl`：评价Hook直接保存的未舍入IoU。
- `各组/best_SSC_mean_epoch_N.pth`：依据未舍入SSC_mean选出的最佳权重。
- `各组/latest.pth`：断点恢复入口；常规权重只保留最近两轮，最佳权重另存。

单帧基线为本轮实际训练得到的模型，不再直接引用旧实验的22%或25.1%作为成绩。
这是更改了优化方式的一轮新实验；结果应单独报告，保留旧冻结实验。
