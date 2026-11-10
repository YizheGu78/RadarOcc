# Doppler 时序 Query 减少率检查

只做离线检查，不训练，不删除原始数据，不改变单帧 RadarOcc 或历史候选点。
输入为原始 Doppler8 的前 175×250 个点，输出 mask 按原 NPZ 点顺序保存。

## 一次运行五个场景

```bash
cd /home/user1/projects/RadarOcc
bash run_doppler_query_audit.sh
```

默认场景 **1、2、3、15、22**；每个场景 order **80–200、stride=5、25 帧**。
场景 1、2 来自完整 val PKL；3、15、22 来自完整 test PKL。
脚本从指定 annotation 文件中查找场景，要求每个场景恰好一个来源且能选满 25 帧。
这些场景用于探索，不是独立验证后的阈值结论。

默认参数：

| 参数 | 默认值 |
| --- | --- |
| `--thresholds` | 0.1、0.2、0.3、0.5 m/s |
| `--visual-threshold` | 0.2 m/s |
| `--static-keep-ratio` | 0.25 |
| `--max-peak-std` | 0.2 m/s，环形峰离散程度上限 |
| `--min-concentration` | 0.8，环形合成向量长度下限 |
| `--max-target-radial-mps` | 30，别名保护采用的世界径向速度绝对值先验上限 |
| `--doppler-sign` | both，两个假设分别输出，不用 GT 挑符号 |
| `--gt-space` | xy |
| GT 评价 ROI | X∈[0,85)、Y∈[-60,60)、Z∈[-5,8) |

Query 减少率的分母为完整输入 **43,750**，不先裁剪 ROI；另外输出 ROI 内保留率。
评估 ROI 只限制 GT 统计，不参与推理决策。
所有峰/残差/Query mask 均在读取当前帧 GT 前计算。

## 三个对照

1. **doppler_heuristic**：静止假设一致的点按 range 分层、方位/俯仰排序，等间隔保留约 25%（逐 range 向上取整）；
   非静止 Doppler 证据、不可靠峰、无可用自车速度的点全部保留。
2. **alias_guarded**：只有速度上限假设内不存在非静止别名候选时，才削减静止一致点。
   速度表范围窄、候选别名多时，该方法可能全部保留，减少率为零。
3. **spatial_same_budget**：不使用 Doppler，按 range 分层和角度排序选代表点，Query 数与启发式逐帧完全相同。
   用于比较相同计算预算下的 GT 覆盖率。

不能把 `doppler_heuristic` 当成已经解决速度混叠的过滤器。
`alias_guarded` 也只在给定的径向速度上限假设内保护别名，不判断前景类别或横向运动。
没有 GT 参与上述三种策略的取点。

## Doppler 和自车运动

- 速度表读取官方 `arr_doppler.mat`；周期按 `N × abs(bin_step)` 计算，不用 max−min。
- 原始前三通道为 Top-3 峰值功率，3:6 为对应 bin，先做功率加权环形统计。
- 跨速度轴边界的相邻峰不会因普通线性平均而变成假零速；多峰不一致则回退保留。
- 从当前帧和 annotation 中前一个真实帧的 pose、timestamp 估计雷达原点速度，包含杆臂位移。
  对 order 80，使用 order 79，而非上一张抽样图 order 75；不读取未来帧。
- 静止目标期望径向速度为 `-dot(ego_velocity, line_of_sight)`，需先统一 Doppler 正负号。
- `wrap(sign*observed + ego_projection, period)` 只是静止一致性残差，不是真实速度。
- 在假设速度范围内枚举可能的折叠阶数数量；不强行挑选某个阶数并称为真实运动。
- 自车位姿或时间间隔不可靠时，本帧 Doppler 策略保留全部 Query。

## 查看输出

默认目录 `outputs/doppler_query_audit/five_scenes_25frames/`：

- **`index.html`**：每帧两种符号的可视化入口；图片展示默认 0.2 m/s 阈值。
- **`frame_metrics.csv`**：每帧每种阈值/符号/方法的减少率及 GT 覆盖情况。
- **`summary_by_scene.csv`**、**`summary_all_scenes.csv`**：逐场景和全场景汇总。
- **`gt_per_box.csv`**：每个 GT 框原有、保留、取消的 Query 数，以及是否从非空变为剪空。
- **`frame_diagnostics.csv`**：自车速度、时间间隔、跨边界峰与峰可靠性。
- **`summary.md`**、**`manifest.json`**：统计解释、参数和可复现记录。
- `scene*/masks/order*.npz`：各阈值、符号、方法的完整 mask。

可视化：蓝色为非静止 Doppler 证据，金色为静止一致，灰色为不确定且保留；
红色表示取消时序 Query，绿色框为 GT。第一行总览，第二行前方细节。
图中没有显示的 XY 范围仍计入完整输入减少率。

```python
import numpy as np
data = np.load('输出目录/scene3/masks/order00080.npz', allow_pickle=False)
keep = np.unpackbits(data['keep_bits'], axis=-1, count=int(data['n_tokens'])).astype(bool)
# keep 的轴为 [sign, threshold, method, original_npz_point_id]。
print(data['signs'], data['thresholds_mps'], data['methods'])
```

模型在线入口会按 power 排序 token，接入训练时必须进行索引顺序映射，不能直接套用本 NPZ 顺序 mask。

## 更换参数

```bash
# 输出目录必须为新目录或空目录；额外参数直接追加。
bash run_doppler_query_audit.sh outputs/doppler_query_audit/five_scenes_keep50 \
  --static-keep-ratio 0.5 --visual-threshold 0.3

# 小规模检查：1 个场景、1 帧；保留完整 43,750 点。
bash run_doppler_query_audit.sh outputs/doppler_query_audit/smoke_scene3 \
  --scenes 3 --start-order 80 --end-order 80 --frames-per-scene 1
```

`--no-plots` 只计算统计和保存 mask；`--plot-max-points 12000` 只减少绘图点数，统计仍用全量点。
`--no-gt` 可禁用 GT，届时不产生有效 GT 覆盖结论。
这些结果估计减少的时序查询量，不直接等于整模型加速比，也不是 Occupancy IoU。
