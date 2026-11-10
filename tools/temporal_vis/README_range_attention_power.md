# 按帧、按框统计回波 Power

入口：`tools/temporal_vis/compare_range_attention_regions.py`。

主表：输出目录中的 **`all_frames_box_power.csv`**。
每行是某一帧中的一个 GT 框或某种提框方法的一个框。
不同 GT 框分别计算；同一个 GT ID 出现在不同帧时也分别保留。
同帧重复 GT ID 通过 `box_index` 区分，不合并或跳过。

## 统计规则

- 统一使用原始 NPZ 的 `power_val[2]`，在线性存储尺度上计算，不使用 RA 特征值。
- 点集沿用当前对比脚本的模型稀疏输入（前 175 个 range、每个 250 点）和分析 ROI，不是完整稠密雷达张量。
- 每个框独立统计框内全部回波，不只统计提框支持点。
- 每行包含点数、均值、最小值、最大值、中位数、P5/P95。
- 最小值、最大值是实际观测极值；P5/P95 是另外保存的分位数。
- 没有回波的框保留，点数为 0，均值和极值留空。
- 重叠框各自统计属于自己的点，不为了跨框去重而删除点。
- 所有帧追加到同一张大表，主表不对不同框或不同帧再次求平均。

## 主表字段

| 字段 | 含义 |
| --- | --- |
| `scene`, `order`, `radar_frame_id`, `occ_frame_idx` | 场景和帧定位信息 |
| `method` | `GT`、`P0_power`、`P0_descriptor` 或 `P1_range_attention` |
| `box_kind`, `box_id`, `box_index`, `gt_id` | GT/自提框类型、框编号及 GT ID |
| `box_key` | 场景、帧、方法、框索引、包含空间组成的唯一键 |
| `center_x_m`, `center_y_m`, `center_z_m`, `center_range_m` | 雷达坐标系下的框几何中心及距离 |
| `n_points`, `mean_power`, `min_power`, `max_power` | 此框内全部回波的点数、均值和上下极值 |
| `median_power`, `p05_power`, `p95_power` | 此框内全部回波的中位数及分位数 |
| `support_*` | 此自提框的实际提框支持点统计 |
| `inside_gt_*` | 此自提框中位于所有 GT 框并集内的点的统计 |
| `outside_gt_*` | 此自提框中位于所有 GT 框并集外的点的统计 |
| `gt_relation`, `gt_overlap_fraction`, `overlapping_gt_ids` | 自提框内回波与 GT 的重合关系，仅用于事后统计 |

查看每个自提框的 GT 外回波强度，使用
`outside_gt_mean_power`、`outside_gt_min_power`、`outside_gt_max_power`。
筛选 `gt_relation=outside_gt`，得到框内回波完全没有进入任何 GT 的自提框。
该标签按回波包含关系定义，不表示框之间的几何交集一定为零，也不表示这些回波一定是噪声。

## 运行示例

```bash
cd /home/user1/projects/RadarOcc
source use_radarocc5060.sh

python tools/temporal_vis/compare_range_attention_regions.py \
  --checkpoint work_dirs/box_z_p0_20pct_scratch_seed42/Single-frame/best_SSC_mean_epoch_9.pth \
  --ann-file data/annotations/kradar_dict_test_official_temporal_doppler8.pkl \
  --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
  --show-gt --gt-space xy --power-space xy \
  --bbox-root data/K-RadarOcc/train \
  --output-dir outputs/ra_box_comparison/scene3_80_200_per_box_power
```

输出目录需要为空。RGB、GIF 等参数可沿用此前的命令。
`--show-gt` 同时启用 GT 读取、显示和 GT 统计。

`--power-space xy`：GT 框和自提框均按 XY 投影包含点，高度受分析 ROI 限制。
`--power-space xyz`：按实际三维框包含点；不会改变图中的框或提框算法。
省略该参数时沿用 `--gt-space`。两种情况下都限制到共同分析 ROI。

`box_power.csv` 是按框和点集分行的明细；`frame_power.csv`、`power_summary.csv`
保留为帧并集、全帧汇总参考。按目标强度分布查阅时，优先使用主表。
本次修改负责输出统计；提框仍沿用原有规则，未用 GT 的统计值设置预测阈值。
