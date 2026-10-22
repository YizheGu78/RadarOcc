# 动态目标 KNN 与 Occupancy 诊断

`audit_dynamic_objects.py` 包含两项实验，不训练模型，不需要 PyTorch/CUDA。
依赖 numpy、scipy；输出图片时需要 matplotlib。请从仓库根目录运行。

## 实验 2：同目标匹配

```bash
conda activate radarocc5060
python tools/temporal_vis/audit_dynamic_objects.py match \
  --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
  --gaps 1 2 3 --radius 2 --k 8 \
  --output-dir outputs/dynamic_object_audit/scene3_match
```

默认 ann-file 为 `data/annotations/kradar_dict_test_official_temporal_doppler8.pkl`。
标注额外从 `data/K-RadarOcc/train/<scene>/bbox/` 读取；按 `occ_frame_idx`
关联 `bbox*.npy`、`bbox_token*.npy`、`object_category*.npy`，不按 radar_frame_idx。
`--bbox-root` 可切换数据根目录。

输出：

- `matching_per_object_history.csv`：一行对应一个当前目标和一个历史帧。
- `matching_by_speed_gap.csv`：按速度区间和实际 order 间隔汇总。
- `run_metadata.json`：参数与指标口径。

`n_queries` 是当前框内、归属不冲突的 P1 查询数。历史存在同 ID 框，且有
至少一个归属不冲突的 P1 token 时，当前查询才进入 `n_eligible` 分母。

| 字段 | 含义 |
|---|---|
| n_no_history | 没有同目标历史 P1 回波，单列，不计入召回率分母 |
| n_outside_radius | 同目标历史回波存在，但最近一个也不在搜索半径内 |
| n_topk_miss | 半径内有同目标回波，但未进入普通 KNN 的 K 个候选 |
| n_success | 至少选中一个同目标回波 |
| same_object_recall | n_success / n_eligible |
| selected_same_precision | 选中的同目标候选数 / 全部选中候选数；分母限有效查询 |

候选搜索仍包含全部可用历史 P1，包括背景与归属冲突点；冲突点不作为同目标真值。
前三种有效匹配结果严格划分 n_eligible。零分母输出空值，不伪装成零召回率。
搜索使用与 `temporal/fusion/local_attention.py` 相同的 scipy KDTree、半径和逐帧 K。
这是几何候选诊断，不涉及 attention 权重，也不代表物理散射点的一一对应。

默认 `--history-mode dataset`，`--gaps 1 2 3` 表示当前 split 中的前三个历史位置，
与 TemporalKradarDataset 的 T=4 构造一致；场景开头无效的占位历史不统计。
CSV 的 `frame_gap` 始终是实际 order 差，`delta_t_sec` 使用真实时间戳。
如果需要固定的场景 order 差，可以显式使用 `--history-mode scene-order`，但它在
split 有缺帧时可能与实际模型使用的历史不同。单独审计更大间隔不等于模型已使用这些帧。

速度使用同 ID 框的**几何中心**，经过 LiDAR Ego Pose 变换后，对真实时间戳做局部
最小二乘拟合，取世界坐标 XY 速度。默认窗口为前后各 2 个 order；至少 3 次观测。
拟合位置 RMSE > 0.5 m 时归为 unknown，可用 `--max-speed-rmse` 调整。
速度区间为 [0,0.5)、[0.5,2)、[2,5)、[5,+inf) m/s；拟合可使用未来标注，
仅供离线诊断，不作为可部署模型输入。

P1 属于雷达坐标；框属于 LiDAR 坐标，使用现有 `radar_to_lidar_default()`，
也支持 `--extrinsic` 覆盖。框遵循底面中心、全尺寸、弧度 yaw 约定。
预处理框经过扩张，框内回波只是同目标归属近似。建议抽查目标框和回波，特别是高速、
重叠和框尺寸异常的样本；不应把 ID 连续性直接当作已验证的轨迹真值。
`--ego-mask-radius` 默认 0；比较 mask37 模型时应使用 3.7，保持输入候选规则一致。

## 实验 3：成对 Occupancy 对比

已有 `tools/test.py --show --show-dir` 会导出
`<输出根目录>/<scene>/<lidar_token>/pred_c.npy`，不需要修改网络或训练流程。
例如分别导出两套模型（GPU 推理；下面命令会处理整个 test split）：

```bash
python tools/test.py \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py \
  work_dirs/radarocc_small_singleframe_geomfix/best_SSC_mean_epoch_5.pth \
  --show --show-dir outputs/audit_single_predictions \
  --cfg-options data.workers_per_gpu=0

python tools/test.py \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py \
  work_dirs/radarocc_small_temporal_t4_ego_attention_p1/best_SSC_mean_epoch_3.pth \
  --show --show-dir outputs/audit_knn_predictions \
  --cfg-options data.workers_per_gpu=0
```

应使用与各 checkpoint 一致的训练配置，并确保相同测试集、输入预处理、GT 和网格。
上面路径是本地现有模型的示例，不代表它们的训练差异已经得到控制。
不要对时序模型的输入 PKL 直接做 stride 抽样，否则会改变其历史输入。
只分析 Scene 3 时，可以保留该场景完整连续序列构建单独 PKL，并同时覆盖两个配置的
`data.test.ann_file`；匹配诊断也使用同一 PKL。

```bash
python tools/temporal_vis/audit_dynamic_objects.py occupancy \
  --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
  --single-root outputs/audit_single_predictions \
  --knn-root outputs/audit_knn_predictions \
  --matching-csv outputs/dynamic_object_audit/scene3_match/matching_per_object_history.csv \
  --output-dir outputs/dynamic_object_audit/scene3_occupancy
```

输出：

- `occupancy_per_object.csv`：每个目标分别记录各历史框区域，以及历史框并集。
- `occupancy_summary.csv`：按速度、间隔和区域类型汇总 TFPR、目标 Occupancy Recall。
- `order*_object*_occupancy.png`：相同目标局部 GT / 单帧 / KNN 三面板叠加图。
- `skipped_frames.json`：显式启用 `--skip-missing` 后跳过的预测缺失帧；默认缺失即报错。

拖尾区域 = 对齐到当前帧的历史框（或历史框并集）减当前目标框，再限制为当前 GT 为零、
处于水平 FoV、非 ignore 的体素。历史姿态变换包含旋转，不只是中心平移。
TFPR = 该区域预测占用数 / 该区域空体素数。
目标 Recall = 当前框内 GT 占用且被预测占用的体素数 / 当前框内 GT 占用体素数。
统计使用二值占用，不要求预测类别与原始 GT 的类别编号一致。

`region=history_union, frame_gap=0` 是历史框并集，0 不是零时间间隔；该行 delta_t 为空。
该行的 Same-object Recall 对相关历史帧的分子、分母先求和。单历史行按实际 gap 关联。
`matching_pairs_expected/joined` 显式报告关联覆盖；缺任何相关匹配行时关联召回率留空。
`miss_vs_delta_tfpr_pearson` 使用 (1 - 匹配召回率) 与 (KNN TFPR - 单帧 TFPR)，
至少 3 对且两变量均非恒定时计算。这是描述性相关，重复目标、帧和共享体素并非独立样本，
不能据此认定因果关系或统计显著性。汇总体素数是目标区域加权计数，重叠区域可能重复计入，
不是全场景去重统计。历史框并集也不是两帧间连续 swept volume。

### 坐标与 Free 的口径

本仓库原始 GT 稀疏数组为 `[x,y,z,class]`，save_occ 导出预测为 `[z,y,x,class]`；
工具分别读取，绝不自动猜测。默认网格为 LiDAR 坐标的 128×128×14，范围
`[0,-25.6,-2.6,51.2,25.6,3]`。使用不同数据时可指定 `--gt-order`、`--grid-frame`、
`--grid-shape` 和 `--pc-range`。FoV 在转换后的雷达坐标计算。

默认 GT 未列出体素按 benchmark empty 处理，**不声称它们都是明确观测过的 Free**。
如已有有效观测 mask，使用：

```text
--valid-mask-template 'data/valid_masks/{scene}/mask{occ_frame_idx}.npy'
```

mask 必须为同网格的二值 dense XYZ 数组，1 表示可评价；模型预测不应用于生成此 mask。
支持占位符 scene、order、occ_frame_idx、token。GT=255 始终排除；零体素区域的 TFPR 留空。

## 验证

```bash
python -m unittest discover -s tests -p test_dynamic_object_audit.py -v
```

测试覆盖三类有效匹配结果和无历史回波、旋转框与重叠排除、XYZ/ZYX 与越界检查、
实际模型的 split 历史选择、去除自车运动后的速度、TFPR/目标召回/匹配关联与有效 mask。

## 第一阶段方向性方框实验

新增独立 `box` 命令，同时比较等体积的对称框、GT 正确方向框与反方向框；
保持 K=8 和原 Query 欧氏距离排序，不运行球形对照或修改模型。
完整命令、指标和坐标约定见 [README_directional_box_matching.md](README_directional_box_matching.md)。
