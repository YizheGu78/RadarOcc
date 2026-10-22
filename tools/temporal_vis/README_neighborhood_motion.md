# 阶段 B Doppler 与邻域运动投票

本阶段只新增离线方向判断与选框验证，不修改模型、训练入口、config、阶段 A 或原 KNN。
先固定 K=8 对照；可再用相同脚本指定 K=64 查看候选池效果。没有实现或训练阶段 C Attention。

## 运行

```bash
cd /home/user1/projects/RadarOcc
conda activate radarocc5060
python tools/temporal_vis/audit_neighborhood_motion.py \
  --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
  --gaps 1 2 3 --box-lengths 4 6 8 \
  --box-reverse 0.5 --box-y-half 1.5 --box-z-half 1.0 --k 8 \
  --doppler-bins data/K-Radar-official-meta/resources/arr_doppler.mat \
  --doppler-sign both \
  --output-dir outputs/dynamic_object_audit/scene3_motion_voting_k8
```

几何表 `--bins` 默认是原实验的 `info_arr.mat`。**速度表为另外的文件**，默认读取
官方 K-Radar 的 `resources/arr_doppler.mat`、键 `arr_doppler`，不按旧传统管线的
固定 min/resolution 猜测。也支持 MAT/NPZ 键 `arrDoppler`、`doppler_mps` 和一维 NPY。
参考官方读取代码：
https://github.com/kaist-avelab/K-Radar/blob/main/datasets/kradar_detection_v2_1.py

`--doppler-sign both` 分别跑 +1/-1 两个假设，分开输出，不汇总、不使用 GT 选择符号。
其中 +1 表示表中正值对应远离雷达，-1 表示表中正值对应靠近雷达。
当前仓库尚无足够证据确认采集及预处理链的符号；应由独立的物理约定或已知静止场景确认后，
指定 `--doppler-sign 1` 或 `--doppler-sign=-1`。不能按本次目标 GT 的得分挑符号后称为无 GT 推理。

## 方向如何产生

1. 保留原 `load_p1` 的前 175×250 个 Token 和原几何、Ego mask，直接读取未经归一化的
   Doppler8：前三通道为峰值功率，3:6 为对应零基 Doppler bin，后两通道不参与速度估计。
   Top-3 不假设已排序；用功率加权的物理速度均值，同时检查加权标准差。
2. 使用当前 split 中上一个真实条目的 pose 和实际 timestamp 求雷达原点的因果差分速度，
   在当前 radar 坐标中表示。对 `pose @ extrinsic` 的原点求差，包含旋转时的杆臂位移。
   这是区间平均速度近似；首帧没有历史 pose 时两种预测方法均回退，不读取未来帧。
3. 逐点计算 `sign * measured_radial + dot(ego_velocity_radar, line_of_sight)`。
   这是直接补偿，不把所有点都加上自车 X 速度，也不把补偿值重新取模。
4. 单点基线：可靠、位于前方且 LOS X 分量足够大的 Token，补偿速度大于阈值则历史方向为 -X，
   小于负阈值则为 +X，其余为不确定。
5. 邻域方案：在**当前帧全部 Token** 中取至多 64 个、2 m 内的其他 Token，排除自身。
   先按空间距离与 log 峰值功率相似度求加权速度中位数，再按与中位数的速度一致性投票。
   同时检查加权 MAD、加入速度权重前后的支持率，避免把混合邻域加权成虚假的一致结果。
   当前 Query 的速度符号不用于筛邻居，允许邻居修正一个反号异常点。
6. 邻域投票、单点预测均只接收 XYZ、原始 Doppler8、速度表、自车速度及阈值。
   不接收 bbox、Object ID、GT、历史关联或 Oracle 方向。先为全部当前 Token 预测，再读取 GT 评价。

默认超参数都是待验证的起始值，未用测试集 GT 调参：

| 参数 | 默认值 |
|---|---:|
| `--vote-radius` | 2 m |
| `--vote-max-neighbors` | 64 |
| `--vote-min-neighbors` | 3（不含自身） |
| `--speed-threshold` | 0.5 m/s |
| `--min-los-cos` | 0.7 |
| `--min-support` | 0.7 |
| `--max-mad` | 1 m/s |
| `--velocity-sigma` | 1 m/s |
| `--power-sigma` | 2（log1p 峰值功率） |
| `--max-peak-std` | 0.5 m/s |

近零、零功率、非有限数据、非法 bin、峰值分歧大、LOS 不可靠、邻居不足或投票混合时回退到对称框。
未引入卡尔曼、速度解混叠、跨帧 KNN 估速、目标跟踪或速度 head。

## 四组配对比较

| 方法 | 方向来源 | 不确定时 |
|---|---|---|
| `symmetric` | 不判断 | 对称框 |
| `oracle_direction` | 阶段 A 的 GT 对齐中心位移符号 | 按阶段 A 共同排除规则 |
| `single_token` | 当前 Token Doppler + ego 速度投影 | 对称框 |
| `neighborhood` | 当前帧邻域加权投票 | 对称框 |

复用阶段 A 原 `BoxSearch`，不改变候选排序。框尺寸不依赖 GT 位移大小或估计速度。
三种历史槽 L=4/6/8 m，反方向余量 0.5 m、Y 半宽 1.5 m、Z 半宽 1 m。
对称回退使用 X=±(L+0.5)/2，体积与方向框相同。所有方法使用相同查询与历史、相同框体积和 K。
按到原 Query 的距离选最近 K 个，同距离按原 P1 相对索引打破平局；不足 K 不补点。

仍使用阶段 A 的共同筛选：同 ID 历史框存在、当前中心位于前方、GT dx 非近零及可选纵向筛选。
这些筛选只定义评价人群，**不会进入方向预测**。因此本结果也不代表对所有静态/横向目标的完整验证。
GT ID 只做同目标归属和 Oracle 参考。没有同目标历史 P1 的查询记入 `n_no_history`，
不进入候选 Recall 分母；若有配对历史框，仍可评价方向。这两个分母分别输出。
与阶段 A 比较时，务必使用相同 ann/bins/extrinsic/history-mode/选择范围及筛选参数。

## 输出与读法

| 文件 | 内容 |
|---|---|
| `motion_results.md` | 每个符号假设 × 历史槽 × 四种方法的对照表 |
| `motion_summary_by_history.csv` | micro 汇总、原始分子分母、Coverage/Recall@K/Precision |
| `motion_per_object_history.csv` | 逐目标、历史、方法明细，便于目标级检查 |
| `motion_excluded_pairs.csv` | 四组共同排除的对象历史对及原因 |
| `motion_frame_diagnostics.csv` | 逐帧 ego 速度、有效 Token、投票回退原因及混叠风险提示 |
| `motion_protocol.json` | 完整 CLI 参数、实际帧列表、物理速度表及源代码/表文件哈希 |
| `motion_token_details.csv` | 可选 `--save-token-details`，流式记录每个当前 Token 的预测与速度统计，不含 GT |

方向正确率分两种：`direction_accuracy_decided` 是正确 /（正确+反向）；
`direction_correct_fraction` 是正确 / 全部可评价方向查询。
`direction_reversal_fraction` 与 `direction_uncertain_fraction` 均除以全部可评价方向查询，
且正确+反向+不确定=全部。不能单独看有选择性的正确率而忽略大量回退。
Recall 仍表示 Top-K 至少包含一个同目标历史 Token 的查询比例，不是 Occupancy IoU。
符号假设从不相加；`symmetric` / Oracle 会在两种假设下重复，分别作为配对参考。
同一个查询出现在不同历史槽时会重复评价，不是独立样本。

## 物理边界

Doppler 补偿后仍是径向分量。LOS 门限只限制观测几何，无法凭一个径向速度识别真实横向运动。
区间平均 ego 速度也不等于雷达精确采样时刻的瞬时速度。

**本阶段没有消除速度混叠。** 单个相干速度峰也可能是混叠结果，Top-3 一致不保证真实方向可靠。
`n_static_prediction_outside_doppler_table` 统计按 ego 投影预测的静止回波速度是否已超出表范围，
只作为混叠风险提示，不能当作真值或真实混叠率。没有静止区域标签来自动“纠正”速度。
若诊断显示大量该风险或两符号结果不稳定，应先检查物理表、速度符号和混叠，
不能把输出数值称为真实三维车速，也不能据此断言阶段 B 已成功。

## 验证及下一阶段

```bash
python -m unittest discover -s tests -p test_neighborhood_motion.py -v
python -m unittest discover -s tests -p test_directional_box_matching.py -v
python -m unittest discover -s tests -p test_dynamic_object_audit.py -v
```

先检查 K=8 是否改善对称框，再查看方向反转率与不确定率。验证有效后可将上方命令
改为 `--k 64` 并换一个输出目录；它报告的是 Recall@64，不代表 Attention 重排序后的 Recall@8。
本阶段不向 RadarOcc 主流程接入新推理逻辑。
