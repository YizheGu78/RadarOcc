# P0 稀疏 Query：64 / 32 + Learned Top-8

位置仍是取点之后、Range Attention 之前，项目内部命名 P1，与你说的 P0 相同。
本实验沿用现有 Learned Top-8 的搜索框、冻结 ranker 和融合适配器；只改变当前帧参与时序计算的 Query。

| 每个 range 的时序 Query | 每帧 Query | 相对 175×250 的减少率 |
| ---: | ---: | ---: |
| 64 | 11,200 | 74.4% |
| 32 | 5,600 | 87.2% |

原始单帧输入始终保持 43,750 个 token；所有历史候选完整保留。减少率不是整模型加速比。

## 筛选规则

训练和可视化共用 `temporal/core/query_budget.py`，不使用 GT 决定保留点。

1. 三个 Doppler 峰按 power 加权做循环统计，处理 FFT 速度边界；用实际周期 N×bin 步长。
2. 使用输入序列中最近的有效过去位姿与实际时间差，估计雷达原点自车速度，包含雷达外参的旋转杠杆臂。
3. 计算视线投影自车补偿后的模周期残差。阈值为 0.2 m/s，可靠峰 std≤0.2 m/s、concentration≥0.8。
4. 默认 `both_consensus`：正负两种 Doppler 符号均有非静止证据，才归运动证据类；两种均与静止一致才归静止一致类；不一致、多峰、不可靠或无可用自车速度归不确定类。不利用 GT 选择符号。
5. 每个 range 分配运动证据 / 不确定 / 静止一致配额 50% / 25% / 25%。即 Q64 为 32/16/16，Q32 为 16/8/8。某类不足，将剩余名额依次补给运动证据、不确定、静止一致类。
6. 在各类内把当前 range 的角度覆盖划分成 8×4 个单元，按轮次从有点的单元选代表点，单元内部按最大峰 power 降序。先覆盖各单元，再取第二轮代表点。此处 power 是局部排序，不是全局取最强点。

上述配额是本轮固定起点，尚未证明最优。模周期静止一致不等于真实静止，运动证据不等于前景。固定预算会削减不确定点；无自车速度时全部归不确定类，然后按空间与 power 选满预算。

在线直接用模型当前坐标及描述子重算索引。离线 mask 保存的是原 NPZ 顺序，不能直接按行套入已排序的模型 tensor；可视化会逐帧验证重新按 power 排序后选中的观测一致。

## 第一步：只看可视化

```bash
cd /home/user1/projects/RadarOcc
bash run_temporal_query_budget.sh visualize
```

默认场景 1、2、3、15、22，各 order 80–200、stride=5、25 帧，共 125 帧。
场景 1/2 来自完整 val，3/15/22 来自 test，仅作探索性可视化；后续训练最佳权重只按固定 val 选择，不能把这些 test 图片的覆盖统计当作独立测试结论。

查看 `outputs/query_budget/five_scenes_64_32/index.html`：每帧一张图，上排 Q64、下排 Q32；三列分别是完整点、保留 Query、取消 Query。绿框仅用于 GT 评价。图中展示 X=0–85、Y=±25 m，计数和保存的 mask 使用全部原始点。

- `summary.csv`：各场景和总计的 Query 减少率、GT 点保留率、原非空 GT 框被剪空次数。
- `frame_metrics.csv`：逐帧统计，以及各类别实际保留点数。
- `gt_per_box.csv`：每个 GT 框分别统计原有/保留点数。XY 投影 GT，ROI X[0,85)、Y[-60,60)、Z[-5,8)。原空框不算被剪空。
- `scene*/order*.npz`：`keep_Q64`、`keep_Q32`、`coords_rae`、`xyz`、`state`，均为原 NPZ 顺序。
- `manifest.json`：策略、源码校验值、完整帧清单、是否完成图片。可视化不会训练或更新权重。

如默认目录已有结果，使用新目录：

```bash
QUERY_REVIEW_DIR=outputs/query_budget/review_v2 bash run_temporal_query_budget.sh visualize
```

只跑某一帧验证入口时可传 `--scenes 3 --start-order 80 --end-order 80 --frames-per-scene 1`。正式查看建议使用默认五场景。

## 第二步：确认可视化后训练

脚本默认只执行可视化，不会自动接着训练。你看过并确认以后才执行：

```bash
bash run_temporal_query_budget.sh train --visual-reviewed
```

训练顺序 Q64 → Q32，固定 20% train 和 20% val，seed=42，15 epochs。
默认 RadarOcc 和新增 Attention 从相同随机初始化开始；已有选点器 `work_dirs/p1_top8_ranker/best.pth` 加载并冻结，不重新训练它。
主干和 Occupancy Head 均参与训练。本次没有使用 Z0–Z3 模块。

训练开始前会检查可视化已完成、策略与源码一致；源码/策略变更后应在新目录重新生成可视化。
若需要统一从某个**单帧** checkpoint 初始化两组：

```bash
QUERY_INIT=work_dirs/box_z_p0_20pct_scratch_seed42/Single-frame/best_SSC_mean_epoch_9.pth \
QUERY_WORK_DIR=work_dirs/temporal_query_budget_20pct_single20_init \
bash run_temporal_query_budget.sh train --visual-reviewed
```

不要把从头训练结果和 checkpoint 初始化结果直接混在同一目录中。两组始终使用相同的占用模型初始化与训练预算。

可选环境变量：`QUERY_EPOCHS`、`QUERY_SEED`、`QUERY_RANKER`、`QUERY_WORK_DIR`、`QUERY_REVIEW_DIR`。
已开始的相同实验再次运行会从各自 `latest.pth` 恢复；完整结束的组跳过。代码、输入权重或训练预算改变需新 work-dir。

## 计算压缩与旧缓存

`temporal/core/sparse_topk.py` 对保留的 Query 先生成紧凑数组，再调用原冻结 ranker；历史候选仍完整，每个 Query 每历史帧 pool64 → Top8，三帧最多 24 个 key。
Attention 只对紧凑 Query 循环，融合后可微写回完整当前 tensor。被取消时序的行及 Doppler bin ID 不变。

已有全量 Top-8 索引缓存可选复用：

```bash
TOP8_CACHE_DIR=data/top8_selection bash run_temporal_query_budget.sh train --visual-reviewed
```

此时缓存使用原来的全量输入 key，在命中后取所需 Query 行；缓存必须与权重、设备、PyTorch、输入排序等原有条件一致。严格 read 模式 miss 会报错，不会偷偷退回全量搜索。默认不设此变量，在线只搜索和打分所选 Query。原 `learned_topk.py`、`box_candidates.py` 及缓存 namespace 保持原代码。

## 查看 val

每组目录 `work_dirs/temporal_query_budget_20pct_seed42/Q64` 或 `Q32`：

- `val_metrics.jsonl`：每轮完整精度 IoU，包括全范围、25.6 m、12.8 m 的 SC/SSC。
- `best_SSC_mean_epoch_*.pth`：按固定 val 的 SSC_mean 保存最佳权重；普通 epoch 只留最近两轮。
- `best_val.csv` / `best_val.json`：最佳 checkpoint 对应的完整 val 指标。
- `*.log.json` / `console.log`：训练损失、耗时等。

两组根目录：`epoch_iou.csv`（所有轮）、`comparison.csv` / `.json` / `.md`（各组最佳轮）。
训练过程中可以手工刷新汇总：

```bash
bash run_temporal_query_budget.sh summarize
```

## 验证范围

CPU 测试覆盖预算和配额、Doppler 折叠边界、位姿因果性、排序不变性、稀疏/全量选点一致、实际 ranker/Attention 只处理所选行、原行不变、梯度、冻结 ranker、旧缓存只读复用。
当前工作环境 CUDA 不可用，因此本次未执行整模型 GPU 训练或测量训练加速。覆盖率统计不能代替 Occupancy IoU。
