# Autoware + GM2019 传统基线 + Random Forest

独立目录 `RadarOcc/tradition_real`，与 `tradition` 同级。运行不依赖旧管线，
旧目录、旧脚本、旧模型不需要更改。**新 RF 必须重新训练，不能复用旧 42D checkpoint。**

流程：RPC → 位姿对齐 → Autoware CPU 二维射线证据 → GM2019 第 3.1 节
log-odds 融合 → Occupied 体素 DBSCAN → 42D 特征 → RF → Free / BG / FG。

## 来源边界：哪些是复刻，哪些是适配

| 部分 | 本地代码 | 依据与范围 |
|---|---|---|
| 坐标转换、Free 射线、端点 Occupied | `autoware/costmap.py` | fork 中的 CPU `OccupancyGridMap::worldToMap/raytraceFreespace/raytrace2D/updateCellsByPointCloud` 等价翻译 |
| 地图原点移动 | `autoware/costmap.py` | `OccupancyGridMapInterface::updateOrigin` CPU 逻辑，保留首次清空与按格移动 |
| Bresenham | `autoware/costmap.py` | Autoware 继承的 Nav2 humble `raytraceLine/bresenham2D` |
| 可选 BBF 更新 | `autoware/costmap.py` | `OccupancyGridMapBBFUpdater::applyBBF`，保留 float32、lround、1–254 限幅与官方默认参数 |
| 默认 GM2019 更新 | `gm2019/fusion.py` | 第 3.1 节公式 (3)–(4)，Unknown 不更新；论文基线复现，没有官方 GM 代码 |
| 14 层高度适配与坐标对齐 | `adapters/layers.py`、`pipeline.py` | 本项目适配，保持上面的二维核心不变 |
| DBSCAN、特征、RF | `semantics/` | 本项目分类层，静止不等于背景、运动不等于前景 |
| 数据与评估 | `adapters/dataset.py`、`evaluation/` | K-Radar / RadarOcc 的索引、坐标和指标接口 |

Autoware 固定提交：`1c93b65555410c8c7b8299dab2f48320d1cb19e4`。
原始 C++/头文件与文件哈希随代码保存在 `vendor/` 和 `SOURCE_MANIFEST.json`。

**移植的是上述 CPU 子集，不是整个 Autoware，也不是其 CUDA
FixedBlindSpot / Projective 双点云节点。** RPC 已是检测点列表，本版没有引入
LiDAR ground filter、额外 CFAR、可靠性筛点或障碍物膨胀。射线终点裁剪、
处理顺序（全部 Free 完成后再标记全部命中点）、半开边界等按 C++ 保留。

GM2019 论文主要提出神经网络 Occupancy Net。本目录实现的是该论文讲述的
**传统对比基线**，不包含 Occupancy Net，也不能声称复现了其网络指标。
论文没有给出足够参数去唯一重现其传统 baseline；这里的 `p_hit=.7`、
`p_free=.35`、`prior=.5`、阈值和时间窗口均明确作为项目配置。
参考：[GM2019 论文](https://arxiv.org/abs/1904.00415)。

## 二维核心如何输出 RadarOcc 三维结果

- RadarOcc Small 栅格固定为 `[128,128,14]`，0.4 m，范围
  `[0,-25.6,-2.6]` 至 `[51.2,25.6,3.0]`，数组顺序 `[X,Y,Z]`。
- 输入先转为当前 LiDAR 坐标。每个回波按端点高度进入一个层，该层独立调用
  Autoware 的二维射线算法；这是 **2.5D 分层模型，不是真正的三维射线模型**。
  所以不能把它称为 Autoware 原生 3D occupancy，也不会把 BEV 障碍整列填满高度。
- 核心接收以格为单位的 XY 坐标（分辨率 1.0）。物理米/0.4 的转换在适配层，
  原点临时扩展到包含历史传感器位置，射线完成后裁剪回评估区域。
- 同时计算原生二维 BEV 概率，保存为 `bev_probability`，便于单独检查二维结果。
- 默认按 5 帧因果窗口，从 prior 重放每帧一次。历史传感器原点和点都随 pose
  变换，不把整个滑窗重复累加到上一轮 posterior。切换场景或缺帧会重置；
  同一场景重复/倒序帧直接报错。
- 本版没有动态跟踪或速度解混叠。RPC 第 4 列（零基）物理 Doppler 原样作为
  RF 特征，未做 Kalman、整数 k 或 Doppler 解缠。运动物体可能在时间窗口内拖影；
  `temporal_window=1` 可作为单帧对照，但需要按对应配置重新训练 RF。

内部始终区分 Free、Occupied、Unknown。RF 只改变 Occupied 的 BG/FG 标签。
本版不使用 RF 擦除占据点，也不把 DBSCAN 噪声点直接丢弃，而是保留为单体素提案。

## 输入约定

- RPC：`[N,11]` NPY，列为
  `x,y,z,power,doppler,range,azimuth,elevation,range_index,azimuth_index,elevation_index`。
  Doppler 已经是 m/s，不是需要再次换算的 bin 索引。
- RPC/RGB/annotation 按各自场景内顺序匹配，不要求文件号相等。RPC 数量和该场景
  annotation 数量不一致时立即报错，避免跳过样本后仍冒充完整测试集。
  ordinals 在 `--scenes` / `--max-frames` 筛选前建立。
- Pose：沿用 annotation 的 `lidar_token` 定位
  `data/K-RadarOcc/{train,val,test}/SCENE/pose/lidar_ego_poseN.npy`；应为 LiDAR-to-world。
- Calibration：读取 `data/K-Radar_calib/SCENE/info_calib/calib_radar_lidar.txt`，
  例如 `30,-2.54,0.3`，默认 RPC-to-LiDAR 平移为 `(+2.54,-0.30,-0.70)`，
  Z 可用 `--radar-z` 设置。文件中的 frame difference 记入日志，不再叠加到已经顺序配对的编号上。
  **假设无额外旋转外参**，与当前项目约定一致；不自动猜测坐标方向。
- GT：通过 PKL 的 `occ_path` 读取；`--gt-order xyz` 默认值沿用现有评估。
  GT 255 忽略，GT 1 为 BG，其他非零语义类别归 FG。GT 只供训练标签和评估使用，
  不参与推理的 occupancy、聚类或特征计算。

## 三阶段运行（推荐：先缓存，再训练 / 评估）

在仓库根目录激活 `radarocc5060` 等 Python 环境。仅依赖 numpy、scikit-learn、
joblib；视频另需 OpenCV，不需要 ROS、CUDA、torch 或 mmcv。

```bash
python -m pip install -r tradition_real/requirements.txt

# 第一次：两个 split 各做一次 RPC → Autoware → GM2019 → DBSCAN → 42D
bash run_tradition_real.sh preprocess

# 后续只从缓存读取特征，并用 GT 重新生成提案 BG/FG 标签
bash run_tradition_real.sh train

# 完整 test_official：缓存 + RF → 三类标签 → 15 个 IoU 指标
bash run_tradition_real.sh evaluate
```

也可第一次 `bash run_tradition_real.sh all` 连续执行三阶段。
已有缓存时直接运行 train/evaluate；preprocess **拒绝覆盖已有缓存**，不会静默跳过或混用。
改变预处理配置时指定新的 `CACHE_ROOT` 重建两个 split，例如：

```bash
CACHE_ROOT=data/frame_fusion_w1 TEMPORAL_WINDOW=1 bash run_tradition_real.sh preprocess
CACHE_ROOT=data/frame_fusion_w1 RUN_DIR=work_dirs/tradition_real/w1 bash run_tradition_real.sh train
CACHE_ROOT=data/frame_fusion_w1 RUN_DIR=work_dirs/tradition_real/w1 bash run_tradition_real.sh evaluate
```

单独生成一个 split：`bash run_tradition_real.sh preprocess-train` 或 `preprocess-test`。
完整路径与参数见 `bash run_tradition_real.sh help`。脚本使用当前激活的 Python，
可用 `PYTHON_BIN=/path/to/python` 指定。`RADAR_ROOT`、`POSE_ROOT`、`CALIB_ROOT`、
`TRAIN_ANNOTATION`、`TEST_ANNOTATION` 可覆盖默认路径；含空格的路径请加引号。

### 对应的 Python 命令

```bash
python -m tradition_real preprocess \
  --annotation data/annotations/kradar_dict_train_official_doppler8.pkl \
  --radar-root data/K-Radar_rpc --pose-root data/K-RadarOcc \
  --calib-root data/K-Radar_calib --output data/frame_fusion/train_official

python -m tradition_real preprocess \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --radar-root data/K-Radar_rpc --pose-root data/K-RadarOcc \
  --calib-root data/K-Radar_calib --output data/frame_fusion/test_official

python -m tradition_real train \
  --annotation data/annotations/kradar_dict_train_official_doppler8.pkl \
  --frame-fusion-root data/frame_fusion/train_official \
  --output work_dirs/tradition_real/rf_200 --n-estimators 200

python -m tradition_real evaluate \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --frame-fusion-root data/frame_fusion/test_official \
  --model work_dirs/tradition_real/rf_200/random_forest.joblib \
  --output work_dirs/tradition_real/rf_200_test --save-predictions
```

缓存模式只需要 PKL、缓存、GT，视频额外需要 RGB；RPC、pose、calibration 原文件
可以离线。不会调用 `Pipeline.map_frame()`、Autoware、GM2019、DBSCAN 或特征提取。
不传 `--frame-fusion-root` 仍可使用原来的在线 train/evaluate 入口进行对照。

### 缓存结构和有效性

```text
data/frame_fusion/
├── train_official/
│   ├── manifest.json
│   ├── pipeline_config.json / frames.json / run.json
│   └── SCENE/frame_TOKEN.npz
└── test_official/
    ├── manifest.json
    └── SCENE/frame_TOKEN.npz
```

manifest 放在各 split **内部**，移动整个 split 时不需要移动另一个伴随文件。

| NPZ 字段 | 含义 |
|---|---|
| `native_labels_xyz` | `[128,128,14]`，0 Free / 1 Occupied / 255 Unknown，尚未做 RF 分类 |
| `occupancy_probability` / `bev_probability` | 原精度融合概率，3D / BEV |
| `observed_mask` | 是否收到过非 Unknown 证据；不等同于最终已分类区域 |
| `proposal_features` | `[K,42]`，保留 float64，字段顺序绑定 `FEATURE_NAMES` |
| `proposal_voxels` | `[M,3]` XYZ 索引，按 proposal 顺序拼接 |
| `proposal_offsets` | `[K+1]`，第 i 个 proposal 为 `voxels[offsets[i]:offsets[i+1]]` |
| `scene` / `token` / `ordinal` | 帧身份与原始场景内顺序序号 |
| `format` / `signature` | 格式版本、预处理配置、代码哈希与特征定义 |

**不缓存 GT 或 BG/FG training target**。preprocess 不读取 GT 文件。
空提案帧保留 `[0,42]` 特征、`[0,3]` 体素、`[0]` offsets，不跳帧。
每帧 NPZ 和 manifest 使用临时文件后原子替换；中断时缓存仍为 running/failed，
训练拒绝使用。当前不支持断点续跑；中断后使用新的输出目录完整重跑。

manifest 绑定完整预处理配置（窗口、hit/free/prior、阈值、栅格、DBSCAN）、
`radar_z`、特征字段和预处理源代码哈希，并记录 PKL SHA256、每个 RPC/pose 的
SHA256、标定文件 SHA256 / 平移、每帧 NPZ SHA256 和帧清单。
训练/评估会拒绝配置不匹配、错误 split、缺帧、重复帧、错误 ordinal、损坏 NPZ
和未完成缓存，不会回退到现场重新建图。模型也记录预处理签名。

默认信任缓存内的标定快照，不访问原始输入。若要检查本机更新后的标定，显式传
`--calib-root data/K-Radar_calib`，文件内容变化会报错。RPC/pose 原文件如果后来改变，
须主动重新预处理；缓存消费不会扫描原数据去检测修改。代码更新或更换预处理参数后
使用新缓存目录；同一套 RF 的训练缓存和测试缓存必须采用相同预处理签名。

`--scenes` / `--max-frames` 可做小样本检查，缓存只包含选中帧。
完整训练/评估若选中了未缓存的帧会立即报错。全量指标请勿设置这两个限制。

### 重复调 RF，无需重建缓存

```bash
N_ESTIMATORS=500 bash run_tradition_real.sh train
N_ESTIMATORS=500 bash run_tradition_real.sh evaluate

# 同树数的其他实验建议用独立 RUN_DIR
MAX_DEPTH=12 CLASS_WEIGHT=balanced POSITIVE_FRACTION=0.3 \
  RUN_DIR=work_dirs/tradition_real/rf_depth12 bash run_tradition_real.sh train
RUN_DIR=work_dirs/tradition_real/rf_depth12 bash run_tradition_real.sh evaluate
```

可调整 `--n-estimators`、`--max-depth`（0 不限制）、`--min-samples-leaf`、
`--class-weight balanced_subsample|balanced|none`、`--n-jobs`、`--seed`。
训练输出 `random_forest.joblib`、`training_features.npz`、`pipeline_config.json`、
`frames.json`、`run.json`。OOB 是训练内部诊断，不是测试 IoU。

`positive_fraction=0.20` / `negative_fraction=0.05` 是 **GT 提案打标签阈值，
不是随机采样比例**。提案内有效 GT 的 FG 比例 ≥ positive 标前景；
FG 比例 ≤ negative 且 BG 比例 ≥ positive 标背景；其他忽略。
GT 255 不参与比例；Free 为主的提案不会自动作为 BG。
每次训练都按当次 GT 和阈值重新算标签，改这两个参数无需重新预处理。

## 完整测试集 + 场景 3 全帧视频

```bash
CAMERA_DIR=data/K-Radar/3/cam-front bash run_tradition_real.sh evaluate

# 只限制视频为 80–200 帧，指标仍覆盖完整 test_official
CAMERA_DIR=data/K-Radar/3/cam-front VIDEO_START=80 VIDEO_END=200 \
  TEST_OUTPUT=work_dirs/tradition_real/rf_200_test_clip bash run_tradition_real.sh evaluate
```

把 `CAMERA_DIR` 改为本机真实场景 3 RGB 文件夹。不设置它就只评估和保存预测。
默认场景 3 全帧视频。不要加 `--scenes 3` 来做完整测试：`--video-scene 3`
只控制视频。视频范围两端包含，指从 0 开始的场景内序号，不是文件号或秒。
`SAVE_PREDICTIONS=0` 可关闭预测 NPZ；默认保留。

输出：

| 文件 | 内容 |
|---|---|
| `metrics.json` / `metrics.csv` | 与旧评估相同的 15 个 `SC*` / `SSC*` 指标 |
| `confusions.npz` | 51.2 / 25.6 / 12.8 m 全数据混淆矩阵 |
| `predictions/SCENE/TOKEN.npz` | 三维标签、原生标签、占据概率、observed/unknown 掩码、BEV 概率 |
| `scene_3.mp4` | Prediction / GT / RGB，Free 蓝、BG 灰、FG 红、Unknown 深灰 |
| `frames.json` | 每帧真实 RPC / GT / pose 路径、校准、占据数和提案数 |
| `run.json` | 配置、选中/处理帧数、成功或失败状态、数据哈希 |

默认导出把 Unknown 映射为 Free，以满足 RadarOcc 三类接口。
**这仅是评估导出策略，不表示雷达观测证明它是空闲。** `native_labels_xyz` 和
`unknown_mask` 保留 Unknown；评估会把这些体素正常计入错误，不会通过忽略预测 Unknown
来虚高 IoU。另一个显式策略为 `unknown_export="background"`，切换后应注明比较条件。
`SSC_mean` 是 BG/FG IoU 均值；SSC_free 单列报告。无定义的 IoU 写 JSON null。

## 配置与可选 Autoware BBF 对照

预处理可用 `--config path/to/config.json`；未写字段使用默认值。缓存训练默认读取
manifest 配置，评估默认读取 RF 模型配置；显式 JSON/窗口参数覆盖后仍须通过缓存/模型
一致性校验。RF 阈值和 Unknown 导出策略不影响预处理缓存，但评估仍须匹配模型配置。配置示例：

```json
{
  "temporal_window": 5,
  "fusion": "gm2019",
  "p_hit": 0.7,
  "p_free": 0.35,
  "occupied_threshold": 0.55,
  "free_threshold": 0.45,
  "dbscan_eps_xy": 1.2,
  "dbscan_eps_z": 0.8,
  "dbscan_min_samples": 2,
  "unknown_export": "free"
}
```

`fusion="autoware_bbf"` 改用官方 BBF 核心作为消融对照，需要另训 RF。
GM2019 与 BBF 是替代关系，同一结果不会顺次做两遍概率融合。
本版 RF 描述符是全新 occupancy-first 42D；模型保存格式、字段顺序和完整配置，
旧 RF 即使维度也是 42 也会被拒绝。训练/测试帧交叉时也会拒绝，避免数据泄漏。

## 验证

```bash
python -m unittest discover -s tradition_real/tests -v
```

- C++ 对照：从随代码保留的原始方法直接提取/编译，仅替换 ROS 消息与日志外壳。
  80 组随机与边界点云逐格对比，另穷举 3 种观测 ×256 种前值的 BBF 字节输出。
  g++ 不存在时此组会标记 skipped。
- 数学/管线：GM2019 公式、Unknown 不更新、因果窗口不重复计数、位姿对齐、
  场景/缺帧重置、高度层隔离、RF 只能修改语义标签。
- 集成：创建不同编号的合成 RPC/RGB，执行在线与缓存两条路径；移除原始 RPC/pose/标定后，
  禁止调用 map_frame/DBSCAN，仍可缓存训练和评估。逐项对比训练特征、标签、全部预测和指标；
  检查 4 帧视频、RF 调参、缺失/损坏/未完成缓存、配置/标定冲突和数据泄漏拒绝。

这些验证不代替真实数据训练/评估。当前没有在完整 K-Radar 上运行，
不能据此承诺真实 IoU、全数据运行速度或与 RadarOcc 相当的精度。
