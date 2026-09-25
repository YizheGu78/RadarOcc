# OctoMap 三维 Occupancy + Random Forest

`tradition_real` 使用 **Python 调用固定版本的原版 C++ OctoMap OcTree**，默认在 CPU 上运行。
纯 Python 移植保留为一致性验证用的参考后端；本次仅替换执行后端，不改变算法和数据流程。
后面的 DBSCAN、42D 特征定义、GT 提案标签规则、RF 分类器和 IoU 接口保持原有逻辑。
旧 `tradition/` 独立保留。**Autoware/GM2019 缓存和 RF 不兼容；已完成的 OctoMap v2
缓存和对应 RF 则继续兼容，不需要因为切换 C++ 重新生成或训练。**

## 当前完整流程

1. **一次性预处理**：读取 RPC、标定、pose → 对齐到当前 LiDAR 坐标 →
   逐帧调用 OctoMap 三维射线和 OcTree 占据更新 → 导出 0.4 m 的
   Free / Occupied / Unknown → Occupied 体素 DBSCAN → 42D 特征 → 保存 NPZ。
2. **反复训练**：读取缓存的 proposal voxels + 42D → 从 GT 计算 BG/FG 标签 →
   `RandomForest.fit()` → 保存模型。不再运行 OctoMap、DBSCAN 或特征提取。
3. **测试 / 验证**：读取对应 split 缓存 → RF 给 Occupied proposal 分类 →
   与缓存 Free / Unknown 组合 → 三类导出、IoU 和可选视频。验证集同样只消费缓存。

GT 不参与 occupancy、聚类或特征生成；不把训练 target 写入 frame_fusion。
RF 只改变 Occupied 的 BG/FG 标签，不擦除占据点。DBSCAN 噪声保留为单体素提案。

## 原版来源与移植边界

固定来源：[YizheGu78/octomap](https://github.com/YizheGu78/octomap/tree/21a8871d7bbd0aa13c73bbdb4e128821d8279af4)，
提交 `21a8871d7bbd0aa13c73bbdb4e128821d8279af4`，OctoMap 1.10.0。
`vendor/octomap/` 保留用于编译对照的原始头文件、C++ 源文件和 BSD 许可证；
`OCTOMAP_SOURCE_MANIFEST.json` 记录每个文件的上游路径与 SHA256。

| 部分 | 本地代码 | 保留的原版行为 |
|---|---|---|
| 三维射线 | `vendor/octomap/`，通过 `octomap/bindings.cpp` 调用；Python 参考为 `octomap/octree.py` | `computeRayKeys` 三维 DDA、边/角相等时的分支顺序、终点不计入 Free |
| 点云更新 | 同上 | `insertPointCloud` / `computeUpdate`；每帧 key 去重，Occupied 优先于 Free，先 miss 后 hit；maxrange、BBX、discretize 分支 |
| 概率更新 | 同上 | float32 log-odds、hit/miss 增量、上下限 clamping、饱和提前返回、`>=` 占据阈值 |
| 真正的八叉树 | 同上 | 深度 16、稀疏子节点、父节点 MAX、8 个等值叶节点剪枝、再次更新时展开并继承父节点概率、lazy inner 更新 |
| 坐标与稠密导出 | `adapters/octomap_grid.py` | 项目适配：平移栅格原点；展开剪枝叶节点覆盖的所有输出体素；无节点即 Unknown |
| 时序窗口 | `pipeline.py` | 项目适配：默认 5 帧因果重放；不是 OctoMap 自带的窗口机制 |
| DBSCAN / 42D / RF | `semantics/` | 原有算法与特征字段顺序；点到体素的索引精度与 OctoMap 对齐 |

Python 参考实现移植的是上述 **OcTree occupancy 核心及其依赖方法**，不是整个库。
C++ 后端通过 pybind11 直接调用相同固定版本的原始代码，不重写射线或更新公式。
绑定只暴露项目需要的接口；没有引入 ROS、GPU、全局持久地图、并行扫描插入或新过滤规则。
坐标对齐、5 帧重放、稠密网格导出、DBSCAN、42D 和 RF 仍由原有 Python 代码执行。
旧 `autoware/`、`gm2019/`、
`SOURCE_MANIFEST.json` 和相关 vendor/test 保留作历史来源记录，当前管线不会调用它们，
也不再提供 Autoware/GM2019 backend 配置。

## 从 OctoMap 到 RadarOcc 栅格

- 输出 `[128,128,14]`，顺序 `[X,Y,Z]`，分辨率 0.4 m，范围
  `[0,-25.6,-2.6]` 到 `[51.2,25.6,3.0]`（上界不含）。
- RPC 加 radar-to-LiDAR 平移，再通过 pose 对齐历史点和历史传感器原点。
  为对齐包含 `min_z=-2.6` 在内的网格边界，送入 OctoMap 前统一减去 `min_xyz`；
  这是坐标平移，不修改射线或概率核心。核心坐标按原版 `point3d` 使用 float32。
- 从真实传感器原点向回波做 **三维射线**；斜射线可以穿越多个高度层。
  不先剔除 ROI 外的回波，它们仍可为 ROI 内提供 Free 证据；导出时才裁剪 ROI。
  没有新增 CFAR、可靠性筛点、障碍膨胀或雷达 Doppler 解缠。
- 每个目标帧建立新树，在当前参考系内把窗口中的每个原始扫描按时间顺序插入一次。
  不重复累加重叠窗口。场景改变或缺帧重置历史，重复/倒序 ordinal 报错。
  保留项目的 5 帧局部方案，不是跨整个场景永久累计的全局地图。
- Unknown 的依据是查询不到节点，而不是概率等于 0.5。已有节点若 log-odds=0，
  在默认阈值下属于 Occupied（原版使用 `>=`）。已观测但低于阈值的节点是 Free。
  没有旧 GM2019 的双阈值灰区。内部概率始终以原版 log-odds 更新并限幅。
- `bev_probability` 只是沿 Z 对已观测概率取最大值的诊断投影；整列 Unknown 为 0.5。
  不再执行二维 BEV 概率融合。
- 滑窗可能导致运动物体拖影。`temporal_window=1` 可作单帧对照，须重建对应缓存和 RF。

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

在仓库根目录激活 `radarocc5060` 等 Python 环境。运行依赖 numpy、scikit-learn、
joblib；视频另需 OpenCV。C++ 后端额外需要编译器、Python 开发头文件和 pybind11，
不需要 ROS、CUDA、torch、mmcv 或系统安装的 OctoMap。

```bash
python -m pip install -r tradition_real/requirements.txt
python -m pip install pybind11

# 编译一次：务必使用运行预处理的同一个 Python/conda 环境
bash run_tradition_real.sh build-octomap

# 第一次：两个 split 各做一次 RPC → OctoMap → DBSCAN → 42D
bash run_tradition_real.sh preprocess

# 后续只从缓存读取特征，并用 GT 重新生成提案 BG/FG 标签
bash run_tradition_real.sh train

# 完整 test_official：缓存 + RF → 三类标签 → 15 个 IoU 指标
bash run_tradition_real.sh evaluate
```

也可第一次 `bash run_tradition_real.sh all` 连续执行三阶段。
`all` 不安装依赖或自动编译，先完成上面的构建命令。
已有缓存时直接运行 train/evaluate；preprocess **拒绝覆盖已有缓存**，不会静默跳过或混用。
改变预处理配置时指定新的 `CACHE_ROOT` 重建两个 split，例如：

```bash
CACHE_ROOT=data/frame_fusion_octomap_w1 TEMPORAL_WINDOW=1 bash run_tradition_real.sh preprocess
CACHE_ROOT=data/frame_fusion_octomap_w1 RUN_DIR=work_dirs/tradition_real/w1 bash run_tradition_real.sh train
CACHE_ROOT=data/frame_fusion_octomap_w1 RUN_DIR=work_dirs/tradition_real/w1 bash run_tradition_real.sh evaluate
```

单独生成一个 split：`bash run_tradition_real.sh preprocess-train` 或 `preprocess-test`。
完整路径与参数见 `bash run_tradition_real.sh help`。脚本使用当前激活的 Python，
可用 `PYTHON_BIN=/path/to/python` 指定。`RADAR_ROOT`、`POSE_ROOT`、`CALIB_ROOT`、
`TRAIN_ANNOTATION`、`TEST_ANNOTATION` 可覆盖默认路径；含空格的路径请加引号。

### C++ 后端安装与确认

`build-octomap` 编译仓库内原版源文件，只在 `tradition_real/octomap/` 生成当前 Python
ABI 对应的 `_native*.so`，不改系统库、不需要 sudo、不修改仓库根目录的安装配置。
Linux/Ubuntu 与 macOS 提供构建入口；Windows 请使用 WSL。
Ubuntu 缺少编译器时可由有权限的管理员安装 `build-essential`；若缺 `Python.h`，
系统 Python 安装对应版本开发头文件，conda 则确认正在使用正确环境的 Python。
可用 `CXX=/path/to/g++` 指定编译器，`PYTHON_BIN=/path/to/python` 指定 Python。

```bash
# 查看实际后端、固定源版本及二进制指纹
python -c 'from tradition_real.octomap.backend import execution_info; print(execution_info())'

# 仅供对照：显式切换回原来的纯 Python 执行器
TRADITION_REAL_OCTOMAP_BACKEND=python bash run_tradition_real.sh preprocess-train
```

默认 `TRADITION_REAL_OCTOMAP_BACKEND=cpp`。没有编译、ABI 不匹配、源码变化但二进制
未重建时都会明确报错，**不会静默退回 Python**。切换 Python 环境或更新绑定代码后重新构建。
原始 vendor 文件有 SHA256 校验；编译不启用 fast-math、OpenMP 或改写算法的优化。
原始输入模式启动时打印 `OctoMap backend: cpp (CPU)`。

已完成的 OctoMap 缓存可直接 `train` / `evaluate` / `validate`，这些步骤
**不加载 C++ 模块，也不需要编译器或 pybind11**。不要为了换后端删除已完成缓存。
未完成缓存仍按原有规则拒绝消费和覆盖；本次不增加断点续跑，改用新输出目录完整预处理。

### 对应的 Python 命令

```bash
python -m tradition_real preprocess \
  --annotation data/annotations/kradar_dict_train_official_doppler8.pkl \
  --radar-root data/K-Radar_rpc --pose-root data/K-RadarOcc \
  --calib-root data/K-Radar_calib --output data/frame_fusion_octomap/train_official

python -m tradition_real preprocess \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --radar-root data/K-Radar_rpc --pose-root data/K-RadarOcc \
  --calib-root data/K-Radar_calib --output data/frame_fusion_octomap/test_official

python -m tradition_real train \
  --annotation data/annotations/kradar_dict_train_official_doppler8.pkl \
  --frame-fusion-root data/frame_fusion_octomap/train_official \
  --output work_dirs/tradition_real/octomap_rf_200 --n-estimators 200

python -m tradition_real evaluate \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --frame-fusion-root data/frame_fusion_octomap/test_official \
  --model work_dirs/tradition_real/octomap_rf_200/random_forest.joblib \
  --output work_dirs/tradition_real/octomap_rf_200_test --save-predictions
```

缓存模式只需要 PKL、缓存、GT，视频额外需要 RGB；RPC、pose、calibration 原文件
可以离线。不会调用 `Pipeline.map_frame()`、OctoMap、DBSCAN 或特征提取。
不传 `--frame-fusion-root` 仍可使用原来的在线 train/evaluate 入口进行对照。

### 缓存结构和有效性

```text
data/frame_fusion_octomap/
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
| `occupancy_probability` / `bev_probability` | OctoMap 概率 / 沿 Z 最大已观测概率投影 |
| `observed_mask` | 该体素是否对应已存在的 OcTree 叶节点（含剪枝父节点） |
| `proposal_features` | `[K,42]`，保留 float64，字段顺序绑定 `FEATURE_NAMES` |
| `proposal_voxels` | `[M,3]` XYZ 索引，按 proposal 顺序拼接 |
| `proposal_offsets` | `[K+1]`，第 i 个 proposal 为 `voxels[offsets[i]:offsets[i+1]]` |
| `scene` / `token` / `ordinal` | 帧身份与原始场景内顺序序号 |
| `format` / `signature` | 格式版本、预处理配置、代码哈希与特征定义 |

**不缓存 GT 或 BG/FG training target**。preprocess 不读取 GT 文件。
空提案帧保留 `[0,42]` 特征、`[0,3]` 体素、`[0]` offsets，不跳帧。
每帧 NPZ 和 manifest 使用临时文件后原子替换；中断时缓存仍为 running/failed，
训练拒绝使用。当前不支持断点续跑；中断后使用新的输出目录完整重跑。

manifest 绑定完整预处理配置（窗口、hit/miss、clamping、占据阈值、max_range、lazy_eval、discretize、栅格、DBSCAN）、
`radar_z`、特征字段和预处理源代码哈希，并记录 PKL SHA256、每个 RPC/pose 的
SHA256、标定文件 SHA256 / 平移、每帧 NPZ SHA256 和帧清单。
训练/评估会拒绝配置不匹配、错误 split、缺帧、重复帧、错误 ordinal、损坏 NPZ
和未完成缓存，不会回退到现场重新建图。模型也记录预处理签名。

默认信任缓存内的标定快照，不访问原始输入。若要检查本机更新后的标定，显式传
`--calib-root data/K-Radar_calib`，文件内容变化会报错。RPC/pose 原文件如果后来改变，
须主动重新预处理；缓存消费不会扫描原数据去检测修改。算法源文件更新或更换预处理参数后
使用新缓存目录；同一套 RF 的训练缓存和测试缓存必须采用相同预处理签名。

本次 C++ 后端替换没有修改签名绑定的算法文件，也没有改变缓存/模型格式。
后端属于执行选项，不是算法配置：相同输入与配置的 Python/C++ 缓存可互用。
`run.json` 的 `octomap_execution` 和新缓存 manifest 的 `execution` 单独记录后端、
固定上游版本、绑定/源码指纹及二进制 SHA256；缓存消费记录 `octomap_executed=false`。
绑定加载时检查当前源代码与编译时指纹一致，拒绝过期二进制。

`--scenes` / `--max-frames` 可做小样本检查，缓存只包含选中帧。
完整训练/评估若选中了未缓存的帧会立即报错。全量指标请勿设置这两个限制。

### 重复调 RF，无需重建缓存

```bash
N_ESTIMATORS=500 bash run_tradition_real.sh train
N_ESTIMATORS=500 bash run_tradition_real.sh evaluate

# 同树数的其他实验建议用独立 RUN_DIR
MAX_DEPTH=12 CLASS_WEIGHT=balanced POSITIVE_FRACTION=0.3 \
  RUN_DIR=work_dirs/tradition_real/octomap_rf_depth12 bash run_tradition_real.sh train
RUN_DIR=work_dirs/tradition_real/octomap_rf_depth12 bash run_tradition_real.sh evaluate
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
bash run_tradition_real.sh evaluate

# 只限制视频为 80–200 帧，指标仍覆盖完整 test_official
VIDEO_START=80 VIDEO_END=200 \
  TEST_OUTPUT=work_dirs/tradition_real/octomap_rf_200_test_clip bash run_tradition_real.sh evaluate
```

`evaluate` 默认使用场景 3 的 RGB 目录 `/home/user1/projects/RadarOcc/data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched`，可通过 `CAMERA_DIR` 覆盖；`validate` 只有明确设置 `CAMERA_DIR` 才会生成视频。
旧视频渲染器逐帧用 Mayavi 绘制 3D voxel，并用 ffmpeg 生成 MP4 和 GIF。运行视频前需要 `radarocc-vis` 中的 Mayavi、Pillow，以及 `ffmpeg` 和图形显示环境（远程无显示时用 `xvfb-run`）。
默认蓝色背景来自 `work_dirs/radarocc_small_fp32_idfix_timealign_v2/visualization_epoch4_test` 中的 RadarOcc Epoch4 `pred_c.npy`；可设置 `VIDEO_BACKGROUND_PREDICTION_ROOT` 覆盖。它只影响视频，不影响预测与 IoU。`KEEP_FRAMES=1` 保留中间 PNG。
默认场景 3 全帧视频。不要加 `--scenes 3` 来做完整测试：`--video-scene 3`
只控制视频。视频范围两端包含，指从 0 开始的场景内序号，不是文件号或秒。
`SAVE_PREDICTIONS=0` 可关闭预测 NPZ；默认保留。

输出：

| 文件 | 内容 |
|---|---|
| `metrics.json` / `metrics.csv` | 与旧评估相同的 15 个 `SC*` / `SSC*` 指标 |
| `confusions.npz` | 51.2 / 25.6 / 12.8 m 全数据混淆矩阵 |
| `predictions/SCENE/TOKEN.npz` | 三维标签、原生标签、占据概率、observed/unknown 掩码、BEV 概率 |
| `scene_3_semantic_overlay.mp4` / `.gif` | 原始 3D Prediction + GT overlay / Camera：RadarOcc 蓝色背景、传统预测前景红、GT 前景黄、前景重叠橙 |
| `frames.json` | 每帧真实 RPC / GT / pose 路径、校准、占据数和提案数 |
| `run.json` | 配置、选中/处理帧数、成功或失败状态、数据哈希 |

默认导出把 Unknown 映射为 Free，以满足 RadarOcc 三类接口。
**这仅是评估导出策略，不表示雷达观测证明它是空闲。** `native_labels_xyz` 和
`unknown_mask` 保留 Unknown；评估会把这些体素正常计入错误，不会通过忽略预测 Unknown
来虚高 IoU。另一个显式策略为 `unknown_export="background"`，切换后应注明比较条件。
`SSC_mean` 是 BG/FG IoU 均值；SSC_free 单列报告。无定义的 IoU 写 JSON null。

## OctoMap 配置

预处理可用 `--config path/to/config.json`；未写字段使用默认值。缓存训练默认读取
manifest 配置，评估默认读取 RF 模型配置；显式覆盖后仍须通过缓存/模型一致性校验。
RF 阈值和 Unknown 导出策略不影响预处理缓存，但评估仍须匹配模型配置。
以下概率与 clamping 默认值来自固定版本的 `AbstractOccupancyOcTree` 构造函数，
窗口与 DBSCAN 是项目配置：

```json
{
  "temporal_window": 5,
  "fusion": "octomap",
  "p_hit": 0.7,
  "p_miss": 0.4,
  "occupancy_threshold": 0.5,
  "clamping_min": 0.1192,
  "clamping_max": 0.971,
  "max_range": -1.0,
  "lazy_eval": false,
  "discretize": false,
  "dbscan_eps_xy": 1.2,
  "dbscan_eps_z": 0.8,
  "dbscan_min_samples": 2,
  "unknown_export": "free"
}
```

`max_range<0` 不限距离；超范围回波只提供截断射线 Free，不在截断端点制造 Occupied。
`discretize=true` 采用原版端点体素中心去重插入，可能改变射线路径；默认关闭。
`lazy_eval=true` 延迟父节点更新，导出前调用 `updateInnerOccupancy()`；按原版语义不自动剪枝。
旧 `p_free`、`prior`、`occupied_threshold`、`free_threshold` 配置会报错；不要直接套用旧 JSON。

缓存格式为 `tradition-real-octomap-frame-fusion-v2`，模型格式为
`radarocc-octomap-occupancy-first-rf-v1`。即使旧 RF 也是 42 维仍会拒绝加载。
默认缓存改为 `data/frame_fusion_octomap`，模型目录改为
`work_dirs/tradition_real/octomap_rf_200`，以隔离旧实验。

## 验证集

先提供真正留出的验证集 PKL；它不能包含 RF 训练帧，重叠会直接报错。
脚本默认验证 annotation 名为 `kradar_dict_val_doppler8.pkl`，不自动创建或推断 split。
若本机名称不同，两个阶段都用同一个 `VAL_ANNOTATION`：

```bash
VAL_ANNOTATION=data/annotations/my_val.pkl bash run_tradition_real.sh preprocess-val
VAL_ANNOTATION=data/annotations/my_val.pkl bash run_tradition_real.sh validate
```

验证缓存默认 `data/frame_fusion_octomap/val`，结果默认
`work_dirs/tradition_real/octomap_rf_200_val`；分别用 `VAL_CACHE`、`VAL_OUTPUT` 覆盖。
`validate` 复用同一 `evaluate` Python 入口和已训练模型。`preprocess` / `all`
只包含 train_official 和 test_official；验证集单独按需运行。

## 自动验证

```bash
python -m unittest discover -s tradition_real/tests -v

# 在本机对比两个后端的完整 map_frame 耗时，并严格检查所有输出一致
python -m tradition_real.octomap.benchmark --points 1000 --frames 6
```

- **OctoMap 原版 C++ 对照**：直接编译 vendored 原版源文件，仅新增输入输出驱动。
  比较随机/边界/角点三维射线，以及多扫描下每个叶节点的 key、跨度、float32 log-odds、
  总节点数、父节点、阈值、clamping、maxrange、BBX、discretize、lazy、剪枝与重新展开。
  同时校验原始源码 SHA256；g++ 不存在时编译对照标记 skipped。
- **项目适配**：三维跨层射线、ROI 外端点、剪枝叶节点稠密展开、0.5 概率与 Unknown 区分、
  稠密结果与逐体素查询一致、因果窗口不重复计数、位姿对齐、场景/缺帧重置、RF 只修改语义。
- **绑定对照**：Python/C++ 对同一数据逐项比较射线 key、所有叶节点、父节点、剪枝、
  clamping 和查询；继续比较移动位姿与完整滑窗后的概率、掩码、proposal voxels 和全部 42D 特征。
  验证 Python 写入的缓存可由 C++ 后端选择下读取，缺失二进制不静默降级。
- **端到端**：生成合成 RPC/GT/RGB，执行 Python 在线与 C++ 缓存路径；删除 RPC/pose/标定，
  禁止调用 OctoMap/map_frame/DBSCAN 后仍完成缓存训练和评估。对比特征、标签、全部预测、
  指标和 4 帧视频；检查 RF 调参、缺失/损坏/未完成缓存、配置/标定冲突及数据泄漏拒绝。
  视频集成测试需要 OpenCV、Mayavi 和 ffmpeg；历史 Autoware/GM2019 测试只验证保留的旧模块。

这些验证不代替完整 K-Radar 实验。未在真实全量数据上测量 IoU 或耗时。
benchmark 是合成数据的本机测量，包括网格导出、DBSCAN 和 42D，但不包含读盘、压缩、GT 或 RF。
它先加载 C++ 模块再计时；同时对输出做精确比较。不能将其倍数直接视为全数据预处理加速倍数。
本次不增加多进程或 GPU，仍保留相同计算顺序；缓存后的 RF 不承担建图成本。

本次一次合成实测：1000 点/帧、6 帧、默认 5 帧滑窗，完整 `map_frame()` 总耗时
Python **68.76 s**，C++ 后端 **12.68 s**，约 **5.42×**，所有输出精确一致。
环境为 Linux / Intel Xeon Platinum 8370C、Python 3.12.14、GCC 13.3、
NumPy 2.3.5、scikit-learn 1.8.0、pybind11 3.1.0。该结果不是用户机器或真实数据集的保证。
