# 大对称框、64 候选与冻结的学习 Top-8

本方案只改变历史 Token 的选择，不预测坐标 offset、不移动 Query、不修改当前
P1 稀疏索引，也不使用 GT 或目标 ID 进行推理。原 KNN、RA-ADTA 和已有离线
box / neighborhood 实验保持可用，只有新配置启用本方案。

重复标注 ID：prepare 自动跳过当前帧 ID 重复的全部目标，以及历史帧同 ID 对应多个框的
关联。重复 ID 框内的点仍可进入候选池，但不参与正负样本监督；不改名、不按类别猜匹配，
不丢弃整帧。日志输出警告，duplicate_ids.json 记录场景、帧及重复 ID 对应的框索引，
manifest.json 记录排除计数。原离线审计的严格 ID 校验保持不变。
此前中断的缓存目录不能继续追加，请为修复后的 prepare 使用新的 output-dir。

## 固定流程

1. 复用官方物理 bin 表与 Ego Alignment，将历史点变换到当前雷达坐标系。
2. t-1 / t-2 / t-3 分别搜索 X±4 / ±6 / ±8 m，Y±1.5 m，Z±1 m。
3. 每个历史帧按原 Query 的欧氏距离保留最多 64 个，距离相同按原候选索引排序。
4. 独立的 6,145 参数 MLP 为候选打分，每帧选择最高分的最多 8 个，共最多 24 个。
5. 将原始特征按这些索引送入已有 P1 LocalTemporalAttention；融合公式、位置/帧偏移
   编码、零初始化残差、Doppler bin ID 保留和 RadarOcc 下游保持原样。

评分器输入为当前/候选的原始 Doppler8、归一化后描述符绝对差、相对 XYZ、欧氏距离、
真实时间差（秒）。功率/统计通道固定 signed-log1p / 80，bin ID / 64；64 只是数值
缩放，不是 bin -> m/s 转换。预处理没有运行统计或可训练 Encoder，不随主干训练变化。
MLP 输出加到负的归一化空间距离上，高分优先；零初始化时退化为空间距离排序。
几何距离保留 FP64，MLP 输入/权重使用 FP32，以避免近距离并列点因提前转换精度而换序。
评分器用秒，已有 Attention 的位置编码仍用帧偏移；不混用两种单位。

不足 64 / 8 时使用 mask，不复制真实点补齐。不丢掉有效范围零点回波；单独配置的
power_mask 仍照常生效，默认关闭。树搜索 CPU、评分器与 Attention 在输入设备执行；
评分器分块推理，未构造完整 Query×所有历史点距离矩阵。

## 第一阶段：准备训练/验证缓存

需要原始 Doppler8 NPZ、带有效 pose/timestamp 的 temporal PKL、官方 bin 表以及已有
processed bbox / bbox_token / object_category NPY。只依赖 PyTorch、NumPy、SciPy，
无需启动 MMCV/spconv 或训练 RadarOcc。

```bash
cd /home/user1/projects/RadarOcc
conda activate radarocc5060

python tools/temporal/train_top8_ranker.py prepare \
  --split train \
  --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8.pkl \
  --max-queries-per-frame 512 \
  --output-dir data/top8_ranker/train

python tools/temporal/train_top8_ranker.py prepare \
  --split val \
  --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  --max-queries-per-frame 0 \
  --output-dir data/top8_ranker/val
```

默认使用指定 split 的所有场景、所有当前帧和前 3 个 split 历史位置，不跨场景。
`--max-queries-per-frame` 对每个当前/历史帧对限制查询数，0 表示不抽样；默认 512。
可用 `--scenes 3 --start-order 80 --end-order 200 --stride 5
--max-frames-per-scene 25` 做小范围运行，但不能把测试集改名当训练集。
`--bbox-root` 默认 data/K-RadarOcc/train，与原审计一致，可按实际目录覆盖。
输出目录必须是新目录；中断产生的不完整目录没有 manifest，不能用于训练。

GT 只定义目标归属、监督和 Stage A 可比评价人群：同 ID 历史框存在、当前框中心在前方、
对齐后 GT dx 非近零。GT 位移、方向和目标 ID 不写入评分器输入。
监督当前目标 Token，其他目标/背景候选为负例，重叠 bbox 候选不进入训练损失；
这些候选仍留在搜索/评价中。所有有效候选参与平衡正负损失，而非只监督硬 Top-8。
候选池没有正例的查询跳过排序损失，仍保留在评价中。无同目标历史 P1 的查询不进入
Recall 分母，与原审计相同。背景查询/全部静态区域的泛化尚需完整模型验证，不能把
所有背景点当成一个同目标正例类。

缓存使用模型的每 range 250/250 P1 选择与排序；完全相同功率的并列项在 CPU/GPU
torch.sort 的顺序可能不同，bbox 归属和 P1 成员不变，但截止处等距候选的索引可能不同。
manifest 记录实际参与的当前和历史帧、采样、几何/标注文件哈希。训练会拒绝 train/val
中任何当前或历史帧重叠，拒绝把 test 缓存作为训练/调参输入。

## 独立训练与评价

```bash
python tools/temporal/train_top8_ranker.py train \
  --train-cache data/top8_ranker/train \
  --val-cache data/top8_ranker/val \
  --epochs 20 --batch-size 256 --device cuda \
  --work-dir work_dirs/p1_top8_ranker

python tools/temporal/train_top8_ranker.py evaluate \
  --cache data/top8_ranker/val \
  --checkpoint work_dirs/p1_top8_ranker/best.pth \
  --device cuda --output work_dirs/p1_top8_ranker/val_comparison.csv
```

work-dir 必须是新目录。best.pth 依据全部有效验证查询的 micro Recall@8 选择；last.pth
保存最后一轮选点器。两者都是独立评分器权重，不是完整 RadarOcc checkpoint，
也不是包含优化器的断点续训文件。
best_val.csv 和每轮 CSV 按历史帧输出 distance_top8 / learned_top8 / pool_top64 的
Coverage、Recall、Precision、Outside Box、Top-K Miss 与实际平均选中数，保留原始分子分母。
抽样缓存结果不能直接当成先前未抽样表格；范围/样本/筛选必须一致才可直接比较。
Recall@64 是固定候选池内至少命中一个同目标 Token 的上限，不表示 Occupancy 性能。
如果学习 Top-8 未改善验证集，先保留距离基线，不因为已经训练就默认其有效。

## 第二阶段：冻结加载到 RadarOcc

```bash
# 从头训练完整 RadarOcc；评分器必须已经训练并保存。
bash run_temporal_t4_box64_top8_p1.sh train \
  work_dirs/p1_top8_ranker/best.pth

# 或显式从已有单帧 checkpoint 微调完整模型。
bash run_temporal_t4_box64_top8_p1.sh train \
  work_dirs/p1_top8_ranker/best.pth \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth

# 相同大框/64池/Attention，仅改成空间距离 Top-8 的配对基线。
bash run_temporal_t4_box64_top8_p1.sh train unused '' distance

# 用完整模型权重评价；第二参数仍是独立评分器 checkpoint。
bash run_temporal_t4_box64_top8_p1.sh test \
  work_dirs/p1_top8_ranker/best.pth \
  work_dirs/radarocc_small_temporal_t4_box64_learned_top8_p1/epoch_15.pth
```

对应配置：
- RadarOcc_Small_5060_true_fp32_temporal_t4_box64_learned_top8_p1.py
- RadarOcc_Small_5060_true_fp32_temporal_t4_box64_distance_top8_p1.py

加载时严格校验 checkpoint 结构、描述符版本、秒单位、框尺寸、64池和8输出。
错误或缺失的权重直接报错，不用随机评分器继续训练。requires_grad=False，且在父模型
train() 后评分器仍为 eval；其调用不建梯度图。按索引提取的 Attention 原始特征没有
detach，后续梯度保持正常。
独立评分器不使用 BN/Dropout。完整模型 checkpoint 也包含冻结评分器状态，恢复时仍应
提供同一独立评分器文件与配置作为构造契约，不要换成其他训练轮次。
硬 Top-8 不可微，本配置不联合训练选点器；用单独缓存训练脚本更新后再做新的配对实验。

## 检查

### 完整模型选点索引缓存（加速冻结选点器）

这与 train_dedup/val_dedup 监督缓存不同。使用完整模型实际 Dataset、GPU P1
排序和所有 Query，缓存最终 24 个历史索引及 mask，不缓存 Attention 特征。
预生成在 Selector 返回后截断，不运行 Attention/Occupancy/backward，不更新权重，
无需提供 Occupancy checkpoint。train/val 各自沿用原数据划分；GT 不参与索引计算。

```bash
# 先验证 train/val 各前 5 个样本；再运行一次可测量缓存命中的耗时。
TOP8_CACHE_DIR=data/top8_selection bash run_temporal_t4_box64_top8_p1.sh \
  cache-check work_dirs/p1_top8_ranker/best.pth

# 全量预生成 train+val。中断可重新运行，已完成条目自动跳过计算。
TOP8_CACHE_DIR=data/top8_selection bash run_temporal_t4_box64_top8_p1.sh \
  cache work_dirs/p1_top8_ranker/best.pth

# 默认严格只读缓存，缺少任何条目立即报错，不悄悄恢复慢速在线选点。
TOP8_CACHE_DIR=data/top8_selection bash run_temporal_t4_box64_top8_p1.sh \
  train work_dirs/p1_top8_ranker/best.pth

# 已有该 learned-top8 完整模型的 epoch checkpoint 时，可续训保留优化器/轮次。
TOP8_CACHE_DIR=data/top8_selection \
TOP8_RESUME_FROM=work_dirs/radarocc_small_temporal_t4_box64_learned_top8_p1/epoch_1.pth \
  bash run_temporal_t4_box64_top8_p1.sh train work_dirs/p1_top8_ranker/best.pth
```

验证比较索引/mask 和非零 Attention 输出完全一致，并打印在线/缓存选点时间；首次
缺失时包含在线选点和写盘，第二次才反映读缓存时间。完整模型 step 时间仍需实际训练日志测量。
未设 TOP8_CACHE_DIR 的原命令继续在线计算；TOP8_CACHE_MODE=readwrite 可边训练边补齐，
第一遍仍然较慢，后续 epoch 读取缓存。cache 默认 train+val，test 需显式
TOP8_CACHE_SPLITS=test 预生成；缓存验证/测试输入不更新任何模型权重。

采用输入内容哈希而非假设 NPZ 序号与 GPU 排序一致：原始特征、排列顺序、对齐坐标、
时间、帧/点 mask、搜索策略、实际加载的选点器权重、实现版本、设备及数值设置进入校验。
相同输入及排列直接复用索引；排序或增强发生变化时只读模式报 miss，不会指向错误的点。
全模型 resume/load 后重新计算选点器权重签名，避免与传入 ranker 不一致。条目 metadata
记录预生成时的 split、数据集序号、场景、帧序列和雷达路径；不依赖 GT/ID。
不缓存原始特征，命中时仍有一次输入拷贝和校验；不能承诺达到单帧训练速度。

每条用 int32 索引和 packbits mask，未压缩存储减少每 epoch 解压：43750 Query、
T=4 时约 4.33 MB/当前帧（含少量元数据），总容量按实际样本数估计。
不要把缓存放在很慢的网络盘。没有更改任何 Occ loss、Attention 公式和已有 checkpoint 结构。

使用 20% 子集时，用实际 20% 配置运行 tools/temporal/cache_top8_selection.py，再用
相同配置训练；快捷脚本依然默认官方完整训练集。需保持相同的 GPU/数值设置。

```bash
python -m unittest discover -s tests -p test_selection_cache.py -v
python -m unittest discover -s tests -p test_learned_top8.py -v
python -m unittest discover -s tests -p test_temporal_attention.py -v
python -m unittest discover -s tests -p test_directional_box_matching.py -v
```

实际 GPU 延迟、显存、完整 MMCV/spconv 训练与真实 K-Radar 指标应在 radarocc5060 环境测量。
代码实现和合成数据测试不等于已经得到训练好的真实数据 checkpoint。
