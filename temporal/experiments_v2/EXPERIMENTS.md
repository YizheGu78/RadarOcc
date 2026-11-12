# RadarOcc主干投影修正后：从头训练安排表（projection_fixed_20261112）

本表在原V/P/T安排表上更新，保留V00～V11、P01～P07、T00～T14编号；新增V12/V13/V14检查和T15无对齐memory对照。

**本轮所有occupancy模型从头训练，T00单帧也必须重跑。** 沿用之前20%连续窗口、seed42、15 epochs协议；“从头”指不加载旧模型，不代表把数据改成全量训练集。

共同主干版本：`small_physical_model_audit_20261112`；训练目录：`work_dirs/geometry_v2/projection_fixed_20261112/<method>/`。
新检查/单批次记录/融合消融：`outputs/geometry_v2/projection_fixed_20261112/`；旧日志、checkpoint和离线报告保留。

## 新RadarOcc代码和各时序方案在哪里

- [Small主干源码](../../projects/occ_plugin/occupancy/detectors/radarocc_self_small.py)：角度反号、采样[R,A,E]、BNC重排及残差投影；替换的旧代码留作注释。
- [稀疏编码器](../../projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py)：Range Attention后过滤越界坐标及对应特征。保持56层稀疏高度和14层输出，E原始bin仅5～32进入spconv，其余仍参与前面的Range Attention。
- [Cross-Attention批次](../../projects/occ_plugin/occupancy/deformable_transformer/encoder_3D.py)：reference和value均为B，注意力入口检查形状。
- 本轮模型复查还修正位置编码轴序、3D注意力初始化、FPN检查点重复更新BN、T14零power编码及损失/GT忽略边界；详见[模型审计报告](../../outputs/geometry_v2/projection_fixed_20261112/model_integrity/REPORT.md)。实际注册的occupancy head为occ_head_fp16.py，当前仍按FP32运行。
- [共享训练配置](base.py)和[全部模型配置](configs/)：T00是修正后的RadarOcc Small；其他T项在同一主干上接入时序。
- 当前表针对Small及继承它的时序模型；Full/Strict不在本轮安排中。CenterFormer/RODNet是移植的时序模块，最终任务仍为occupancy。

共同执行顺序：原始NPZ → 当前/历史P1选点 → 时序融合入口（此前讨论的P0，代码标作P1）→ Range Attention → Sparse Encoder → 球坐标Self-Attention → 修正后的Cartesian投影 → occupancy。

| 方案 | 配置 | 时序实现 |
| --- | --- | --- |
| T00单帧 | [current_only.py](configs/current_only.py) | 不融合历史帧 |
| T01普通KNN、T02距离Top8 | [knn.py](configs/knn.py)、[distance_top8.py](configs/distance_top8.py) | [local_attention.py](../fusion/local_attention.py) |
| T03/T04/T05 LearnedKNN及预算 | [learned_knn.py](configs/learned_knn.py)、[learned_q32.py](configs/learned_q32.py)、[learned_q64.py](configs/learned_q64.py) | [learned_topk.py](../core/learned_topk.py)与local_attention |
| T06～T09 power区域/Z映射 | [box_z1.py](configs/box_z1.py)及box_z0/2/3.py | [box_z_p0.py](../fusion/box_z_p0.py) |
| T10 CenterFormer | [centerformer.py](configs/centerformer.py) | [centerformer/](../centerformer/) |
| T11 RODNet TDC | [rodnet_tdc.py](configs/rodnet_tdc.py) | [rodnet/](../rodnet/) |
| T12/T13 RA-ADTA | [ra_adta.py](configs/ra_adta.py)、[ra_adta_radius.py](configs/ra_adta_radius.py) | [ra_adta.py](../fusion/ra_adta.py) |
| T14近场屏蔽、T15无对齐memory | [knn_mask37.py](configs/knn_mask37.py)、[unaligned_memory.py](configs/unaligned_memory.py) | power_mask与[unaligned_memory.py](../fusion/unaligned_memory.py) |

## 先做哪些：建议按阶段执行

| 阶段 | ID | 做什么、目的 |
| --- | --- | --- |
| 0：公共检查 | V00 → V12/V14 → V13 | 时序几何、CPU投影、真实学习层与梯度、CUDA整网单批次；CPU不能代替GPU验证 |
| 1：基本对照 | T00 → T01 | 先得到新单帧对照，再检验普通KNN是否带来收益；T01前核对V11 |
| 2：LearnedKNN | P01/P02 → P03 → P04；T02/T03/T04 | 重建或核对监督、训练新ranker；比较同候选池距离排序/学习排序以及Q32预算 |
| 3：其他主方法 | T07、T10、T11、T12 | 分别比较power区域、CenterFormer、RODNet TDC和RA-ADTA；各自准备步骤见下表 |
| 4：补充消融 | T05/T06/T08/T09/T13/T14/T15 | 查询预算、高度映射、搜索方式、近场屏蔽、无对齐memory；按研究问题选做 |

CUDA反向梯度也已修正：跨机器同步源码后，先重编译三维采样扩展。V13会核对.so的源码指纹，拒绝旧二进制；梯度失败仍会中止，未放宽检查。
```bash
source use_radarocc5060.sh
(cd third_party/VoxFormer/deform_attn_3d && FORCE_CUDA=1 MAX_JOBS=2 python setup.py build_ext --inplace)
```

第一轮主训练共9项：T00/T01/T02/T03/T04/T07/T10/T11/T12；其余7项是按需要补的对照。单卡逐项执行，不要求一次把全部模型跑完。
V/P用于检查和准备，不产生occupancy训练结果；只有P03训练独立ranker，只有T项训练完整模型。命令只执行指定ID，不会自动运行前置项，也不会自动开始下一模型。

```bash
source use_radarocc5060.sh
python tools/temporal/run_geometry_v2.py run V00
python tools/temporal/run_geometry_v2.py run V12
python tools/temporal/run_geometry_v2.py run V14  # CPU：实际学习层/损失/梯度与16配置初始化
python tools/temporal/run_geometry_v2.py run V13  # 需要GPU；失败时先解决该问题
python tools/temporal/run_geometry_v2.py run T00 --dry-run
python tools/temporal/run_geometry_v2.py run T00
```

## 哪些必须重做、哪些可以复用

| 产物 | 本轮处理 | 判断条件 |
| --- | --- | --- |
| 所有旧occupancy checkpoint/验证成绩 | 所有T模型重新训练 | 单帧投影也已改变；旧权重不能用于resume或作为新主干的公平对照 |
| V00、V12、V13、V14 | 必须重做 | 直接检查当前源码；无GPU时V13尚未完成 |
| P01/P02 ranker监督、P05/P06区域配对 | 有合格v2产物可复用，否则重建 | 坐标为kradar_physical_v2，manifest完整，标注/角度表/策略/查询预算匹配，非smoke产物 |
| P03/P04 ranker训练和评分 | 本轮默认在新目录重新训练/验证 | P01/P02不依赖主干投影；旧正号ranker不能复用；本轮配置统一指向新ranker |
| V01～V11已完成的离线几何报告 | 相同v2协议可复用 | 主干修正未改变原始P1输入；坐标、帧范围、K、半径、提框/匹配参数改变时需重算 |
| P07选点缓存 | 可跳过；若使用需随新ranker重建 | 默认在线选点；本轮新缓存路径与旧缓存隔离 |
| 原始雷达NPZ、LiDAR occupancy GT、位姿、数据划分 | 保留 | 本轮不重新生成GT，不反号GT；仅新模型投影与物理坐标统一 |

V01/V10及此前scene3方向框/候选64报告属于test场景历史诊断，不是必须重复的训练步骤，也不能再用该场景选择新参数。
P01/P02沿用完整train/val的旧ranker协议，额外使用完整训练集bbox监督；occupancy训练本身为20%。这不是整个方法总共只用20%标签的协议。
power配对仍使用预设B门限（Y=4/5/3m，中心距离=4/6/8m）；V04只在train内选参。若改门限，须同步缓存和配置，不自动采用扫描最优值。

## 模型之间具体比什么

| 对照 | 回答的问题 |
| --- | --- |
| T00 ↔ T01 | 最基础的历史KNN融合能否超过修正后的单帧？ |
| T02 ↔ T03 | 固定大对称框、64候选和Top8，仅学习排序有没有用？ |
| T03 ↔ T04 ↔ T05 | 每range时序查询250/32/64的精度、时间和显存如何权衡？ |
| T06 ↔ T07 ↔ T08 ↔ T09 | 同一power区域和历史配对下，高度映射方式有何影响？ |
| T07/T10/T11/T12 ↔ T00/T01/T03 | 各时序方案在相同主干与训练协议下的总体效果 |
| T12 ↔ T13；T01 ↔ T14 | RA-ADTA搜索方式、近场power屏蔽各自有什么作用？ |
| T15 ↔ T00，再与T01对照 | 不显式自车对齐的历史memory能否提供收益？结构不同，不能只归因于对齐 |

主要记录同一验证集的SSC_mean/SSC_Foreground/SC与SSC1/SSC2；同时记录耗时、显存和ranker额外预训练成本。单批次smoke的时间/显存不是完整训练基准。
每项均按验证SSC_mean选best，验证命令仍评估同一val20。最终测试或全量数据实验另行固定协议；不要把当前验证成绩当作独立最终测试成绩。

## 实验总表

| ID | 实验 | 目的 | 本轮处理 | 直接前置 | 命令 |
| --- | --- | --- | --- | --- | --- |
| V00 | 坐标/位姿/GT/反向采样与旧模式回归 | 确认这次修正确实统一了雷达物理坐标、LiDAR GT 和历史位姿，避免后面的实验继续用错坐标。 | 本轮必须重做 | — | `python tools/temporal/run_geometry_v2.py run V00` |
| V01 | 场景3：三种方向搜索框 × K=8/32/64 | 重新回答：搜索历史点时，把框沿物体运动方向伸长，是否比对称框更容易找到同一物体的点？ | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V01` |
| V02 | Doppler方向/静态与运动查询前置验证 | 检查 Doppler 加自车补偿能否可靠判断运动方向，解释方向框和 Q32 查询筛选是否有物理依据。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V02` |
| V03 | 训练场景power框、权重/匹配参考报告 | 检查从高power回波提取出的区域是否有目标意义，以及这些区域在历史帧之间能否配上。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V03` |
| V04 | 区域门限扫描（训练场景内部时间留出） | 检查历史区域匹配门限是否过紧或过宽，找到召回与错配之间的合理取舍。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V03 | `python tools/temporal/run_geometry_v2.py run V04` |
| V05 | 区域门限A/B/C与严格GT纯度验证 | 验证宽松XY投影下的高Recall是否掩盖了混入其他目标或背景的问题，并比较区域门限A/B/C。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V05` |
| V06 | Q32/Q64预算、Doppler投影与GT覆盖 | 确认每range只选32/64个时序查询时，预算确实生效，以及省下计算时是否丢掉太多目标回波。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V06` |
| V07 | CenterFormer真实四帧前向/梯度 | 在完整训练前，排除 CenterFormer 移植模块接线、数值和反向传播的问题。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V07` |
| V08 | RODNet TDC真实四帧前向/梯度 | 在完整训练前，排除 RODNet 时序可变形卷积TDC移植模块的接线和梯度问题。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V08` |
| V09 | 冻结区域门限的跨验证场景检查 | 检查固定的区域门限在其他验证场景是否仍然有效，避免只在训练场景11上看起来合理。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V03 | `python tools/temporal/run_geometry_v2.py run V09` |
| V10 | 场景3：大对称搜索框 × K=8/32/64 | 重新检查 LearnedKNN 使用的大对称候选框，是否能覆盖足够多的同目标历史点，以及 K 大小带来多少差异。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V10` |
| V11 | 普通KNN：GT同目标召回/精度与有效邻居 | 在花时间重训普通 KNN 前，确认修正后的2m近邻搜索实际能否选到同一目标的历史回波。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run V11` |
| V12 | 修正后Small投影链CPU审计 | 确认新单帧主干的物理坐标、采样轴序、特征重排、残差投影及有效稀疏索引一致。 | 本轮必须重做 | V00 | `python tools/temporal/run_geometry_v2.py run V12` |
| V13 | 修正后单帧整网CUDA单批次检查 | 在长期训练前确认真实CUDA采样、spconv、新残差投影和occupancy损失能接通并得到有限梯度。 | 本轮必须重做 | V12, V14 | `python tools/temporal/run_geometry_v2.py run V13` |
| V14 | 实际学习层、初始化、损失和梯度CPU审计 | 检查投影之外的初始化、位置编码、注意力、解码器、损失和实际梯度，避免只证明形状能接上。 | 本轮必须重做 | V00 | `python tools/temporal/run_geometry_v2.py run V14` |
| P01 | 重建选点器train监督 | 用正确的雷达XYZ和GT归属，重新生成LearnedKNN选点器的训练样本，替换受旧坐标影响的监督。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run P01` |
| P02 | 重建选点器val监督 | 重新生成正确坐标下的选点器验证样本，使新选点器能够在一致的验证集合上选best和比较。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V00 | `python tools/temporal/run_geometry_v2.py run P02` |
| P03 | Learned Top8选点器独立预训练 | 重新训练一个能从64个候选中挑选8个历史点的打分器，修复旧坐标导致的错误监督。 | 本轮默认新目录重训ranker；合格v2权重不因主干投影自动失效 | P01, P02 | `python tools/temporal/run_geometry_v2.py run P03` |
| P04 | 新ranker验证：距离Top8/学习Top8/候选64 | 确认新ranker相对于纯距离排序到底改善了多少选点质量，区分候选池不足和排序失败。 | 本轮必须重做 | P03 | `python tools/temporal/run_geometry_v2.py run P04` |
| P05 | 重建power框历史配对 train20 | 为power框融合模型准备修正坐标后的训练帧区域与历史配对，避免训练时重复做区域提取。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V05 | `python tools/temporal/run_geometry_v2.py run P05` |
| P06 | 重建power框历史配对 val20 | 为power框融合模型准备与训练相同规则的验证配对，确保验证时没有换策略或使用GT框。 | 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做 | V05 | `python tools/temporal/run_geometry_v2.py run P06` |
| P07 | 选点缓存预热（可选） | 提前缓存冻结ranker的选点结果，减少后续反复在线打分的开销；不研究新算法。 | 可选；新ranker权重需要新的选点缓存 | P03 | `python tools/temporal/run_geometry_v2.py run P07` |
| T00 | 修正后当前帧对照（主干投影也已修改，需重训） | 提供相同20%数据、15轮训练协议下的当前帧对照，用于衡量时序模块的净收益。 | 本轮从头重训；不得resume旧投影checkpoint | V13 | `python tools/temporal/run_geometry_v2.py run T00` |
| T01 | 普通球形KNN，2m，8点/历史帧 | 重算最基础几何时序融合在正确坐标下能带来多少occupancy收益。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V11 | `python tools/temporal/run_geometry_v2.py run T01` |
| T02 | 大对称框64→距离Top8公平对照 | 给LearnedKNN建立同候选框、同64候选、同8点数量的纯距离对照，隔离学习排序的作用。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V00 | `python tools/temporal/run_geometry_v2.py run T02` |
| T03 | LearnedKNN，全部时序查询 | 重算完整LearnedKNN方案在正确GT监督和几何坐标下的最终效果。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P04 | `python tools/temporal/run_geometry_v2.py run T03` |
| T04 | LearnedKNN，32查询/range | 验证每range只让32个当前查询参与LearnedKNN融合，是否能在减少时序计算时保留最终精度。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P04, V06 | `python tools/temporal/run_geometry_v2.py run T04` |
| T05 | LearnedKNN，64查询/range预算对照 | 提供Q32与全查询之间的中间预算，判断32是否过少以及增至64能否恢复精度。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P04, V06 | `python tools/temporal/run_geometry_v2.py run T05` |
| T06 | power框选区域，Z0映射消融 | 检查在高度不确定时，沿历史区域的整个高度多取样，是否比指定单一对应高度更有效。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P05, P06 | `python tools/temporal/run_geometry_v2.py run T06` |
| T07 | power框选区域，Z1主方案 | 检验power框先定位并配对区域，再保持物理高度采历史回波的主要方案。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P05, P06 | `python tools/temporal/run_geometry_v2.py run T07` |
| T08 | power框选区域，Z2映射消融 | 验证按当前框与历史框的相对高度比例建立对应，是否优于Z1保持绝对高度。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P05, P06 | `python tools/temporal/run_geometry_v2.py run T08` |
| T09 | power框选区域，Z3映射消融 | 检查固定框高度映射不够准确时，学习额外高度缩放与平移能否带来收益。 | 本轮从头重训；不得resume旧投影checkpoint | V13, P05, P06 | `python tools/temporal/run_geometry_v2.py run T09` |
| T10 | CenterFormer移植 | 评估移植的CenterFormer时序注意力，相比KNN或power框融合是否更有效。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V07 | `python tools/temporal/run_geometry_v2.py run T10` |
| T11 | RODNet TDC移植 | 评估RODNet的TDC时序可变形卷积，在当前雷达输入上是否比逐点KNN attention更合适。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V08 | `python tools/temporal/run_geometry_v2.py run T11` |
| T12 | RA-ADTA各向异性KNN（原配置） | 重新评估之前RA-ADTA方案：按雷达方向分辨率组织各向异性邻域、学习采样位置是否改善occupancy。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V00 | `python tools/temporal/run_geometry_v2.py run T12` |
| T13 | RA-ADTA radius-anchor消融（可选） | 单独检验RA-ADTA的候选搜索方式，判断原各向异性KNN设置是否限制了采样效果。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V00 | `python tools/temporal/run_geometry_v2.py run T13` |
| T14 | KNN近场power屏蔽消融（可选） | 检查近车区域强回波是否干扰普通KNN融合，屏蔽它们能否改善occupancy。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V00 | `python tools/temporal/run_geometry_v2.py run T14` |
| T15 | 无显式自车对齐的历史memory对照（可选） | 补回之前无显式自车对齐的历史memory方案，检验历史描述符本身是否能提供收益。 | 本轮从头重训；不得resume旧投影checkpoint | V13, V00 | `python tools/temporal/run_geometry_v2.py run T15` |

## 训练、检查和看结果的通用用法

新单帧日志在`work_dirs/geometry_v2/projection_fixed_20261112/current_only/*.log.json`，普通KNN在`work_dirs/geometry_v2/projection_fixed_20261112/knn/*.log.json`，其他方法目录见逐项说明。
新ranker权重为`work_dirs/geometry_v2/projection_fixed_20261112/ranker/best.pth`；所有新T模型的load_from/resume_from均为None。
Q32/Q64只限制参与时序融合的当前查询，每range250个P1输入仍保留；不是只向主干输入32/64个点。

```bash
python tools/temporal/run_geometry_v2.py run T01 --dry-run  # 看完整训练命令
python tools/temporal/run_geometry_v2.py run T01 --smoke    # 整网单批次，不训练权重
python tools/temporal/run_geometry_v2.py run T01
python tools/temporal/run_geometry_v2.py val T01 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T01 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/latest.pth
python tools/temporal/run_geometry_v2.py collect
```

将N替换为实际best文件名。resume只用于本轮被中断的训练，不用于旧正号/旧主干模型。
配置和checkpoint保存projection_version；新入口会拒绝旧投影checkpoint。表格本轮指标与历史记录分开：experiments.csv的historical_record保留旧行，collect只扫描新训练目录。
已有v2缓存按上表核对后可跳过生成步骤。V/P的--smoke仅用于小样本调试，不能替代正式缓存。

## 只检查K=8及复核之前的候选64结论

V01/V10默认跑K=8/32/64三组。若已有相同v2结果无需重复；只想重做K=8时可运行：

```bash
python tools/temporal_vis/audit_dynamic_objects.py box --scene 3 \
  --start-order 80 --end-order 200 --stride 5 --max-frames 25 --k 8 \
  --coordinate-convention kradar_physical_v2 --output-dir outputs/geometry_v2/projection_fixed_20261112/direction_boxes_k8_only
```

若复核此前correct_direction的候选8/32/64实验，保持相同帧、框长度4/6/8、反向0.5、Y±1.5、Z±1，显式加入--coordinate-convention kradar_physical_v2并使用新输出目录；不要混同T03实际使用的大对称框。

## 同一新KNN checkpoint开/关融合

```bash
python tools/temporal/run_geometry_v2.py ablate-knn \
  --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/best_SSC_mean_epoch_N.pth
```

结果写入`outputs/geometry_v2/projection_fixed_20261112/knn_fusion_ablation_<checkpoint名>/REPORT.md`。用同一新权重和val20先on再off；off仍是联合训练过的主干，不能替代独立T00。
该命令不训练参数；旧投影checkpoint不允许混入本轮正式对比。

## 每项做什么、为什么做、看什么及完整命令

### V00 坐标/位姿/GT/反向采样与旧模式回归

**运行定位：** 公共必做。直接前置：无。

**本轮处理：** 本轮必须重做。当前记录：本轮时序几何回归通过。

**目的：** 确认这次修正确实统一了雷达物理坐标、LiDAR GT 和历史位姿，避免后面的实验继续用错坐标。

**具体做什么：** 运行坐标专项回归：真实角度表的 R/A/E→XYZ、非对称 GT 框归属、含转弯/横移的历史对齐、XYZ→原始网格反向采样、旧产物拒绝检查及配置解析；另取训练场景中的真实帧统计 GT 框内点数。CPU，不训练模型。

**产物与怎么看：** outputs/geometry_v2/projection_fixed_20261112/checks.json 的 passed、失败项和 actual_GT_frames。此项检查时序几何；单帧主干另看V12/V13。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 所有修正版实验的公共起点；若失败，先查失败原因，再做选点器和整网训练。

**GT怎么用：** 读取 GT 做几何检查，不用 GT 训练参数。

**配置备注：** CPU，读取真实角度表/帧；不启动训练。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V00
```

**展开后的实际命令：**

```bash
python tools/temporal/check_geometry_v2.py --output outputs/geometry_v2/projection_fixed_20261112/checks.json
```

### V01 场景3：三种方向搜索框 × K=8/32/64

**运行定位：** 可选：复核历史结论。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：历史有产物；按复用条件核对。

**目的：** 重新回答：搜索历史点时，把框沿物体运动方向伸长，是否比对称框更容易找到同一物体的点？

**具体做什么：** 使用 test 中的场景3，当前帧序号80～200、间隔5，共25帧；比较等体积的对称框、正确方向框、反向框。方向来自 GT 位移，属于理想方向诊断。默认依次运行 K=8、32、64。

**产物与怎么看：** outputs/geometry_v2/direction_boxes_k*/ 下的报告；看同目标命中、框外漏检、框内有正确点但 Top-K 没选中的比例及所选点纯度。结合样本数看结果。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 只用于重算之前的方向框结论；不是 T01/T03 必需输入，也不能把 GT 方向当成推理时已知条件。

**GT怎么用：** GT 同时提供理想运动方向和评分身份；不是可直接部署的选点方案。

**配置备注：** 旧协议复现诊断，scene3属于test；禁止用它调最终参数。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V01
```

**展开后的实际命令：**

```bash
python tools/temporal/run_geometry_v2.py direction-sweep
```

### V02 Doppler方向/静态与运动查询前置验证

**运行定位：** 建议：使用Doppler查询预算时复核。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 检查 Doppler 加自车补偿能否可靠判断运动方向，解释方向框和 Q32 查询筛选是否有物理依据。

**具体做什么：** 使用训练场景11，比较对称框、GT理想方向、单个token的Doppler方向、邻域投票方向；同时统计 Doppler 符号假设及方向不确定的样本。

**产物与怎么看：** outputs/geometry_v2/motion/；看有判定样本的方向准确率、反向率、不确定比例及相应历史点命中。方向准确率必须和判定覆盖率一起看。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 辅助解释 V06/T04/T05；不训练新方向网络，也不会自动修改 Q32 的筛选参数。

**GT怎么用：** 实际 Doppler 预测不读 GT；GT 用于理想方向对照和事后评分。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V02
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/audit_neighborhood_motion.py --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8.pkl --scene 11 --start-order 80 --end-order 200 --stride 5 --output-dir outputs/geometry_v2/motion --coordinate-convention kradar_physical_v2
```

### V03 训练场景power框、权重/匹配参考报告

**运行定位：** power框路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 检查从高power回波提取出的区域是否有目标意义，以及这些区域在历史帧之间能否配上。

**具体做什么：** 在训练场景11提取高power种子区域（97百分位），按XY组织区域并比较匹配代价权重，生成可复用参考报告。这里的框是回波区域框，不能直接视作车辆真实外形。

**产物与怎么看：** outputs/geometry_v2/regions_train/report.json 及同目录报告；看区域数量、覆盖、匹配正确/错误/未知数、Recall和Precision。当前GT归属是宽松XY投影，至少1个点即可，需结合V05严格审计。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 给 V04 和 V09 提供相同区域提取配置及参考数据；不生成整网训练用的 P05/P06 配对缓存。

**GT怎么用：** 提框和匹配不读 GT；GT 用于离线打分和比较权重。

**配置备注：** 仅train场景；GT用于评分，框与匹配不读GT。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V03
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/benchmark_region_matching.py --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8.pkl --scenes 11 --start-order 80 --end-order 200 --stride 5 --seed-percentile 97 --merge-space xy --max-box-size-m 20 6 6 --gt-space xy --match-space xy --gt-purity 0 --min-gt-points 1 --output-dir outputs/geometry_v2/regions_train --coordinate-convention kradar_physical_v2
```

### V04 区域门限扫描（训练场景内部时间留出）

**运行定位：** 可选：需要重新选择门限时做。直接前置：V03。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 检查历史区域匹配门限是否过紧或过宽，找到召回与错配之间的合理取舍。

**具体做什么：** 继承 V03 的训练场景和区域配置，扫描Y差、XY中心距离及代价权重。当前帧序号≤140用于选参，后续当前帧用于时间留出检查；相邻历史帧可能重叠，不是独立场景验证。

**产物与怎么看：** outputs/geometry_v2/gate_sweep/ 的 gate_sweep.csv、summary.md 和推荐组合；同时看选参段与留出段的Recall、保守Precision/F1、正确目标被门限排除的数量。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 只输出参数研究结论，不会自动写回 P05/P06 或训练配置；采用新门限时，需要同步另建缓存和对应配置。

**GT怎么用：** GT 用于训练场景内的参数评分；匹配器输入不含 GT。

**配置备注：** 继承v2参考报告；不读取旧scene3报告。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V04
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/sweep_region_matching_gates.py --reference-report outputs/geometry_v2/regions_train/report.json --output-dir outputs/geometry_v2/gate_sweep
```

### V05 区域门限A/B/C与严格GT纯度验证

**运行定位：** power框路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 验证宽松XY投影下的高Recall是否掩盖了混入其他目标或背景的问题，并比较区域门限A/B/C。

**具体做什么：** 在验证集对比A原门限、B只放宽Y、C放宽Y和中心距离；同时用更严格XYZ归属检查区域支持点、区域框及配对纯度。这里的A/B/C是区域匹配门限，不是V01的三种方向框。

**产物与怎么看：** outputs/geometry_v2/region_abc_val/summary.md、report.json及低纯度样例；看宽松/严格Recall、目标覆盖、support/box purity、保守Precision。宽松与严格Recall分母不同，不能直接作增减比较。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 解释power框匹配是否可信，是 P05/P06 的前置审计。默认仍固定B门限，不会根据本次验证集结果自动挑最优参数。

**GT怎么用：** GT 只负责归属、纯度与评分，不用于提框、配对或训练缓存生成。

**配置备注：** A=原门限，B=只放宽Y，C=放宽Y和中心；与V01方向框不同。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V05
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/audit_region_quality.py --ann-files data/annotations/kradar_dict_val_official_temporal_doppler8.pkl --start-order 80 --end-order 200 --stride 5 --output-dir outputs/geometry_v2/region_abc_val --coordinate-convention kradar_physical_v2
```

### V06 Q32/Q64预算、Doppler投影与GT覆盖

**运行定位：** Q32/Q64路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 确认每range只选32/64个时序查询时，预算确实生效，以及省下计算时是否丢掉太多目标回波。

**具体做什么：** 在验证场景1、2上运行当前的Doppler/自车补偿/空间分布查询筛选，展示Q32和Q64的查询位置与GT覆盖。原单帧每range250个token和历史候选输入仍保留。

**产物与怎么看：** outputs/geometry_v2/query_budget/；看各range查询数、筛选类别、空间分布、GT目标/点覆盖。这里检查选哪些当前查询，不是在评估历史邻居选得对不对。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 为 T04/T05 提供筛选正确性和覆盖解释；是否保持最终精度还要比较 T03/T04/T05。

**GT怎么用：** 查询筛选不读 GT；GT 仅用于可视化和覆盖评分。

**配置备注：** Q32指每range最多32个时序查询，原单帧250个token仍保留。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V06
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/visualize_query_budget.py --ann-files data/annotations/kradar_dict_val_official_temporal_doppler8.pkl --scenes 1 2 --output-dir outputs/geometry_v2/query_budget --coordinate-convention kradar_physical_v2
```

### V07 CenterFormer真实四帧前向/梯度

**运行定位：** CenterFormer路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮命令已完成；训练结果查看日志。

**目的：** 在完整训练前，排除 CenterFormer 移植模块接线、数值和反向传播的问题。

**具体做什么：** 读取一组真实四帧雷达输入，在CPU上执行模块前向和两次小优化更新；检查初始恒等输出、有限数值、可学习偏移的非零梯度及输入未被原地修改。

**产物与怎么看：** 控制台检查结果：identity、finite_gradients、offset_gradient 和有效历史支持。两次更新使用描述符数值目标，不是occupancy训练。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 通过后再跑 T10；通过只说明该模块在这个样本上可计算、可求梯度，不说明整网GPU训练或最终IoU已验证。

**GT怎么用：** 不加载occupancy GT；不使用bbox监督。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V07
```

**展开后的实际命令：**

```bash
python -m temporal.check_p0 --method centerformer --device cpu --config temporal/experiments_v2/configs/centerformer.py
```

### V08 RODNet TDC真实四帧前向/梯度

**运行定位：** RODNet路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮命令已完成；训练结果查看日志。

**目的：** 在完整训练前，排除 RODNet 时序可变形卷积TDC移植模块的接线和梯度问题。

**具体做什么：** 与V07使用同类真实四帧检查，在CPU上前向并做两次小优化更新，检查TDC偏移分支有梯度、输出有限和原始描述符未被修改。

**产物与怎么看：** 控制台的identity、finite_gradients、offset_gradient和历史支持统计；检查目标不是occupancy损失。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 通过后再跑 T11；这是模块可运行检查，不是检测效果、完整GPU训练或IoU证明。

**GT怎么用：** 不加载occupancy GT；不使用bbox监督。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V08
```

**展开后的实际命令：**

```bash
python -m temporal.check_p0 --method rodnet --device cpu --config temporal/experiments_v2/configs/rodnet_tdc.py
```

### V09 冻结区域门限的跨验证场景检查

**运行定位：** 建议：power框跨场景复核。直接前置：V03。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 检查固定的区域门限在其他验证场景是否仍然有效，避免只在训练场景11上看起来合理。

**具体做什么：** 继承V03的区域配置，在验证场景比较原门限、只放宽Y、只改中心门限及固定B组合。B的t-1/t-2/t-3门限为Y=[4,5,3]m、XY中心距离=[4,6,8]m。

**产物与怎么看：** outputs/geometry_v2/gates_cross_val/summary.md；看分场景和汇总Recall、保守Precision/F1、明确错配和未知身份数，留意是否只有个别场景受益。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 给P05/P06默认固定B门限提供跨场景检查；不会读取V04推荐值，也不会自动把验证集最优组合写回训练。

**GT怎么用：** GT 只用于评分；区域提取和匹配不读 GT。

**配置备注：** 继承v2训练场景报告；固定B门限，禁止用旧报告替代。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V09
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/validate_region_matching_gates.py --reference-report outputs/geometry_v2/regions_train/report.json --ann-files data/annotations/kradar_dict_val_official_temporal_doppler8.pkl --exclude-scenes 11 --y-gates 4 5 3 --center-gates 4 6 8 --output-dir outputs/geometry_v2/gates_cross_val
```

### V10 场景3：大对称搜索框 × K=8/32/64

**运行定位：** 可选：复核候选框。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 重新检查 LearnedKNN 使用的大对称候选框，是否能覆盖足够多的同目标历史点，以及 K 大小带来多少差异。

**具体做什么：** 同样使用场景3的25帧，只比较大对称框；t-1/t-2/t-3 的 X 半宽分别4/6/8m，Y半宽1.5m、Z半宽1m。默认跑 K=8、32、64。

**产物与怎么看：** outputs/geometry_v2/large_symmetric_k*/ 的覆盖、Top-K 命中和纯度；区分框本身漏掉目标与距离排序把正确点挤掉。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 辅助解释 T02/T03；不生成选点器权重或训练缓存。场景3来自 test，仅作历史诊断，不据此调新参数。

**GT怎么用：** GT 用于判断同目标与评分；对称框构造和距离排序不读取 GT。

**配置备注：** X半宽4/6/8m；旧scene3诊断，不用于新参数选择。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V10
```

**展开后的实际命令：**

```bash
python tools/temporal/run_geometry_v2.py direction-sweep --large-box
```

### V11 普通KNN：GT同目标召回/精度与有效邻居

**运行定位：** 普通KNN路线前置。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：历史有产物；按复用条件核对。

**目的：** 在花时间重训普通 KNN 前，确认修正后的2m近邻搜索实际能否选到同一目标的历史回波。

**具体做什么：** 在训练场景11、序号80～200、间隔5上做离线匹配；历史帧先做自车对齐，每个查询在2m内最多选8个近邻。不训练 attention 或 occupancy。

**产物与怎么看：** outputs/geometry_v2/knn_matching/；看同目标召回、所选点精度、有效近邻数，以及 t-1/t-2/t-3 的退化。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 为 T01 提供选点质量解释；几何召回高不等于最终 occupancy IoU 一定高，后者由 T01 检验。

**GT怎么用：** GT 只用于评分，不参与 KNN 搜索。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V11
```

**展开后的实际命令：**

```bash
python tools/temporal_vis/audit_dynamic_objects.py match --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8.pkl --scene 11 --start-order 80 --end-order 200 --stride 5 --radius 2 --k 8 --output-dir outputs/geometry_v2/knn_matching --coordinate-convention kradar_physical_v2
```

### V12 修正后Small投影链CPU审计

**运行定位：** 公共必做：主干投影。直接前置：V00。

**本轮处理：** 本轮必须重做。当前记录：本轮CPU投影审计通过；GPU另看V13。

**目的：** 确认新单帧主干的物理坐标、采样轴序、特征重排、残差投影及有效稀疏索引一致。

**具体做什么：** 执行当前Small源码的CPU方法，用独立物理公式、非对称位置编码网格、真实Scene3第80帧检查18项契约。学习层与稀疏卷积使用明确的CPU夹具，不把它称为整网验证。

**产物与怎么看：** outputs/geometry_v2/projection_fixed_20261112/projection_audit/REPORT.md和report.json；必须fail=0、review=0，CUDA是否运行另列。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 通过后运行V13；该审计只证明几何/张量对应，不预测修正后的IoU。

**GT怎么用：** GT用于加载轴序和评价器检查，不训练参数。

**配置备注：** 执行当前Small源码；不读取旧checkpoint，不训练模型。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V12
```

**展开后的实际命令：**

```bash
python tools/temporal/audit_projection_chain.py --require-pass --output-dir outputs/geometry_v2/projection_fixed_20261112/projection_audit
```

### V13 修正后单帧整网CUDA单批次检查

**运行定位：** 公共必做：GPU整网接线。直接前置：V12, V14。

**本轮处理：** 本轮必须重做。当前记录：本轮命令已完成；训练结果查看日志。

**目的：** 在长期训练前确认真实CUDA采样、spconv、新残差投影和occupancy损失能接通并得到有限梯度。

**具体做什么：** 先核对.so中的梯度版本/源码指纹，比较实际CUDA采样核与独立插值的前向/反向，再构造修正后的T00，从头初始化，取一个真实train样本做loss.backward，另取一个val样本推理。不执行optimizer.step，不加载或保存模型权重。首次同步CUDA梯度修正后须重编译扩展，不能只更新Python。

**产物与怎么看：** outputs/geometry_v2/projection_fixed_20261112/model_smoke/T00.json；失败先看projection_audit_cuda/cuda_failure.json中的stage/error，以及command_logs/内完整子进程日志。无CUDA时明确未完成；完整检查通过才有completed=true。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 所有T实验共用此主干检查；每种时序模型准备好产物后，还可运行run Txx --smoke检查自己的整网接线。

**GT怎么用：** 训练样本GT用于一次occupancy损失反向；验证标签只用于评价输出。

**配置备注：** 真实训练/验证数据；整网前向、反向和推理；不保存权重，不执行长期训练。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V13
```

**展开后的实际命令：**

```bash
python tools/temporal/run_geometry_v2.py model-smoke T00
```

### V14 实际学习层、初始化、损失和梯度CPU审计

**运行定位：** 公共必做：模型数值与梯度。直接前置：V00。

**本轮处理：** 本轮必须重做。当前记录：本轮CPU实际学习层/梯度审计通过；GPU另看V13。

**目的：** 检查投影之外的初始化、位置编码、注意力、解码器、损失和实际梯度，避免只证明形状能接上。

**具体做什么：** 实际构造并初始化16个T配置；用独立八邻域oracle检查CPU三维采样；真实Self/Cross Attention、ResNet3D、FPN、OccHead在合成缩小网格前向/反向及临时SGD更新；检查checkpoint的BN统计、GT忽略边界、T14屏蔽、冻结ranker和全部序列因果关系。未执行spconv/CUDA，临时ranker不代表正式预训练完成。

**产物与怎么看：** outputs/geometry_v2/projection_fixed_20261112/model_integrity/REPORT.md和report.json；passed=true、failures/errors为空。whole_model_verified保持false，整网另看V13。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 通过后执行V13；所有T实验使用相同已修正学习层；不能用CPU梯度测试预测真实IoU。

**GT怎么用：** 合成GT检查损失/评价边界；真实GT只抽样统计，未训练实际模型。

**配置备注：** 16配置实际构造/初始化；真实学习层缩小网格反向；不代替CUDA整网验证。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run V14
```

**展开后的实际命令：**

```bash
python tools/temporal/audit_model_integrity.py --require-pass --output-dir outputs/geometry_v2/projection_fixed_20261112/model_integrity
```

### P01 重建选点器train监督

**运行定位：** LearnedKNN路线必做。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 用正确的雷达XYZ和GT归属，重新生成LearnedKNN选点器的训练样本，替换受旧坐标影响的监督。

**具体做什么：** 遍历完整训练集四帧窗口，生成历史对齐后的候选框、最多64候选及同一GT实例的正负标签，保存相对几何/描述符等特征；每帧每历史对默认最多512个合格查询。不训练参数。

**产物与怎么看：** data/geometry_v2/ranker/train/manifest.json及分片；检查坐标版本、帧数、查询/候选数量、正负样本与排除统计。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 与P02一起供P03训练；必须重建，不能把旧ranker缓存复制到新目录当作修正后的数据。

**GT怎么用：** 使用完整训练集bbox产生监督标签；这是额外标注预算，不能称为整个方法只用20%标签。

**配置备注：** 保持旧ranker完整train/val预训练协议；occupancy仍20%。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P01
```

**展开后的实际命令：**

```bash
python tools/temporal/train_top8_ranker.py prepare --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8.pkl --split train --output-dir data/geometry_v2/ranker/train --coordinate-convention kradar_physical_v2
```

### P02 重建选点器val监督

**运行定位：** LearnedKNN路线必做。直接前置：V00。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 重新生成正确坐标下的选点器验证样本，使新选点器能够在一致的验证集合上选best和比较。

**具体做什么：** 在完整验证集构造与P01同策略的历史候选和GT标签；默认不限制每帧合格查询数。不执行选点器参数更新。

**产物与怎么看：** data/geometry_v2/ranker/val/manifest.json及分片；核对坐标、策略、样本数量，并确保train/val连同历史帧没有重叠。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 供P03每轮验证和P04最终选点比较；验证标签用于评分与选best，不用于梯度训练。

**GT怎么用：** 使用验证bbox提供评分标签，不是独立test结果。

**配置备注：** 保持旧ranker完整train/val预训练协议；occupancy仍20%。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P02
```

**展开后的实际命令：**

```bash
python tools/temporal/train_top8_ranker.py prepare --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8.pkl --split val --output-dir data/geometry_v2/ranker/val --coordinate-convention kradar_physical_v2 --max-queries-per-frame 0
```

### P03 Learned Top8选点器独立预训练

**运行定位：** LearnedKNN路线必做；仅训练选点器。直接前置：P01, P02。

**本轮处理：** 本轮默认新目录重训ranker；合格v2权重不因主干投影自动失效。当前记录：本轮待运行。

**目的：** 重新训练一个能从64个候选中挑选8个历史点的打分器，修复旧坐标导致的错误监督。

**具体做什么：** 只训练独立ranker，用P01的GT正负标签监督候选打分；20轮、seed42，以P02验证Recall@8保存best。此步骤不训练RadarOcc occupancy主模型。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/ranker/best.pth、每轮val CSV与best_val.csv；看训练损失及验证Recall@8，检查checkpoint记录kradar_physical_v2。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** P04检查通过后，将这个权重冻结供T03/T04/T05使用；三项occupancy训练共享这一次ranker预训练。

**GT怎么用：** 训练集bbox监督ranker损失；验证bbox负责评分和选best。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P03
```

**展开后的实际命令：**

```bash
python tools/temporal/train_top8_ranker.py train --train-cache data/geometry_v2/ranker/train --val-cache data/geometry_v2/ranker/val --work-dir work_dirs/geometry_v2/projection_fixed_20261112/ranker --epochs 20 --seed 42
```

### P04 新ranker验证：距离Top8/学习Top8/候选64

**运行定位：** LearnedKNN路线必做。直接前置：P03。

**本轮处理：** 本轮必须重做。当前记录：本轮待运行。

**目的：** 确认新ranker相对于纯距离排序到底改善了多少选点质量，区分候选池不足和排序失败。

**具体做什么：** 在同一P02验证缓存上比较距离Top8、学习Top8和整个候选Top64；不更新权重。候选64表示搜索候选池的覆盖参考，不是一个同成本的8点算法。

**产物与怎么看：** outputs/geometry_v2/projection_fixed_20261112/ranker_val.csv；比较同目标Recall、所选点Precision、不同历史间隔的结果与候选池可达覆盖。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 若64候选本来就找不到同目标，先查框/对齐；若候选有正确点但学习排序仍很差，先查ranker，再跑T03/T04/T05。无预设必须超过多少的阈值。

**GT怎么用：** GT 只用于评价已冻结ranker；推理输入没有GT标签。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P04
```

**展开后的实际命令：**

```bash
python tools/temporal/train_top8_ranker.py evaluate --cache data/geometry_v2/ranker/val --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ranker/best.pth --output outputs/geometry_v2/projection_fixed_20261112/ranker_val.csv
```

### P05 重建power框历史配对 train20

**运行定位：** power框路线必做。直接前置：V05。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 为power框融合模型准备修正坐标后的训练帧区域与历史配对，避免训练时重复做区域提取。

**具体做什么：** 对20%训练连续窗口提取power区域、对齐历史并用固定B门限配对，保存current/history框、配对标记和逆变换；不训练参数，不读取GT。

**产物与怎么看：** data/geometry_v2/box_pairs/train20/manifest.json及分片；检查complete、坐标版本、门限、帧数和GT未使用标记。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 供T06～T09共同读取；四个高度方案共享这份缓存。若改变匹配门限，需生成另一路径的缓存并同步配置。

**GT怎么用：** 不读取bbox或occupancy GT，不在这里产生GT监督标签。

**配置备注：** 无GT输入；默认固定B门限，尚不宣称新最优。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P05
```

**展开后的实际命令：**

```bash
python tools/temporal/cache_box_z_pairs.py --ann-file data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl --output-dir data/geometry_v2/box_pairs/train20 --coordinate-convention kradar_physical_v2
```

### P06 重建power框历史配对 val20

**运行定位：** power框路线必做。直接前置：V05。

**本轮处理：** 同协议、完整kradar_physical_v2产物可复用；旧正号或设置变化时重做。当前记录：本轮待运行。

**目的：** 为power框融合模型准备与训练相同规则的验证配对，确保验证时没有换策略或使用GT框。

**具体做什么：** 对20%验证连续窗口执行与P05一致的无GT区域提取、历史对齐和固定B配对，不训练参数。

**产物与怎么看：** data/geometry_v2/box_pairs/val20/manifest.json及分片；核对策略与train20一致、数据划分正确且缓存完整。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 与P05一起供T06～T09训练过程中的验证使用；不是使用验证GT修正搜索区域。

**GT怎么用：** 配对生成不读GT；后续occupancy验证另用该划分的occupancy标签评分。

**配置备注：** 无GT输入；默认固定B门限，尚不宣称新最优。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P06
```

**展开后的实际命令：**

```bash
python tools/temporal/cache_box_z_pairs.py --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl --output-dir data/geometry_v2/box_pairs/val20 --coordinate-convention kradar_physical_v2
```

### P07 选点缓存预热（可选）

**运行定位：** 可选：工程加速。直接前置：P03。

**本轮处理：** 可选；新ranker权重需要新的选点缓存。当前记录：本轮待运行。

**目的：** 提前缓存冻结ranker的选点结果，减少后续反复在线打分的开销；不研究新算法。

**具体做什么：** 使用T03配置和P03权重，在CUDA上预计算选点并写入独立缓存。不更新ranker或occupancy模型。

**产物与怎么看：** data/geometry_v2/top8_selection/中的缓存产物与完整性记录；关注是否命中、是否对应同一配置/坐标/权重。

**对比谁：** 前置检查或数据准备，不直接比较occupancy IoU

**和后续实验的关系：** 可以完全跳过：当前T03/T04/T05默认在线选点，执行P07不会自动把这些训练配置切换为读缓存。

**GT怎么用：** 冻结ranker推理不读GT。

**配置备注：** 需CUDA；训练默认在线选点，不依赖缓存预热。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run P07
```

**展开后的实际命令：**

```bash
python tools/temporal/cache_top8_selection.py temporal/experiments_v2/configs/learned_knn.py --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ranker/best.pth --cache-dir data/geometry_v2/projection_fixed_20261112/top8_selection
```

### T00 修正后当前帧对照（主干投影也已修改，需重训）

**运行定位：** 主对照：修正主干后重训。直接前置：V13。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮整网单批次通过；正式训练待运行。

**目的：** 提供相同20%数据、15轮训练协议下的当前帧对照，用于衡量时序模块的净收益。

**具体做什么：** 从头训练current-only RadarOcc，不融合历史帧；数据和优化设置与其他T实验保持同一套协议。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/current_only/；以验证SSC_mean为主，辅看Foreground、SC、SSC1/SSC2，作为其他T实验对照。

**对比谁：** 所有T模型的共同单帧对照；首次必须重新训练

**和后续实验的关系：** 旧单帧权重不能替代新主干的独立训练对照。本轮配置已经自动使用独立work_dir，保留旧日志。

**GT怎么用：** 使用occupancy训练标签；无额外ranker bbox预训练。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T00
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/current_only.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T00 --smoke
python tools/temporal/run_geometry_v2.py val T00 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/current_only/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T00 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/current_only/latest.pth
```

### T01 普通球形KNN，2m，8点/历史帧

**运行定位：** 主训练：普通KNN。直接前置：V13, V11。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 重算最基础几何时序融合在正确坐标下能带来多少occupancy收益。

**具体做什么：** 历史帧自车对齐后，对每个当前查询在2m球内每历史帧最多选8个近邻，再用attention融合；从头联合训练时序模块和occupancy网络。无独立选点器。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/knn/的验证日志与best；和同协议T00比较最终SSC_mean/Foreground，结合V11解释邻居质量。

**对比谁：** T01 vs T00：基础KNN时序收益

**和后续实验的关系：** 这是你列出的普通KNN重跑主项；也作为T14屏蔽实验和其他时序模型的基础参照。

**GT怎么用：** occupancy损失使用训练标签；近邻搜索本身不读GT。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T01
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/knn.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T01 --smoke
python tools/temporal/run_geometry_v2.py val T01 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T01 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/latest.pth
```

### T02 大对称框64→距离Top8公平对照

**运行定位：** 建议：LearnedKNN公平对照。直接前置：V13, V00。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 给LearnedKNN建立同候选框、同64候选、同8点数量的纯距离对照，隔离学习排序的作用。

**具体做什么：** 使用与T03相同的大对称历史框和最多64候选，按距离选8点；其余融合与训练设置按对应配置运行，不加载ranker。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/distance_top8/；重点与T03比较，才能回答学习排序是否改善最终occupancy。T01的球形候选域不同，T01→T03差异不能只归因于ranker。

**对比谁：** T03 vs T02：同候选64/保留8点，隔离学习排序的作用；T02 vs T01改变了候选搜索域

**和后续实验的关系：** 研究LearnedKNN改进来源时建议跑；不必先训练P03才能做这项。

**GT怎么用：** occupancy损失使用标签；框内搜索和距离排序不读GT。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T02
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/distance_top8.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T02 --smoke
python tools/temporal/run_geometry_v2.py val T02 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/distance_top8/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T02 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/distance_top8/latest.pth
```

### T03 LearnedKNN，全部时序查询

**运行定位：** 主训练：LearnedKNN。直接前置：V13, P04。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 重算完整LearnedKNN方案在正确GT监督和几何坐标下的最终效果。

**具体做什么：** 冻结P03新ranker，从每个历史帧最多64候选中选8点；不启用Q32/Q64当前查询预算，随后训练时序融合与occupancy主模型。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/learned_knn/；与T02比较学习选点收益，与T04/T05比较查询预算影响；独立记录显存和迭代时间才能评估计算代价。

**对比谁：** T03 vs T02检验学习排序；T03 vs T00检验完整方案

**和后续实验的关系：** 这是你列出的LearnedKNN主训练，也作为Q32/Q64预算方案的全查询参照；不能使用旧坐标ranker权重。

**GT怎么用：** 训练使用occupancy标签；额外bbox监督已在P03使用，冻结选点时不读GT。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T03
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/learned_knn.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T03 --smoke
python tools/temporal/run_geometry_v2.py val T03 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_knn/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T03 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_knn/latest.pth
```

### T04 LearnedKNN，32查询/range

**运行定位：** 主训练：Q32。直接前置：V13, P04, V06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 验证每range只让32个当前查询参与LearnedKNN融合，是否能在减少时序计算时保留最终精度。

**具体做什么：** 在T03基础上启用Q32查询筛选；只对被选查询搜索历史8点并融合，其余当前token仍保留给单帧主网络。使用相同新ranker，从头训练occupancy模型。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/learned_q32/；与T03/T05比较SSC_mean、Foreground和实际时间/显存，结合V06目标覆盖解释精度变化。

**对比谁：** T04 vs T03/T05：Q32/Q250/Q64的精度、时间和显存

**和后续实验的关系：** 对应你说的“每个range保留32个点”时序预算实验；实际不是把整个原始输入压成32点，也不是每查询只留32个历史邻居。

**GT怎么用：** 查询筛选与冻结ranker不读GT；使用occupancy训练标签和P03已有bbox预训练。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T04
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/learned_q32.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T04 --smoke
python tools/temporal/run_geometry_v2.py val T04 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_q32/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T04 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_q32/latest.pth
```

### T05 LearnedKNN，64查询/range预算对照

**运行定位：** 可选：查询预算对照。直接前置：V13, P04, V06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 提供Q32与全查询之间的中间预算，判断32是否过少以及增至64能否恢复精度。

**具体做什么：** 与T04相同，只把每range时序查询上限从32改为64；同一ranker、相同每查询历史Top8和occupancy训练协议。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/learned_q64/；与T03/T04共同观察预算—精度—计算量取舍。不要仅因查询数减半就宣称总训练耗时减半。

**对比谁：** T05 vs T04/T03：同ranker、同候选，仅改变当前查询预算

**和后续实验的关系：** 预算消融项；如果当前只想重跑你关注的Q32，可以先跳过，之后再补。

**GT怎么用：** 与T04一致；筛选过程不读GT。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T05
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/learned_q64.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T05 --smoke
python tools/temporal/run_geometry_v2.py val T05 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_q64/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T05 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/learned_q64/latest.pth
```

### T06 power框选区域，Z0映射消融

**运行定位：** 可选：power框Z0消融。直接前置：V13, P05, P06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 检查在高度不确定时，沿历史区域的整个高度多取样，是否比指定单一对应高度更有效。

**具体做什么：** power框配对后，XY按当前/历史框的中心与尺寸映射；Z0在历史框内均匀取8个高度，有效样本聚合成该历史帧特征，再训练融合与occupancy。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/box_z0/；与T07～T09比较SSC和Foreground，并看z_valid_sampling_rate、有效融合查询数，区分采不到回波与融合无收益。

**对比谁：** T06/T07/T08/T09：同区域/配对缓存，仅比较Z映射方案

**和后续实验的关系：** 高度映射消融；与其他Z方案共享P05/P06，框和配对不变，只改变高度取样方式。

**GT怎么用：** 提框和配对无GT；网络使用occupancy标签训练。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T06
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z0.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T06 --smoke
python tools/temporal/run_geometry_v2.py val T06 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z0/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T06 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z0/latest.pth
```

### T07 power框选区域，Z1主方案

**运行定位：** 主训练：power框Z1。直接前置：V13, P05, P06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 检验power框先定位并配对区域，再保持物理高度采历史回波的主要方案。

**具体做什么：** 使用P05/P06；XY按两框映射，Z1直接保留当前查询的物理Z，不按框的高度比例缩放。通过逆自车变换回到历史雷达网格采样，再训练融合与occupancy。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/box_z1/；与T00/T01/T03比较总体收益，与T06/T08/T09比较高度处理；关注采样有效率和GT-free区域匹配审计。

**对比谁：** T07 vs T00/T01/T03：power区域融合；另和Z0/Z2/Z3对照

**和后续实验的关系：** 这是你列出的“按回波power框选目标”的主要重跑项。power区域是雷达回波区域，不能直接当成GT车辆长方体。

**GT怎么用：** 提框和配对不使用GT；occupancy损失使用训练标签。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T07
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z1.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T07 --smoke
python tools/temporal/run_geometry_v2.py val T07 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z1/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T07 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z1/latest.pth
```

### T08 power框选区域，Z2映射消融

**运行定位：** 可选：power框Z2消融。直接前置：V13, P05, P06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 验证按当前框与历史框的相对高度比例建立对应，是否优于Z1保持绝对高度。

**具体做什么：** 与T07共享框、配对和XY映射；Z2把查询在当前框内的相对高度，按历史框中心和高度尺寸映射过去，不额外学习高度变换参数。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/box_z2/；重点对比T07的SSC、Foreground和有效采样率，判断回波区域高度变化是否值得做比例映射。

**对比谁：** T08 vs T07：相对高度映射 vs 保持物理高度

**和后续实验的关系：** 用于区分固定物理Z与框归一化Z两种假设；如果只跑主power方案，可后补。

**GT怎么用：** 框和对应不读GT；使用occupancy标签训练。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T08
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z2.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T08 --smoke
python tools/temporal/run_geometry_v2.py val T08 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z2/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T08 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z2/latest.pth
```

### T09 power框选区域，Z3映射消融

**运行定位：** 可选：power框Z3消融。直接前置：V13, P05, P06。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 检查固定框高度映射不够准确时，学习额外高度缩放与平移能否带来收益。

**具体做什么：** 在Z2框比例映射上增加可学习高度缩放exp(alpha)和偏移beta，并对二者加正则；共享P05/P06，从头训练融合与occupancy。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/box_z3/；看SSC/Foreground、有效采样率、z_alpha_mean、z_shift_mean、z_scale_mean及相应正则损失。参数有变化本身不等于效果变好。

**对比谁：** T09 vs T08：学习高度缩放/平移的增益

**和后续实验的关系：** 和T08对比学习高度修正的增益；和T07比较最终是否值得增加这份复杂度。

**GT怎么用：** occupancy标签训练映射参数；无GT框指导采样位置，另有参数正则。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T09
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z3.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T09 --smoke
python tools/temporal/run_geometry_v2.py val T09 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z3/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T09 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/box_z3/latest.pth
```

### T10 CenterFormer移植

**运行定位：** 补充主训练：移植CenterFormer。直接前置：V13, V07。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 评估移植的CenterFormer时序注意力，相比KNN或power框融合是否更有效。

**具体做什么：** 四帧先按物理坐标对齐、组织特征网格，以当前token为query做空间感知融合和多尺度可变形注意力，再接回现有前端位置；从头训练整个occupancy模型。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/centerformer/；比较SSC/Foreground及实际训练成本，结合V07检查与有效历史支持判断模块是否工作。

**对比谁：** T10 vs T00/T01/T03：当前主干上的CenterFormer时序移植

**和后续实验的关系：** 属于之前移植方案的修正版重跑；移植的是时序组件并做了适配，不是完整CenterFormer检测器，没有新增检测热图选点头。

**GT怎么用：** 使用occupancy标签；不额外训练CenterFormer检测头或使用GT热图选query。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T10
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/centerformer.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T10 --smoke
python tools/temporal/run_geometry_v2.py val T10 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/centerformer/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T10 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/centerformer/latest.pth
```

### T11 RODNet TDC移植

**运行定位：** 补充主训练：移植RODNet TDC。直接前置：V13, V08。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 评估RODNet的TDC时序可变形卷积，在当前雷达输入上是否比逐点KNN attention更合适。

**具体做什么：** 四帧物理对齐后构成特征网格，按物理高度切片执行共享权重的因果TDC，学习空间采样偏移，再采回当前查询并训练occupancy。只用当前和历史帧。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/rodnet_tdc/；和T10/T01等比较SSC/Foreground及训练代价，检查TDC偏移和有效历史支持。

**对比谁：** T11 vs T10/T01/T00：当前主干上的RODNet TDC移植

**和后续实验的关系：** 属于之前RODNet移植方案重跑；当前移植的是TDC，不是完整RODNet，也不是额外的Doppler速度预测网络。

**GT怎么用：** 使用occupancy训练标签；不使用未来帧或检测GT作为融合输入。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T11
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/rodnet_tdc.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T11 --smoke
python tools/temporal/run_geometry_v2.py val T11 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/rodnet_tdc/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T11 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/rodnet_tdc/latest.pth
```

### T12 RA-ADTA各向异性KNN（原配置）

**运行定位：** 补充主训练：RA-ADTA。直接前置：V13, V00。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 重新评估之前RA-ADTA方案：按雷达方向分辨率组织各向异性邻域、学习采样位置是否改善occupancy。

**具体做什么：** 使用修正物理坐标及历史自车对齐，保持此前anisotropic_knn搜索、8候选、每头2个采样点等参数，从头训练RA-ADTA与occupancy模型。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/ra_adta/；与T01/T03等比较最终SSC/Foreground，辅看有效采样及训练开销。此项不是在P03 ranker上继续训练。

**对比谁：** T12 vs T01/T03：RA-ADTA采样方案

**和后续实验的关系：** 补齐先前依赖雷达XYZ的另一条方案；如果此前做过该方案，应按修正版本重跑才可比较。

**GT怎么用：** 使用occupancy标签；无需bbox ranker预训练。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T12
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/ra_adta.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T12 --smoke
python tools/temporal/run_geometry_v2.py val T12 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ra_adta/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T12 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ra_adta/latest.pth
```

### T13 RA-ADTA radius-anchor消融（可选）

**运行定位：** 可选：RA-ADTA消融。直接前置：V13, V00。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 单独检验RA-ADTA的候选搜索方式，判断原各向异性KNN设置是否限制了采样效果。

**具体做什么：** 与T12保持同样主要网络和采样设置，只把search_mode换为radius_anchor，独立从头训练。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/ra_adta_radius/；主要与T12比较最终SSC/Foreground、有效历史支持和计算代价。

**对比谁：** T13 vs T12：仅替换RA-ADTA搜索方式

**和后续实验的关系：** 搜索方式消融，不是修正坐标必需的新模型；先完成T12再按需要补。

**GT怎么用：** 使用occupancy训练标签；不依赖P03选点器。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T13
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/ra_adta_radius.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T13 --smoke
python tools/temporal/run_geometry_v2.py val T13 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ra_adta_radius/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T13 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/ra_adta_radius/latest.pth
```

### T14 KNN近场power屏蔽消融（可选）

**运行定位：** 可选：近场屏蔽消融。直接前置：V13, V00。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 检查近车区域强回波是否干扰普通KNN融合，屏蔽它们能否改善occupancy。

**具体做什么：** 在T01基础上，对当前与历史各帧水平距离≤3.7m的token屏蔽power通道，并将这些点标为无效候选；其余2m/8近邻和训练协议保持对应设置，从头训练。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/knn_mask37/；主要与T01比较SSC/Foreground及近场效果，同时留意屏蔽是否损失近处真实目标信息。

**对比谁：** T14 vs T01：近场power屏蔽；不是新增独立主方法

**和后续实验的关系：** 可选的近场处理消融；3.7m是预设对照半径，不是本次修正后已证明的最优值。

**GT怎么用：** 屏蔽只由几何距离决定，不读GT；训练使用occupancy标签。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T14
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/knn_mask37.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T14 --smoke
python tools/temporal/run_geometry_v2.py val T14 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn_mask37/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T14 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn_mask37/latest.pth
```

### T15 无显式自车对齐的历史memory对照（可选）

**运行定位：** 可选：补回已有无对齐方案。直接前置：V13, V00。

**本轮处理：** 本轮从头重训；不得resume旧投影checkpoint。当前记录：本轮待运行。

**目的：** 补回之前无显式自车对齐的历史memory方案，检验历史描述符本身是否能提供收益。

**具体做什么：** 同样使用四帧和修正后的Small主干；将历史token压为256个memory token，再更新当前描述符，不按物理XYZ执行自车对齐。独立从头训练。

**产物与怎么看：** work_dirs/geometry_v2/projection_fixed_20261112/unaligned_memory/；和T00比较历史特征的净收益，和T01/T03比较精度及计算开销。

**对比谁：** T15 vs T00：无显式对齐的历史memory；与KNN比较时同时存在结构差异

**和后续实验的关系：** 这是不同结构的时序对照；与KNN的差异不能全部归因于是否做自车对齐。

**GT怎么用：** 使用occupancy标签；memory选择不使用GT。

**配置备注：** 修正主干；20%窗口；15 epochs；seed42；scratch；新目录。

**运行命令：**

```bash
python tools/temporal/run_geometry_v2.py run T15
```

**展开后的实际命令：**

```bash
bash tools/dist_train.sh temporal/experiments_v2/configs/unaligned_memory.py 1 --seed 42
python tools/temporal/run_geometry_v2.py run T15 --smoke
python tools/temporal/run_geometry_v2.py val T15 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/unaligned_memory/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T15 --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/unaligned_memory/latest.pth
```

