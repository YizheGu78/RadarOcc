# RadarOcc投影修正后：从头训练入口

完整安排表仍在原位置：[EXPERIMENTS.md](EXPERIMENTS.md)。
它保留原V00～V11、P01～P07、T00～T14编号，新增V12/V13/V14公共检查与T15无对齐memory对照。
每项写明目的、操作、前置、与谁比较、GT用途、结果位置，以及训练/验证/恢复命令。

所有模型训练结束后，运行 `bash tools/temporal/run_all_fusion_ablation.sh`，
按方案串行验证同一最佳权重的融合开启/旁路两臂，覆盖T01～T15，包括CenterFormer和RODNet TDC。
T00独立单帧只验证一次，用于区分推理融合收益与整个训练方案的收益。
默认使用各自`best_SSC_mean*.pth`，缺少权重的方案明确跳过，不回退到latest。
可先加`--dry-run`核对清单；用`--ids T00 T01 T03 T10 T11`只跑指定方案；
有多个best时用`--checkpoint T10=/绝对路径/指定权重.pth`指定。
结果在`outputs/geometry_v2/projection_fixed_20261112/fusion_on_off/REPORT.md`与`summary.csv`。
同一命令再次执行会复用已完成、权重/验证标注/源码指纹一致的两臂；变化后需用`--output-dir`另建输出目录。
T14旁路仍保留近场power屏蔽；所有方案严格加载全部权重，从checkpoint保存配置构建，使用统一val20，且不执行训练更新。

本轮沿用之前的**20%连续窗口、15 epochs、seed42、AdamW 3e-4、scratch**协议。
“从头”表示不加载旧occupancy权重，不表示改成完整训练集。
新单帧RadarOcc是T00，其余T模型均使用同一个修正后的Small主干；所有T模型都需要重训。

## 源码和本轮目录

- [Small源码](../../projects/occ_plugin/occupancy/detectors/radarocc_self_small.py)：负atan2/asin、采样轴序、BNC重排、先投影残差再相加。
- [Sparse Encoder](../../projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py)：Range Attention之后过滤越界坐标与对应特征。
- [Transformer批次](../../projects/occ_plugin/occupancy/deformable_transformer/encoder_3D.py)：Cross-Attention的value/reference统一为B。
- [base.py](base.py)保存公共训练协议和`projection_version='small_physical_model_audit_20261112'`；[configs/](configs/)保存各模型配置。
- 本轮还修正位置编码、注意力初始化、FPN检查点重复更新BN、T14零power编码及损失/GT忽略边界；见V14的[模型审计报告](../../outputs/geometry_v2/projection_fixed_20261112/model_integrity/REPORT.md)。
- [registry.py](registry.py)统一管理命令、依赖和表格；运行`table`会更新原Markdown/CSV/JSON。

替换的旧模型代码保留为注释。E轴保留原56层稀疏网格和14层输出：
原始E索引5～32进入spconv，0～4、33～36越界而被裁掉；裁剪之前Range Attention仍使用全部输入。
本轮不是扩展到保留全部37个E bin的模型。Full/Strict不在本表范围内。

新occupancy和ranker训练目录：`work_dirs/geometry_v2/projection_fixed_20261112/<method>/`。
新检查、单批次记录、融合消融：`outputs/geometry_v2/projection_fixed_20261112/`。
旧`work_dirs/geometry_v2/knn/`等目录保留；`run T01`现在会直接使用新目录。
已有checkpoint不会迁移到新目录，也不会自动作为初始化权重。

## 建议起步顺序

本轮还修正了CUDA三线性插值的坐标反向梯度。若在其他机器同步了源码，需先在该机器重编译扩展；审计会核对`.so`内的源码指纹并拒绝旧二进制：

```bash
source use_radarocc5060.sh
(cd third_party/VoxFormer/deform_attn_3d && FORCE_CUDA=1 MAX_JOBS=2 python setup.py build_ext --inplace)
```

```bash
source use_radarocc5060.sh
python tools/temporal/run_geometry_v2.py list
python tools/temporal/run_geometry_v2.py run V00
python tools/temporal/run_geometry_v2.py run V12
python tools/temporal/run_geometry_v2.py run V14  # CPU：实际学习层/初始化/损失/梯度
python tools/temporal/run_geometry_v2.py run V13  # GPU：实际采样核与整网单批次
python tools/temporal/run_geometry_v2.py run T00 --dry-run
python tools/temporal/run_geometry_v2.py run T00
```

V00检查时序几何/配置，V12检查当前Small投影契约，V13运行真实CUDA采样核前向/反向，
再取一个train样本反向和一个val样本推理。V13不执行optimizer.step，不加载或保存模型权重，
也不产生可比较的IoU。没有GPU时V13明确报未运行，CPU检查不能代替它。
V13失败时，先看`outputs/geometry_v2/projection_fixed_20261112/projection_audit_cuda/cuda_failure.json`的`stage/error/traceback`，以及`command_logs/`中的完整日志；外层`CalledProcessError`只是转报。
V14运行实际学习层的缩小网格前向/反向，并核查16个配置的真实构造/初始化；没有执行spconv，不能代替V13。
命令只执行指定项；前置不会自动运行，也不会自动接着训练其他模型。

普通KNN：核对V11报告，然后执行T01。
若已有相同坐标和参数的完整v2报告，可复用V11，不必重复生成。

```bash
python tools/temporal/run_geometry_v2.py run V11             # 仅在需要重做时
python tools/temporal/run_geometry_v2.py run T01 --smoke     # 本方法的整网单批次
python tools/temporal/run_geometry_v2.py run T01
```

## LearnedKNN与Q32

```bash
python tools/temporal/run_geometry_v2.py run P01  # 合格完整v2监督可复用，否则生成
python tools/temporal/run_geometry_v2.py run P02
python tools/temporal/run_geometry_v2.py run P03  # 本轮新目录，从头训练ranker
python tools/temporal/run_geometry_v2.py run P04
python tools/temporal/run_geometry_v2.py run T02  # 同候选64的纯距离Top8对照
python tools/temporal/run_geometry_v2.py run T03  # LearnedKNN，全时序查询
python tools/temporal/run_geometry_v2.py run V06  # 如已有相同v2预算检查，可复用
python tools/temporal/run_geometry_v2.py run T04  # 每range32个时序查询
```

T05提供64查询对照。Q32/Q64仅限制时序融合的当前查询，主干仍保留每range250个P1 token。
各方法在准备齐产物后均可先`run Txx --smoke`，它只执行整网单批次，不训练或保存权重。
P07选点缓存预热可跳过，默认在线选点；使用时缓存必须对应本轮新ranker。

P01/P02保持完整train/val的旧ranker协议，额外使用完整训练集bbox监督；
occupancy训练为20%。请报告这个额外标签预算，不能把整个方法描述成只用20%标签。

## power区域与其他时序主方法

- power：2026-11-16采用C门限，Y=2/2/1.5m、XY中心距离=8/10/12m。重建P05/P06，再分别训练T07/T08/T09（Z1/Z2/Z3）；T06的Z0是可选对照。
- CenterFormer：V07模块检查 → T10；RODNet TDC：V08模块检查 → T11。
- RA-ADTA：T12；T13替换搜索方式。
- T14在T01上增加近场power屏蔽。
- T15补回无显式自车对齐的历史memory方案，与T00比较历史特征收益；它与KNN的结构也不同。

Z1保留对齐坐标系中的当前高度，Z2按两框高度比例映射，Z3在Z2基础上学习缩放/偏移。
三者使用同一份C缓存、相同20%数据和随机种子，从头训练整个模型。
沿用当前XY提框，不额外施加“最终框高度超过4/6米就整框删除”的规则。

```bash
python tools/temporal/run_geometry_v2.py run P05
python tools/temporal/run_geometry_v2.py run P06
python tools/temporal/run_geometry_v2.py run T07 --smoke
python tools/temporal/run_geometry_v2.py run T08 --smoke
python tools/temporal/run_geometry_v2.py run T09 --smoke
# 每条命令独立执行；检查通过后再正式训练
python tools/temporal/run_geometry_v2.py run T07
python tools/temporal/run_geometry_v2.py run T08
python tools/temporal/run_geometry_v2.py run T09
```

C缓存位于`data/geometry_v2/projection_fixed_20261112/box_pairs_c_20261116/`。
结果位于`work_dirs/geometry_v2/projection_fixed_20261112/box_z1_c_20261116/`及z2/z3对应目录。
仅更新配置与入口，不代表已完成长训练。旧结果保留。

### Z3全量在线配对（不生成配对缓存）

现有`BoxZTemporalDatasetV2`支持`box_pair_mode=online`。读取一个当前帧时，
用已加载的当前帧和有效历史帧即时提取P97/MAD3.0的power区域，做自车对齐与
C门限匹配（Y=2/2/1.5m，XY中心距离=8/10/12m），将配对直接送入Z3。
历史GT不参与配对；occupancy训练仍使用当前帧GT。场景开头缺少的历史槽忽略。
每次读取样本都重新计算，不读写manifest/配对NPZ，也不保留跨样本的配对缓存。
训练与验证采用相同在线策略。原`cache`模式仍为默认值，旧命令可继续运行。

```bash
source use_radarocc5060.sh
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z3.py 1 \
  --seed 42 \
  --work-dir work_dirs/geometry_v2/projection_fixed_20261112/box_z3_c_full_online \
  --cfg-options \
  data.train.ann_file=data/annotations/kradar_dict_train_official_temporal_doppler8.pkl \
  data.val.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  data.test.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  data.train.box_pair_mode=online \
  data.val.box_pair_mode=online \
  data.test.box_pair_mode=online \
  model.box_z_cfg.diagnostic_path=work_dirs/geometry_v2/projection_fixed_20261112/box_z3_c_full_online/eval_diagnostics.jsonl
```

这条命令直接启动单卡、15轮、seed42的从头训练，无需P05/P06。
配置中的旧`box_cache`路径在online模式被忽略，无需删除已有缓存。
使用`dist_train.sh`覆盖在线参数；`run T09`的默认入口仍按缓存方案检查前置产物。
在线提框增加CPU数据加载开销，实际速度取决于运行环境；本轮不据此承诺训练吞吐。

### Z3全量按需缓存：首次生成，后续epoch复用

设置`box_pair_mode=readwrite`后，读取当前帧时先检查它的配对文件：不存在则按上述
online规则生成并原子写入，存在则直接读取。无需预先导出完整数据集，也无需在第二个
epoch手动切换模式；中断后重新启动仍能复用已生成的条目，未完成的条目继续生成。
第一轮验证同样按需生成独立的验证缓存。缓存只保存框、配对mask和逆变换；Z3可学习的
高度缩放/偏移、特征采样和融合每次前向都重新计算，不会冻结为第一轮输出。

```bash
source use_radarocc5060.sh
bash tools/dist_train.sh temporal/experiments_v2/configs/box_z3.py 1 \
  --seed 42 \
  --work-dir work_dirs/geometry_v2/projection_fixed_20261112/box_z3_c_full_readwrite \
  --cfg-options \
  data.train.ann_file=data/annotations/kradar_dict_train_official_temporal_doppler8.pkl \
  data.val.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  data.test.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  data.train.box_pair_mode=readwrite \
  data.val.box_pair_mode=readwrite \
  data.test.box_pair_mode=readwrite \
  data.train.box_cache=data/geometry_v2/projection_fixed_20261112/box_pairs_c_readwrite/train_full \
  data.val.box_cache=data/geometry_v2/projection_fixed_20261112/box_pairs_c_readwrite/val_full \
  data.test.box_cache=data/geometry_v2/projection_fixed_20261112/box_pairs_c_readwrite/val_full \
  model.box_z_cfg.diagnostic_path=work_dirs/geometry_v2/projection_fixed_20261112/box_z3_c_full_readwrite/eval_diagnostics.jsonl
```

固定C门限：Y=2/2/1.5m，XY中心距离=8/10/12m；Z3、单卡、15轮、seed42、从头训练。
不需要P05/P06。两种缓存格式必须使用不同目录，不能把旧离线manifest缓存放到此路径。
`policy.json`锁定标注、角度表、提框/匹配代码、区域参数和门限；每帧NPZ另核对四帧源文件
路径、大小、修改时间及有效历史mask。协议或输入变化时报错，应使用新缓存目录重新生成。
源雷达NPZ应保持只读；这里的逐帧检查是文件元数据检查，不是每次读取都重算内容SHA256。
多个worker/rank可同时使用同一目录，单条缓存原子替换，未写完的临时文件不会被当成有效结果。
后续epoch仍需读取雷达数据并训练模型，跳过的只是提框和配对计算，不能承诺具体加速比例。

第一轮建议9项主训练：T00/T01/T02/T03/T04/T07/T10/T11/T12。
其余7项按研究问题补做，不要求把所有消融放进第一轮。

## 可以复用什么

原始雷达NPZ、LiDAR GT、位姿和20%划分保持不变；GT不反号。
P01/P02监督和P05/P06区域配对发生在主干前，已经是完整`kradar_physical_v2`产物且协议一致时可复用。
需要核对manifest中的坐标约定、角度表/标注指纹、候选框/门限、采样预算；smoke产物不能替代正式缓存。
旧正号产物需要重建。P03本轮默认训练新ranker，P04重新评分；旧ranker权重不自动拷贝。

V01～V11离线报告在相同v2协议下可复用，改变原始选点、K/半径、提框/匹配参数才需重算。
V01/V10和此前scene3报告是test场景的历史诊断，不能用于选择新参数。
power配对采用用户固定的C门限；旧B缓存不能用于本次实验；
若修改门限，应同步新缓存和配置，V04不会自动修改训练设置。

## 验证、恢复与记录

```bash
python tools/temporal/run_geometry_v2.py val T01 \
  --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/best_SSC_mean_epoch_N.pth
python tools/temporal/run_geometry_v2.py resume T01 \
  --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/latest.pth
python tools/temporal/run_geometry_v2.py collect
python tools/temporal/run_geometry_v2.py table
```

将N替换为实际文件名。验证仍用val20，按验证SSC_mean选择best；最终独立测试另固定协议。
resume只用于本轮训练中断，入口检查checkpoint的投影版本并拒绝旧主干checkpoint，单帧也不例外。
`collect`只扫描新训练目录。CSV本轮指标从空白开始，旧表行保留在`historical_record`列，不混入新模型成绩。

同一个新KNN checkpoint开/关融合：

```bash
python tools/temporal/run_geometry_v2.py ablate-knn \
  --checkpoint work_dirs/geometry_v2/projection_fixed_20261112/knn/best_SSC_mean_epoch_N.pth
```

结果在新检查目录的`knn_fusion_ablation_<checkpoint名>/REPORT.md`。
off保留联合训练过的主干，不能代替独立T00；旧投影权重不纳入本轮正式消融。

完整GT生成和坐标修正依据见[投影修正报告](../../outputs/projection_chain_audit/20261112_fixed/REPORT.md)。
CPU几何、权重结构兼容性通过不等于GPU整网已验证，也不保证重训后IoU提升。
