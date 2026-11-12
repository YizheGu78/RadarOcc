# RadarOcc投影修正后：从头训练入口

完整安排表仍在原位置：[EXPERIMENTS.md](EXPERIMENTS.md)。
它保留原V00～V11、P01～P07、T00～T14编号，新增V12/V13/V14公共检查与T15无对齐memory对照。
每项写明目的、操作、前置、与谁比较、GT用途、结果位置，以及训练/验证/恢复命令。

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

- power：核对V03/V05，按需要V04/V09；准备P05/P06，再训练T07（Z1）。T06/T08/T09比较Z0/Z2/Z3。
- CenterFormer：V07模块检查 → T10；RODNet TDC：V08模块检查 → T11。
- RA-ADTA：T12；T13替换搜索方式。
- T14在T01上增加近场power屏蔽。
- T15补回无显式自车对齐的历史memory方案，与T00比较历史特征收益；它与KNN的结构也不同。

第一轮建议9项主训练：T00/T01/T02/T03/T04/T07/T10/T11/T12。
其余7项按研究问题补做，不要求把所有消融放进第一轮。

## 可以复用什么

原始雷达NPZ、LiDAR GT、位姿和20%划分保持不变；GT不反号。
P01/P02监督和P05/P06区域配对发生在主干前，已经是完整`kradar_physical_v2`产物且协议一致时可复用。
需要核对manifest中的坐标约定、角度表/标注指纹、候选框/门限、采样预算；smoke产物不能替代正式缓存。
旧正号产物需要重建。P03本轮默认训练新ranker，P04重新评分；旧ranker权重不自动拷贝。

V01～V11离线报告在相同v2协议下可复用，改变原始选点、K/半径、提框/匹配参数才需重算。
V01/V10和此前scene3报告是test场景的历史诊断，不能用于选择新参数。
power配对默认固定B门限（Y=4/5/3m，中心距离=4/6/8m），并不宣称它最优；
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
