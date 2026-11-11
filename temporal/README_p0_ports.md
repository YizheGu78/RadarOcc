# P0 时序模块移植：CenterFormer 与 RODNet TDC

所有新代码和配置都在 `temporal/`。不修改 `projects/`、原模型、旧时序模块、
数据集、旧配置或训练脚本；旧方案继续按原命令运行。

## 接入位置与数据

P0 沿用你们现在的含义：每个 range 取点后、Range Attention 前。
原模型将这个入口命名为 P1。新 `RadarOccP0Temporal` 子类只安装一个独立
`nn.Module` 和已有 `P1TopKAdapter`，原 `extract_pts_feat` 继续执行后续网络。
新配置通过 `plugin_dir='temporal/p0_plugin/'` 注册子类，不修改原注册文件。

输入四帧按 oldest → current 排列。当前与历史只读取八维描述符中的
0、1、2、6、7 五个连续通道；不使用 3:6 Doppler bin ID，也不解释速度。
这只保证新增时序分支不依赖 Doppler；后续原 RadarOcc 仍读取原描述符。

实际 R/A/E bin 表 → 雷达 XYZ → 使用现有外参和位姿做完整 SE(3) 自车对齐
→ 当前雷达坐标系中的高度分层 XYZ 特征体。变换为
`inverse(P_current @ E) @ (P_history @ E)`，只应用一次。
每个 XYZ voxel 对连续描述符的 signed-log1p/80 取均值，空 voxel 有独立 mask。
默认 XY 目标范围 x=[-4,84)、y=[-64,64)，cell=0.8m；XY padding 到四的倍数后，
实际 x=[-4,85.6)、y=[-64,64)，112×160 网格，使各空间尺度共享物理范围。
Z 为当前雷达坐标系中的 [-4,8)m、1m 一层，共12层；高度界限与分辨率可在
`height_bounds`、`height_cell_size` 配置。字段形状为 [B,T,C,Z,H,W]，不再沿高度平均。
范围外的当前点保持原输入，日志 `p0_queries_in_volume` 可检查覆盖数。
这保留了高度分层和插值信息，不等于复现旧 Z0–Z3 的框间高度映射。

所有稀疏特征采样都同时插值 `feature * observation` 与 `observation`，
结果除以有效观测权重；低于1e-4或位于空间范围外的采样无效。
真实观测的零值仍占观测权重，缺失 voxel 不参与均值。
CenterFormer 的 softmax 还排除无效空间候选；多尺度池化也只平均有观测的 voxel。

两种模块最终都在当前点位置采样潜在特征，然后用共有的零初始化输出层
产生五通道有界相对校正：

```
correction = 0.5 * tanh(output(mixed))
out_cont = current_cont + current_cont * expm1(correction)
```

初始化精确等于当前帧输入；没有有效历史、采不到历史支持或点在 BEV 范围外，
精确回退。保留当前三个 Doppler ID、点坐标、点数、零值和符号。
有限原始功率的常见量级约 1e34，可在 FP32 下处理；所有时序计算禁用 AMP。
仍使用原 occupancy 损失，原网络和新增模块共同从随机初始化训练。
没有框匹配、离线框缓存、目标 ID、GT query 或新增运动监督。

## CenterFormer

目录：`centerformer/`；官方仓库 https://github.com/TuSimple/centerformer
固定源提交 `5a949b88ed7bb15aafb39bf78c95f1452063ebea`。

移植源：

- `det3d/models/ops/modules/ms_deform_attn.py`：投影、偏移、权重、径向初始化。
- `det3d/models/ops/functions/ms_deform_attn_func.py`：PyTorch 参考采样核心。
- `det3d/models/necks/rpn_transformer.py`：当前特征产生空间 gate 加权历史的逻辑。

前两个原文件原样保存在 `upstream/*.py.txt`，不会导入旧 CUDA 扩展。
MIT 和采样核心的 Apache-2.0 许可证分别保存在目录中。

流程：共享单层编码 → 当前图的 avg/max 通道池化产生空间 gate →
加权历史、拼接和时间编码形成 query 上下文 → 当前已选 P0 点作为 query →
当前三尺度与三张历史图作为 6 个 source levels → 每头每层预测 4 个采样点
和 softmax 权重 → 带观测归一化的三线性采样聚合 → 残差、LayerNorm、FFN → P0 适配器。
默认四个头、32维、单层 temporal block。偏移单位为每层 feature cell，
`align_corners=False`，初始采样在当前 query 高度沿各头 XY 方向逐渐远离参考点，
XYZ 三个偏移分量都可学习。三个当前尺度只缩小 XY，始终保留全部 Z 层。
空间 gate 和 query 上下文使用核高为1的 Conv3d，各高度层共享二维卷积权重，
不混合不同高度。查询位置编码从二维扩展为三维。

这是 temporal-only 移植，不是完整检测器复现：不移植检测热图、GT候选辅助、
框回归、query 间全局 self-attention 或原 LiDAR backbone。
不做官方八帧配置中的四帧组内累积；四帧分别保留。参考点是当前点的物理 XYZ。
官方二维采样扩展为高度分层特征体上的三维采样，未修改保存的上游源码快照。
无独立对应点监督，偏移通过 occupancy loss 学习。时间编码用相对帧数 / 3，
不是秒；`timestamps` 不参与计算。

## RODNet TDC

目录：`rodnet/`；官方仓库 https://github.com/yizhou-wang/RODNet
固定源提交 `44fd61fa7b433b920469bd8455a54d5676e57fd8`。
源文件 `rodnet/ops/dcn/deform_conv_3d.py`，原样保存在 `upstream/`，MIT许可证在目录中。

保留 `DeformConvPack3D` 的逻辑：Conv3d 从时空输入预测每个核位置的二维空间
偏移，offset 层零初始化，逐采样点按观测权重归一化后按卷积权重求和。
旧 C++/CUDA 扩展替换为原生 PyTorch `grid_sample`，支持当前 PyTorch 的自动求导。
偏移明确按每个 `(time,ky,kx)` tap 的 `(dy,dx)` 排列。

流程：共享单层编码 → 每个高度层的 [B,C,T,H,W] → offset Conv3d →
时序可变形卷积 → GELU → 按当前 P0 点 XYZ 在分层输出体中采样 → P0 适配器。
实际将 B×Z 作为批次维，各层共享 TDC 参数；不先把 Z 压掉。
默认核为 4×3×3，覆盖三帧历史与当前。
时间偏移固定，不学习时间坐标。只输出当前时刻，不使用未来帧。
不移植 M-Net、多 chirp 输入、Hourglass、ConfMap 和检测后处理。
空间坐标由原 RA 改为各物理高度层的 XY；因此这是 TDC 的 P0 改造版。
保留原二维空间偏移，TDC 本身不预测跨高度位移；输出按 query Z 在相邻层
做观测归一化插值。它不会把整根高度柱合并成一个特征。
TDC 把输入当固定间隔时间槽；无速度输入和额外时间编码。

无观测 mask 的兼容接口中，初始化 TDC 零偏移等于相同权重的普通 Conv3d；
生产路径提供 mask 来避免空洞稀释特征。整个 P0 插件因最终零输出层
保持 identity。偏移本身不受额外限幅。空历史 mask 禁止占位帧进入 feature map，
回退 mask 使用与 TDC 完全相同的历史采样偏移。

## 运行

从仓库根目录执行，两种方案输出到不同目录；不会启动或恢复旧 Z 实验。

```bash
bash temporal/run_p0.sh centerformer train
bash temporal/run_p0.sh rodnet train
```

默认固定 20% 数据、15 epochs、seed=42、AdamW lr=3e-4、B=1，每轮验证，
按 SSC_mean 保存最佳权重。训练可显式恢复：

```bash
bash temporal/run_p0.sh centerformer train work_dirs/centerformer_p0_height_20pct_scratch_seed42/latest.pth
bash temporal/run_p0.sh rodnet train work_dirs/rodnet_tdc_p0_height_20pct_scratch_seed42/latest.pth
```

验证同一 20% held-out 集：

```bash
bash temporal/run_p0.sh centerformer val /path/to/checkpoint.pth
bash temporal/run_p0.sh rodnet val /path/to/checkpoint.pth
```

新训练日志输出 `p0_used_history`、`p0_queries_with_history`、
`p0_mean_abs_offset_cells`，以及 height slices、source levels / temporal taps。
这些键不含 loss，不参与优化。旧 Z 的 gate/贡献数值不可直接横向比较。

## 验证

```bash
source use_radarocc5060.sh
python -m unittest discover -s temporal/tests -v
python -m temporal.check_p0 --method centerformer --device cpu
python -m temporal.check_p0 --method rodnet --device cpu
```

`check_p0` 读取真实四帧雷达、原 P0 取点和位姿，不读 occupancy GT；
检查 identity、连续通道输出、两步更新后的 offset 梯度、源输入不变。
它用描述符目标检查计算图，不等价于完整 RadarOcc occupancy 训练或 IoU 验证。
实际 GPU 显存、吞吐量与收益必须在 CUDA 可用时测量，不能从 CPU 检查推断。

回归测试包括：稀疏角点1不再被稀释成0.25/0.125；观测零值仍参与均值；
错误位姿必须改变非零融合输出，正确补偿恢复结果；同XY不同Z的回波不混合；
CenterFormer的Z采样偏移得到梯度；无历史回退、Doppler独立性和训练/推理一致性。
几何测试使用非均匀输出权重，并断言融合确实改变原输入，避免 LayerNorm 零和
让恒定输出头产生假通过。

高度分层版使用新的 `*_p0_height_*` 工作目录；不要恢复之前高度池化版的权重。
输入体和偏移维度已有变化，需要重新训练；旧 Z 实验和原配置不受影响。
GPU 显存与速度需要重新测量，不能沿用高度池化版数据。

2026-11-10 修正后的验证：13 项新模块测试、10 项旧 Box-Z 测试、27 项旧
temporal 测试通过。两种模块均在真实 val index=10 的四帧、每帧43,750个
P0 点上完成 CPU 两步更新检查，初始化精确 identity，梯度有限且偏移层有非零
梯度。默认体积内有26,893个当前点，CenterFormer/TDC分别有26,704/26,469个
点取得历史支持；其余点保留原输入。这些是该帧的工程检查，不能代表验证集统计。
当前会话无可用 CUDA，完整 RadarOcc GPU 训练、显存和 IoU 尚未验证。
