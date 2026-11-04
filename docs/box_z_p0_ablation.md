# P0 入口的 Z0–Z3 高度映射消融（20%）

本方案的 P0 指用户指定的 **每个 range 取点后、Range Attention 前**，即旧代码的 P1
入口。没有 P4 分支，不调用 KNN、Top-64、学习 Top-8 或旧 temporal plugin。

## 一次运行

从仓库根目录执行：

```bash
bash run_box_z_p0_20pct.sh \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth
```

默认输出 `work_dirs/box_z_p0_20pct`，每组 10 epochs、seed=42、AdamW lr=1e-4。
可指定新的输出目录和 epoch 数：

```bash
bash run_box_z_p0_20pct.sh \
  work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth \
  work_dirs/box_z_p0_screen_v2 10
```

脚本自动激活 radarocc5060，依次：

1. 检查固定 20% 子集；两份都不存在时生成，已有文件不覆盖。
2. 生成/复用训练与验证的 GT-free 框关联缓存。
3. 真实帧预检：初始化预测完全等于关闭融合时的预测；冻结权重；Z3 数据梯度贯通。
4. 在同一验证子集评价单帧 checkpoint，保存基线 IoU。
5. 顺序训练 Z1、Z2、Z3、Z0；每个 epoch 验证，按 SSC_mean 保存最佳模型。
6. 每组完成及脚本退出时汇总全部 IoU 和完整验证轮次的诊断。

若中断，同样命令可复用完整缓存、跳过完成的组、从未完成组的 latest.pth 恢复。
实验目录记录初始化权重、数据、配置和关键实现的 SHA256；这些发生变化时须换目录，
避免比较表混入不同实验。缓存中断可以 `--resume`，只有 complete manifest 才可训练。
首次离线提框仍需时间；训练不再逐帧提框或执行框匹配。

## 数据与预算

使用现有连续窗口子集：

- train：`kradar_dict_train_official_temporal_doppler8_20pct.pkl`，当前为 2,274 帧，17 场景。
- val：`kradar_dict_val_official_temporal_doppler8_20pct.pkl`，当前为 211 帧，2 场景。

两者都是各场景的一段连续 20% 窗口，历史只从同一子集构造。窗口开头缺失历史以
占位帧补形状、mask 禁用，不跨场景，也不把测试集用于选模型。
同一 B=1、同一数据、相同 epoch 数意味着四组具有同样的优化步数。
默认 10 epochs 为每组 22,740 个训练样本步，不承诺其训练时长。

初始化只用同一个单帧 RadarOcc checkpoint；Z3 不加载训练过的 Z2 时序权重。
四组共有参数的初始值与随机状态相同，Z3 额外的 Linear(7,2) 共 16 参数且初始化为零。

## 框与坐标

`cache_box_z_pairs.py` 只读 NPZ、物理 bin 表、PKL 中的 pose/时间/序列信息；**不读 GT
bbox 或目标 ID**。反射区域提取完全固定：P1 前 175×250 原始点、power channel 2，
seed percentile 97、seed MAD 3、XY merge、gap 0.5/0.5 m、初始尺寸6/3/3 m、
最大尺寸20/6/6 m、保持 XY 尺寸上限。XY 模式的高度尺寸不被6m硬裁剪，沿用原规则。

在历史区域自车对齐后的 AABB 中，固定 Y 门限4/5/3 m、XY中心距离门限4/6/8 m，
纯中心距离权重1/0，每个当前框每个历史帧最多关联一个区域。允许多对一。
缓存目录 `data/box_z_pairs/{train20,val20}`，四组共享，不保存不同 Z 的采样点。
缓存记录来源 fingerprint、框参数、匹配 mask 和当前雷达到历史雷达的逆变换。

当前 Query 是实际已选中的原始点。多个框覆盖时，按当前框体积升序、再按区域顺序
唯一归属。框外或无匹配点原样输出。

先在当前雷达坐标中做框间 inverse mapping，再**仅一次**乘以
`inverse(P_history @ radar_to_lidar) @ (P_current @ radar_to_lidar)`，
直接得到历史雷达坐标。不再重复做 LiDAR 转换。

XY 始终按历史/当前 AABB 的长宽比例映射。Z 策略如下：

| 模式 | 历史采样高度（当前雷达坐标） |
|---|---|
| Z0 | 历史框内8个等距高度的区间中点，分别插值后对有效高度取均值 |
| Z1 | 当前 Query 的物理 Z，`z_h=z_t` |
| Z2 | `c_hz + (h_h/h_t)*(z_t-c_tz)` |
| Z3 | `c_hz + (h_h/h_t)*exp(alpha)*(z_t-c_tz) + beta` |

Z3 使用约定的7维框几何输入；alpha=log(2)*tanh(a)、beta=0.5*tanh(b) m。
Linear 零初始化使坐标公式等于 Z2。损失为原项目归一化 Occupancy Loss，加上
每个有效匹配框对等权的 `0.001*mean(alpha²) + 0.001*mean(beta²)`。
这两个正则在父模型的 loss_norm **之后**加入，避免被分别归一化为常数1。

## P0 插值与融合适配

P0 不是密集可学习特征体，不能套用 P4 的粗球坐标采样近似。此处用实际物理
range/azimuth/elevation bin 表构造 `[1,8,R,A,E]` 描述符场及单独观测 mask。
缺失网格不是真实零值观测；同一 cell 多个回波先平均。

五个连续通道使用 signed-log1p / 80，三个 Doppler bin 描述通道 / 64。
对归一化描述符和观测 mask 都做 grid_sample，输出除以插值到的观测权重，
只有权重>=1e-4且坐标在物理 bin 范围、关联框范围内才有效。
这是原始规则网格上的8角三线性插值，不是 KNN 选8点；`align_corners=False`、
`padding_mode=zeros`，tensor D/H/W=R/A/E，grid 分量顺序 E/A/R。
非均匀物理 bin 通过分段线性连续索引处理，Z3 坐标梯度保留。

每个 Query 每个历史帧得到一个8维描述符，经过64维、4头的时间 Cross-Attention，
只有三帧时间 bias，没有额外空间 bias 或 Doppler Guidance。历史 bin 描述通道仅作为
线性投影的数值输入，插值后的数值不作为离散 bin ID 写回当前点。

原始连续通道可能达到1e34量级，因此采用四组共用的有界相对 Gate/Residual 适配器：

```
delta = 0.5 * tanh(Output(H))
F_out_cont = F_current_cont + F_current_cont * gate * expm1(delta)
```

三个当前 Doppler bin ID、点坐标、点数保持原值；连续通道的符号与零值也保持。
输出投影零初始化使四组初始都是精确 Identity。所有历史无效时同样精确回退。
**这不是在1e34原始功率上直接相加64维潜在特征。**四组共享此适配器，差异只有Z策略。
Z0 的插值次数为其余组8倍，不能当作等算量消融。

冻结原 RadarOcc 参数，BN与Dropout保持eval；但 spconv 的 SparseConvolution 在训练时
必须使用其可微 training kernel，否则其 eval kernel 会截断输入梯度。
这个开关不解冻任何卷积权重、不更新BN统计；验证仍使用eval路径。
冻结后续网络不会省去其输入梯度计算，所以不能据此保证极短训练时间。

## 输出与比较

根目录下：

| 文件 | 内容 |
|---|---|
| `single_frame_metrics.json` | 同一验证子集上的原模型 IoU |
| `preflight.json` | 真实帧 Identity、梯度、采样检查 |
| `epoch_iou.csv` | 四组每个 epoch 的全部五项 IoU |
| `comparison.csv/json/md` | 各组验证 SSC_mean 最佳的 epoch 和指标、最后epoch SSC_mean |
| `sampling_diagnostics.csv` | 每个完整验证轮次的有效率、Gate、贡献、Z3 scale/shift分布 |
| `Z*/eval_diagnostics.jsonl` | 每个验证帧的原始计数与逐框对参数，含epoch和eval_run |
| `Z*/best_SSC_mean*.pth` | 项目 EvalHook 保存的最佳完整模型 |
| `Z*/latest.pth` | 含优化器状态的恢复入口 |

**SSC_mean 沿用当前项目定义：Background 与 Foreground IoU 的均值，不含 free。**
IoU CSV 保留原项目0–1单位和原评估mask/类定义；不把数值乘100再混入日志。
采样有效率按总有效采样数/总尝试数汇总，Z0与其他组分母不同；同时比较有效时间token数。
逐框参数原值保存在JSONL；汇总mean、median、P95、median/P95绝对shift。
Gate与相对贡献可区分“采样无效”和“模型抑制历史”。

单独重建比较表（不训练）：

```bash
python tools/temporal/summarize_box_z_ablation.py --root work_dirs/box_z_p0_20pct --val-frames 211
```

测试：

```bash
python -m unittest discover -s tests -p test_box_z_p0.py -v
```

仓库只提供实现与工程验证；正式四组完成前，IoU 对照是待测结果，不能预填推测值。
