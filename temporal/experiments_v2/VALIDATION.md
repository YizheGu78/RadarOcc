# 本地验证记录

- 坐标专项7项通过：实际K-Radar角度表与独立官方公式、非对称GT归属、位姿/外参组合、Box-Z正反索引与插值、旧权重拒绝、全部15份配置解析。
- 相关旧模块回归117项通过：ranker15、Box-Z10、移植模块13，以及时序/Query Budget/RA-ADTA/区域匹配等79项。
- CenterFormer和TDC均使用真实val四帧，每帧43,750点；初始化identity、两步反传、Doppler ID不变、源输入不变、offset梯度非零。
- 实际旧ranker和旧Box-Z缓存被v2入口拒绝；新坐标小样本ranker可加载。
- 小样本数据流程通过：train/val监督准备、ranker一轮CPU短训练、GT-free框缓存、区域匹配、区域A/B/C严格GT审计、Doppler方向和Q32/Q64覆盖。
- 小样本监督/训练位于 data/geometry_v2/smoke 和 work_dirs/geometry_v2/smoke，不混入正式目录。
- 各CLI参数解析已检查，实验依赖和15份配置文件均存在；git diff --check通过；projects/与third_party/无代码改动。

当前会话 `torch.cuda.is_available()` 为False，因此尚未执行完整GPU occupancy训练、显存/吞吐或IoU验证。
CPU短训练不代表正式选点器或下游模型已完成重训，实验表的正式指标留空。
原单帧网络自身的极坐标到occupancy映射未在本次修复中认证。

详细机器记录见 [validation.json](validation.json)；日志与小样本报告位于仓库 outputs/geometry_v2/。

## 2026-11-11：KNN模型构造失败修复

v2的8份TemporalPlugin配置曾含有未被MMCV继承合并消除的`_delete_`，
实际构造会报`TemporalPlugin.__init__() got an unexpected keyword argument '_delete_'`。
已从这些v2配置中移除该标记，原始RadarOcc代码与旧配置未修改。

- 更新后的坐标/配置专项8项通过，新增真实KNN模块构造检查；所有v2模型配置都检查未残留合并标记，TemporalPlugin参数与构造函数签名一致。
- 完整KNN模型CPU构造通过（94,646,840参数），`init_weights()`通过。
- 实际train/val数据集构造通过，样本数分别2,274/211。
- 原检测器初始化固定网格写死`.cuda()`；CPU构造检查仅在进程内临时将这些网格保留于CPU，未修改源文件。没有执行GPU前向、反向或训练。
- 无需重跑V11或其他已完成离线审计；仍使用原T01启动命令。

上述是追加的启动配置检查；前面的117项旧模块回归为此前记录，本次没有重复执行。

## 2026-11-12：同一KNN checkpoint的融合开/关入口

- `run_geometry_v2.py ablate-knn --checkpoint ...`从checkpoint内保存配置构造同一模型；两次均读取相同val20四帧输入，off仅旁路时序输出。
- 专项9项通过，新增旁路前后权重与state_dict键不变、输出为当前描述符、遇到异常仍恢复原forward的检查。
- epoch4最佳checkpoint在CPU构造检查中严格加载成功（全部键匹配），val数据集211帧。保留spconv的state_dict版本元数据，避免错误转置其卷积核。
- CPU构造仍仅临时绕过固定网格的`.cuda()`；当前会话CUDA不可用，尚未执行这两次GPU整网验证，未产生on/off指标。
- 几何一致性复核另见`outputs/geometry_v2/alignment_recheck/REPORT.md`，其中说明静态GT由同一套位姿生成这一限制。
