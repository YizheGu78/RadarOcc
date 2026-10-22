# 第一阶段：等体积方向性方框对比

只运行离线 Token 搜索，不调用 RadarOcc，不训练，不使用 Doppler 或 Attention，也不运行球形 KNN。
复用 `audit_dynamic_objects.py` 的 P1、Ego Alignment、bbox / Tracking ID 归属与有效查询定义。
原 `match` / `occupancy` 命令保留，新增独立 `box` 命令一次运行三个对照。

```bash
cd /home/user1/projects/RadarOcc
conda activate radarocc5060
python tools/temporal_vis/audit_dynamic_objects.py box \
  --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
  --gaps 1 2 3 --box-lengths 4 6 8 \
  --box-reverse 0.5 --box-y-half 1.5 --box-z-half 1.0 --k 8 \
  --output-dir outputs/dynamic_object_audit/scene3_box_comparison
```

## 三组区域和排序

所有范围相对**当前 Query**，坐标为当前雷达坐标，历史点已 Ego Alignment。
GT 方向取 `sign(历史框几何中心经对齐后的 X - 当前框几何中心 X)`。
负值向 X 负方向找历史，正值向 X 正方向找历史；不能用车头朝向或同向 / 对向类别替代该符号。
GT 位移的绝对值不改变框尺寸，也不平移 Query。横向、纵向位移会写入明细以便复查。

| 方法 | X 范围（历史位于负方向时） | Y 范围 | Z 范围 |
|---|---|---|---|
| symmetric | [−(L+0.5)/2, +(L+0.5)/2] | ±1.5 m | ±1.0 m |
| correct_direction | [−L, +0.5] | ±1.5 m | ±1.0 m |
| wrong_direction | [−0.5, +L] | ±1.5 m | ±1.0 m |

历史位于正方向时，正确与反向框的 X 范围互换。L 对三个历史槽依次为 4 / 6 / 8 m。
三组体积在同一历史槽完全相同，依次为 27 / 39 / 51 m³；对称框的 X 半长为
2.25 / 3.25 / 4.25 m。边界按闭区间处理。

实现先用归一化 KDTree 的 L∞ 查询提取整个框内候选，再按到**原 Query** 的欧氏距离
选最多 8 个；同距离用原 P1 索引打破平局。不按到框中心的距离排序，不用标签筛候选，
不改变 43,750 个 P1 token 的读取规则。背景和框归属冲突的 Token 也可进入候选集，
但不会被算作正确同目标回波。半径参数不参与本实验。

## 可比样本和筛选

- 三组严格使用相同的当前查询、历史帧、GT 目标归属和有效分母。
- 当前目标中心必须位于雷达前方；没有同 ID 历史框时无法提供方向 Oracle，三组共同排除。
- `abs(GT dx) <= --min-direction-dx` 时方向不明确，三组共同排除；默认 1e-6 m，仅排除近零值。
- 可显式使用 `--min-direction-dx 0.1` 检查小位移 / 标注抖动的敏感性，但这会改变样本分母，
  应输出到不同目录并单独报告，不能和默认结果混在一起。
- `--min-longitudinal-cos` 默认 0，不额外筛除横向运动；若只研究沿 X 轴运动的目标，可显式设为
  0.8 等阈值，约束 `abs(dx)/sqrt(dx²+dy²)`。阈值需事先指定并独立报告，避免按结果挑样本。
- 同 ID 框存在但没有同目标历史 P1 时，计为 `n_no_history`，不进入 Recall / Coverage 分母。
- 当前查询落入多个扩张框时，其目标归属不明确，不作为查询真值。

默认历史槽与 TemporalKradarDataset 一致，使用当前 split 的前三个历史位置。
`history_slot` 表示 t-1 / t-2 / t-3，`frame_gap` 记录实际 order 差，`delta_t_sec` 为真实时间差。
固定 4 / 6 / 8 m 依槽位指定，不按 GT 位移或实际时间差自适应缩放。
如果显式指定 `--history-mode scene-order`，`--gaps` 改为场景 order 差。
`--box-lengths` 必须与 `--gaps` 一一对应。

## 指标与输出

| 指标 | 定义 |
|---|---|
| Candidate Coverage | 框内至少有一个同目标历史 Token 的查询数 / 有效查询数 |
| Same-object Recall@8 | 最终最多 8 个候选中至少有一个同目标 Token 的查询数 / 有效查询数 |
| Selected Same Precision | 选中的同目标 Token 数 / 所有选中 Token 数，限有效查询 |
| Outside Box | 同目标历史回波存在，但框内一个也没有的查询数 |
| Top-K Miss | 框内有同目标回波，但未进入最近 K 个的查询数 |

必须满足：`n_success + n_topk_miss + n_outside_box = n_eligible`，
以及 `n_covered = n_success + n_topk_miss`；程序逐个对象 / 历史 / 方法断言检查。
汇总使用分子、分母求和后的 micro 指标，零分母留空。

| 文件 | 内容 |
|---|---|
| box_results.md | 三种框 × 三个历史槽的直接对比表 |
| box_summary_by_history.csv | 按历史槽与方法汇总，含各项计数和候选池大小 |
| box_summary_by_direction.csv | 额外区分历史 X 正 / 负方向 |
| box_summary_by_object.csv | 按目标 ID 与历史槽汇总，便于检查结果是否由少数目标主导 |
| box_per_object_history.csv | 每目标 / 历史 / 方法明细，含 GT 位移、框边界、体积和全部计数 |
| box_excluded_pairs.csv | 对三组共同排除的目标历史对及原因 |
| box_comparison.png | Coverage、Recall@K、Precision 三面板对比图 |
| box_protocol.json | 方向定义、实际帧列表、样本数、源代码哈希 |
| run_metadata.json | 完整运行参数 |

结果只能说明 bbox 近似归属、GT 方向完全可用情况下的几何候选效果。
重复目标、相邻帧与查询并非独立样本；这里不输出显著性或因果结论。
方向 Oracle 仍可能受标注抖动或 ID 错误影响。未验证 Doppler 推断方向的准确性，也未验证 Occupancy 改善。

## 测试

```bash
python -m unittest discover -s tests -p test_directional_box_matching.py -v
python -m unittest discover -s tests -p test_dynamic_object_audit.py -v
```

包括等体积与方向镜像、去除自车运动和旋转后的 GT 方向、方向筛选能改善固定 K 的例子、
Coverage 高但 Top-K 依然漏选的例子、保持原 Query 距离排序、KDTree 与穷举逐项对照、
闭区间边界及同距候选的确定性选择。

## 正确方向框的 K=8 / 32 / 64 对比

这是离线候选匹配实验，无需模型训练。新增 `--box-methods` 可只运行正确方向框；
默认仍运行原有三组对照。以下命令保持相同样本和框参数，仅改变 K：

```bash
conda activate radarocc5060
for audit_k in 8 32 64; do
  python tools/temporal_vis/audit_dynamic_objects.py box \
    --box-methods correct_direction \
    --scene 3 --start-order 80 --end-order 200 --stride 5 --max-frames 25 \
    --gaps 1 2 3 --box-lengths 4 6 8 \
    --box-reverse 0.5 --box-y-half 1.5 --box-z-half 1.0 \
    --k "$audit_k" \
    --output-dir "outputs/dynamic_object_audit/scene3_correct_box_k${audit_k}" || break
done
```

各目录的 `box_summary_by_history.csv` 与 `box_results.md` 报告 Recall@对应的 K，
而非固定 Recall@8。Coverage、Outside Box 和有效查询应保持不变，Recall 随 K 非递减。
`mean_selected` 是每个有效查询实际选中的平均候选数；框内不足 K 时不会补点或复制候选。
Precision 的分母是实际选中的候选数，不能假设恰好等于 K×有效查询数。
K=32/64 的匹配改善只表示更大的候选池包含更多正确回波，不代表仍选 8 个时会改善，
也不代表 Attention、时序融合或 Occupancy 精度已改善。
