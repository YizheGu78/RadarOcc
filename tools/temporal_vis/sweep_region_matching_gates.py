#!/usr/bin/env python3
"""Reuse one proposal extraction to sweep matching gates with a temporal holdout.

The reference report fixes all extraction, GT and matching-space settings.
GT is only used for scoring/tuning, never candidate gating or match ranking.
This is scene-specific exploratory tuning, not independent test evaluation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis.benchmark_region_matching import FrameData, WEIGHTS, history_regions


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(n, d):
    return n/d if d else None


def collect(config):
    args = SimpleNamespace(**config)
    for key in ('ann_file', 'bbox_root', 'bins', 'output_dir', 'extrinsic'):
        if getattr(args, key, None) is not None:
            setattr(args, key, Path(getattr(args, key)))
    data = FrameData(args)
    data.index()
    queries = []
    dims = 2 if args.match_space == 'xy' else 3
    for scene in args.scenes or sorted(data.scenes):
        frames = data.scenes[scene]
        chosen = [p for p, f in enumerate(frames)
                  if args.start_order <= int(f['order_in_scene']) <= args.end_order]
        chosen = chosen[::args.stride][:args.max_frames_per_scene or None]
        for pos in chosen:
            order = int(frames[pos]['order_in_scene'])
            current = data.frame(scene, order)
            for slot in (1, 2, 3):
                if pos < slot:
                    continue
                history_order = int(frames[pos-slot]['order_in_scene'])
                previous = data.frame(scene, history_order)
                candidates = history_regions(current, previous, data.extrinsic)
                for region in current['accepted']:
                    actual = current['labels'].get(int(region['id']))
                    centers = np.asarray([h['aligned_center'] for h in candidates]).reshape(-1, 3)
                    sizes = np.asarray([h['aligned_size'] for h in candidates]).reshape(-1, 3)
                    distance = np.linalg.norm(centers[:, :dims]-region['center'][:dims], axis=1)
                    dy = np.abs(centers[:, 1]-region['center'][1])
                    size_error = np.mean(np.abs(np.log(np.maximum(sizes[:, :dims], 1e-6)/
                        np.maximum(np.asarray(region['box_size'])[None, :dims], 1e-6))), axis=1)
                    ids = np.asarray([h['id'] for h in candidates], dtype=int)
                    # Exact reference tie-break: cost, distance, region ID.
                    order_ids = np.lexsort((ids, distance))
                    labels = [previous['labels'].get(h['id']) for h in candidates]
                    true = np.array([actual is not None and label == actual for label in labels])
                    known = np.array([label is not None for label in labels])
                    queries.append(dict(scene=scene, order=order, slot=slot,
                        history_order=history_order, region=int(region['id']), actual=actual,
                        dy=dy[order_ids], distance=distance[order_ids], size=size_error[order_ids],
                        true=true[order_ids], known=known[order_ids]))
            print(f'Extracted scene={scene} order={order} accepted={len(current["accepted"])} '
                  f'labeled={len(current["labels"])}', flush=True)
    return queries, list(data.diagnostics.values())


def pack(queries):
    n = len(queries)
    width = max((len(q['dy']) for q in queries), default=0)
    arrays = {k: np.full((n, width), np.inf) for k in ('dy', 'distance', 'size')}
    arrays.update({k: np.zeros((n, width), bool) for k in ('true', 'known')})
    for i, query in enumerate(queries):
        for key in arrays:
            arrays[key][i, :len(query[key])] = query[key]
    arrays['scorable'] = np.array([q['actual'] is not None for q in queries])
    arrays['order'] = np.array([q['order'] for q in queries])
    return arrays


def evaluate(a, y, center, wd, ws, size_scale, selected):
    scorable = a['scorable'] & selected
    truth = a['true'].any(axis=1)
    y_allowed = a['dy'] <= y+1e-10
    allowed = y_allowed & (a['distance'] <= center+1e-10)
    predicted = allowed.any(axis=1)
    # Avoid 0*inf for padded candidates.
    cost = np.zeros_like(a['distance'])
    if wd:
        cost += wd*a['distance']/center
    if ws:
        cost += ws*a['size']/size_scale
    cost = np.where(allowed, cost, np.inf)
    if cost.shape[1]:
        picked = cost.argmin(axis=1)
        row = np.arange(len(picked))
        correct = predicted & a['true'][row, picked]
        known_prediction = predicted & a['known'][row, picked]
    else:
        correct = known_prediction = np.zeros(len(scorable), bool)
    true_y = (a['true'] & y_allowed).any(axis=1)
    true_gate = (a['true'] & allowed).any(axis=1)
    c = dict(all_queries=int(selected.sum()), all_predictions=int((predicted & selected).sum()),
        current_labeled=int(scorable.sum()), has_true_history=int((scorable & truth).sum()),
        predicted=int((scorable & predicted).sum()), correct=int((scorable & correct).sum()),
        wrong=int((scorable & known_prediction & ~correct).sum()),
        unscorable_prediction=int((scorable & predicted & ~known_prediction).sum()),
        y_missed_true=int((scorable & truth & ~true_y).sum()),
        center_missed_true=int((scorable & true_y & ~true_gate).sum()),
        true_after_gates=int((scorable & true_gate).sum()))
    c['precision'] = ratio(c['correct'], c['correct']+c['wrong'])
    # Unknown predictions are not asserted false, but count against this lower bound.
    c['precision_lower_bound'] = ratio(c['correct'], c['predicted'])
    c['recall'] = ratio(c['correct'], c['has_true_history'])
    c['f1_lower_bound'] = ratio(2*c['correct'], c['predicted']+c['has_true_history'])
    c['gate_retention'] = ratio(c['true_after_gates'], c['has_true_history'])
    c['mean_candidates_all'] = float(allowed[selected].sum()/selected.sum()) if selected.any() else None
    return c


def best(rows):
    # F1 lower bound first, then recall, then tighter gates. Never tune on holdout.
    return max(rows, key=lambda r: (r['f1_lower_bound'] or 0, r['recall'] or 0,
                                   -r['y_gate']*r['center_gate'], -r['y_gate'], -r['center_gate']))


def write_summary(output, report, metrics):
    rows = report['comparison']
    lines = ['# Scene 3 反射区域匹配门限扫描', '',
        '固定参考报告中的区域提取、XY 合并、XY GT 归属和匹配设置；仅扫描匹配门限与权重。', '',
        f'选参：当前帧 order <= {report["split_order"]}；时间留出：后续当前帧。'
        '历史帧可跨该分界，且相邻帧相关，因此不是独立测试集。', '',
        '主方案固定中心/尺寸权重为 1/0。选参指标为保守 F1：'
        '`2*correct/(predicted+has_true_history)`。'
        '`predicted` 包括匹配到未知 GT 身份的历史区域；只统计 GT 身份已知的当前区域。', '',
        '## 固定纯距离权重的推荐参数', '',
        '| 历史帧 | Y 门限 m | XY 中心距离门限 m | 全段 Recall | 原 Recall | 留出 Recall | 原留出 Recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    pct = lambda value: f'{value*100:.2f}%' if value is not None else 'N/A'
    for s in (1, 2, 3):
        get = lambda config, subset: next(r for r in rows if r['configuration'] == config
                                           and r['subset'] == subset and r['slot'] == s)
        a, b = get('distance_only', 'all'), get('baseline', 'all')
        h, hb = get('distance_only', 'holdout'), get('baseline', 'holdout')
        lines.append(f'| t-{s} | {a["y_gate"]:g} | {a["center_gate"]:g} | {pct(a["recall"])} | '
                     f'{pct(b["recall"])} | {pct(h["recall"])} | {pct(hb["recall"])} |')
    lines += ['', '## 汇总对比', '',
        '| 数据段 | 配置 | 正确/存在正确历史 | 明确错误 | 未知历史身份 | Recall | 条件 Precision | 保守 Precision | 保守 F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for subset in ('all', 'tune', 'holdout'):
        for config in ('baseline', 'distance_only', 'wide_control', 'joint_weight_and_gates'):
            selected = [r for r in rows if r['configuration'] == config and r['subset'] == subset]
            count = {k: sum(r[k] for r in selected) for k in
                     ('correct', 'wrong', 'unscorable_prediction', 'predicted', 'has_true_history')}
            c, w, p, t = (count[k] for k in ('correct', 'wrong', 'predicted', 'has_true_history'))
            lines.append(f'| {subset} | {config} | {c}/{t} | {w} | '
                f'{count["unscorable_prediction"]} | {pct(ratio(c,t))} | {pct(ratio(c,c+w))} | '
                f'{pct(ratio(c,p))} | {pct(ratio(2*c,p+t))} |')
    lines += ['', '`wide_control` 为 Y=20 m、中心距离=100 m，并非推荐参数。', '',
        '原始 GT 评价：Precision 仅统计历史身份已知的预测。保守 Precision 将未知身份预测'
        '也计入分母，不意味着这些未知预测已被证明错误。', '',
        '## 如何理解结果', '',
        '- Y 门限并非越宽越好：放宽会同时纳入其他目标和背景反射区域。',
        '- t-1 的 Y=4、中心距离=4 表示有效圆形距离门限已约束 Y；并非证明独立 Y 窗必须恰好 4 m。',
        '- t-3 的推荐 Y 比 t-2 小，来自此样本上的选择结果，不是车辆运动规律。',
        '- 中心距离使用 XY 欧氏距离；它包含反射区域形状/合并中心变化，不等于物体移动距离。',
        '- 固定区域提取参数下，不同帧同一 GT 可能对应多个反射区域，统计单位是区域/历史帧查询。',
        '- gt-purity=0、min-gt-points=1 是宽松投影归属，不能当作精确点级真值。',
        '- 本实验只支持 scene 3、指定帧段的初步门限选择，正式性能评估需要其他场景验证。', '',
        '## 复现', '', '```bash',
        'python tools/temporal_vis/sweep_region_matching_gates.py \\',
        f'  --reference-report {report["reference_report"]} \\',
        f'  --split-order {report["split_order"]} \\',
        '  --output-dir outputs/region_matching_benchmark/scene3_gate_sweep_new', '```', '',
        'gate_sweep.csv：每组门限、权重、数据段的全部统计。',
        'comparison.csv：原参数、前段选出的参数和宽门限对照。',
        'true_pair_diagnostics.csv：同 ID 历史区域的最小 Y 差与最小 XY 距离（可能来自不同历史候选）。',
        'report.json：参考配置、选参结果、限制说明及全段探索性最优结果。', '']
    (output/'summary.md').write_text('\n'.join(lines))
    # A static research figure, separate from the benchmark artifacts.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, slot in enumerate((1, 2, 3)):
        chosen = next(r for r in report['recommendation']
                      if r['criterion'] == 'distance_only' and r['slot'] == slot)
        a = [r for r in metrics if r['slot'] == slot and r['subset'] == 'tune'
             and r['weight_center'] == 1. and r['y_gate'] <= 10 and r['center_gate'] <= 25]
        ys, ds = sorted({r['y_gate'] for r in a}), sorted({r['center_gate'] for r in a})
        for row, metric in enumerate(('recall', 'f1_lower_bound')):
            grid = np.full((len(ys), len(ds)), np.nan)
            for r in a:
                grid[ys.index(r['y_gate']), ds.index(r['center_gate'])] = r[metric]
            ax = axes[row, col]
            im = ax.imshow(grid*100, vmin=0, vmax=100, origin='lower', aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(ds)), [f'{d:g}' for d in ds], rotation=60)
            ax.set_yticks(range(len(ys)), [f'{y:g}' for y in ys])
            ax.set_xlabel('XY center gate (m)'); ax.set_ylabel('Y gate (m)')
            ax.set_title(f't-{slot}: tuning {metric}')
            ax.plot(ds.index(chosen['center_gate']), ys.index(chosen['y_gate']),
                    marker='*', color='red', markeredgecolor='white', markersize=14)
            fig.colorbar(im, ax=ax, label='%')
    fig.savefig(output/'gate_heatmaps.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split-order', type=int, default=140,
                        help='Tune on orders <= this; later current frames are temporal holdout')
    parser.add_argument('--y-grid', type=float, nargs='+',
        default=[.25, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20])
    parser.add_argument('--center-grid', type=float, nargs='+',
        default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 60, 100])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a new output directory')
    if not all(np.isfinite(v) and v > 0 for v in args.y_grid+args.center_grid):
        parser.error('Gate values must be finite and positive')
    reference = json.loads(args.reference_report.read_text())
    queries, frames = collect(reference['config'])
    if not queries:
        raise ValueError('No queries')
    packed = {slot: pack([q for q in queries if q['slot'] == slot]) for slot in (1, 2, 3)}
    metrics = []
    # Check vectorized evaluator against the user's baseline before sweeping.
    for row in reference['metrics']:
        if row['slot'] == 'all':
            continue
        slot = int(row['slot'])
        a = packed[slot]
        result = evaluate(a, reference['config']['y_gates'][slot-1],
            reference['config']['center_gates'][slot-1], row['weight_center'],
            row['weight_size'], reference['config']['size_scale'], np.ones(len(a['order']), bool))
        for key in ('current_labeled', 'has_true_history', 'predicted', 'correct', 'wrong',
                    'unscorable_prediction', 'y_missed_true', 'center_missed_true'):
            if result[key] != row[key]:
                raise ValueError(f'Reference mismatch slot={slot} weights={row["weight_center"]} '
                                 f'{key}: {result[key]} != {row[key]}')
    print('All baseline counts reproduced for 6 weights x 3 history slots.', flush=True)
    for slot, a in packed.items():
        subsets = dict(all=np.ones(len(a['order']), bool), tune=a['order'] <= args.split_order,
                       holdout=a['order'] > args.split_order)
        for wd, ws in WEIGHTS:
            for y in sorted(set(args.y_grid+[reference['config']['y_gates'][slot-1]])):
                for center in sorted(set(args.center_grid+[reference['config']['center_gates'][slot-1]])):
                    for subset, selected in subsets.items():
                        metrics.append(dict(slot=slot, weight_center=wd, weight_size=ws,
                            y_gate=y, center_gate=center, subset=subset,
                            **evaluate(a, y, center, wd, ws, reference['config']['size_scale'], selected)))
        print(f'Swept slot={slot}', flush=True)
    recommendation, comparison = [], []
    for slot, a in packed.items():
        for criterion, wd in [('distance_only', 1.), ('joint_weight_and_gates', None)]:
            fit = [r for r in metrics if r['slot'] == slot and r['subset'] == 'tune'
                   and (wd is None or r['weight_center'] == wd)]
            choice = best(fit)
            recommendation.append(dict(criterion=criterion, slot=slot,
                y_gate=choice['y_gate'], center_gate=choice['center_gate'],
                weight_center=choice['weight_center'], weight_size=choice['weight_size']))
            for subset in ('tune', 'holdout', 'all'):
                row = next(r for r in metrics if r['slot'] == slot and r['subset'] == subset
                    and all(r[k] == choice[k] for k in ('y_gate', 'center_gate', 'weight_center', 'weight_size')))
                comparison.append(dict(configuration=criterion, **row))
        for label, y, center in [('baseline', reference['config']['y_gates'][slot-1],
                                  reference['config']['center_gates'][slot-1]),
                                 ('wide_control', max(args.y_grid), max(args.center_grid))]:
            for subset, selected in dict(all=np.ones(len(a['order']), bool),
                    tune=a['order'] <= args.split_order, holdout=a['order'] > args.split_order).items():
                comparison.append(dict(configuration=label, slot=slot, weight_center=1., weight_size=0.,
                    y_gate=y, center_gate=center, subset=subset,
                    **evaluate(a, y, center, 1., 0., reference['config']['size_scale'], selected)))
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'gate_sweep.csv', metrics)
    write_csv(args.output_dir/'comparison.csv', comparison)
    write_csv(args.output_dir/'frame_diagnostics.csv', frames)
    pair_diagnostics = []
    for q in queries:
        if q['actual'] is None:
            continue
        truth = q['true']
        pair_diagnostics.append(dict(scene=q['scene'], order=q['order'], slot=q['slot'],
            history_order=q['history_order'], region=q['region'], actual=q['actual'],
            has_true_history=bool(truth.any()),
            min_true_dy=float(q['dy'][truth].min()) if truth.any() else None,
            min_true_distance=float(q['distance'][truth].min()) if truth.any() else None,
            true_candidates=int(truth.sum())))
    write_csv(args.output_dir/'true_pair_diagnostics.csv', pair_diagnostics)
    full_best = [best([r for r in metrics if r['slot'] == s and r['subset'] == 'all'
                       and r['weight_center'] == 1.]) for s in (1, 2, 3)]
    report = dict(reference_report=str(args.reference_report), config=reference['config'],
        split_order=args.split_order, recommendation=recommendation,
        full_data_distance_optima_exploratory=full_best,
        baseline_reproduced=True, comparison=comparison,
        notes=['Grid tuning on this scene/test split is exploratory, not a generalization claim.',
               'Temporal holdout refers to current frames; adjacent history frames may overlap tuning frames.',
               'Fit criterion: F1 using precision lower bound correct/all predictions on labeled current regions.',
               'Unknown historical GT predictions are not proven wrong; lower bound is conservative.',
               'GT purity 0/minimum 1 gives weak projected labels; GT is never a matcher input.',
               'There may be multiple regions per physical object; counts are region/history queries.',
               'Changing center gates also changes normalized center/size cost balance for mixed weights.'])
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_summary(args.output_dir, report, metrics)
    print(json.dumps(recommendation, indent=2))


if __name__ == '__main__':
    main()
