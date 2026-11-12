#!/usr/bin/env python3
"""Validate frozen scene-3 gates on other scenes without another parameter search."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis.sweep_region_matching_gates import collect, pack, evaluate, write_csv, ratio
from tools.temporal_vis import vis_alignment as va

COUNT_FIELDS = ('all_queries', 'all_predictions', 'current_labeled', 'has_true_history',
                'predicted', 'correct', 'wrong', 'unscorable_prediction', 'y_missed_true',
                'center_missed_true', 'true_after_gates')


def aggregate(rows):
    c = {key: sum(r[key] for r in rows) for key in COUNT_FIELDS}
    c.update(precision=ratio(c['correct'], c['correct']+c['wrong']),
             precision_lower_bound=ratio(c['correct'], c['predicted']),
             recall=ratio(c['correct'], c['has_true_history']),
             f1_lower_bound=ratio(2*c['correct'], c['predicted']+c['has_true_history']),
             gate_retention=ratio(c['true_after_gates'], c['has_true_history']))
    c['mean_candidates_all'] = ratio(sum(r['mean_candidates_all']*r['all_queries']
        for r in rows if r['mean_candidates_all'] is not None), c['all_queries'])
    return c


def validate_scene(task):
    split, scene, config, gates, output = task
    queries, frames = collect(config)
    if not queries:
        raise ValueError(f'No region queries: {split}/{scene}')
    rows = []
    for name, (ys, ds) in gates.items():
        per_slot = []
        for slot in (1, 2, 3):
            a = pack([q for q in queries if q['slot'] == slot])
            counts = evaluate(a, ys[slot-1], ds[slot-1], 1., 0., config['size_scale'],
                              np.ones(len(a['order']), bool))
            per_slot.append(counts)
            rows.append(dict(split=split, scene=scene, configuration=name, slot=str(slot),
                             y_gate=ys[slot-1], center_gate=ds[slot-1], **counts))
        rows.append(dict(split=split, scene=scene, configuration=name, slot='all',
                         y_gate='', center_gate='', **aggregate(per_slot)))
    directory = Path(output)/f'{split}_scene{scene}'
    directory.mkdir()
    write_csv(directory/'metrics.csv', rows)
    write_csv(directory/'frame_diagnostics.csv', sorted(frames, key=lambda r: r['order']))
    diagnostics = []
    for q in queries:
        if q['actual'] is not None:
            truth = q['true']
            diagnostics.append(dict(order=q['order'], history_order=q['history_order'],
                slot=q['slot'], region=q['region'], actual=q['actual'],
                has_true_history=bool(truth.any()),
                min_true_dy=float(q['dy'][truth].min()) if truth.any() else None,
                min_true_distance=float(q['distance'][truth].min()) if truth.any() else None))
    if diagnostics:
        write_csv(directory/'true_pair_diagnostics.csv', diagnostics)
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    return rows


def render(output, report):
    rows = report['per_scene']
    percent = lambda x: f'{100*x:.2f}%' if x is not None else 'N/A'
    lines = ['# 跨场景验证：固定 scene 3 门限', '',
        '固定纯距离匹配权重 1/0；不在其他场景重新选择参数。', '',
        f'原门限：Y={report["gates"]["baseline"][0]} m，'
        f'XY 中心距离={report["gates"]["baseline"][1]} m。',
        f'推荐门限：Y={report["gates"]["recommended"][0]} m，'
        f'XY 中心距离={report["gates"]["recommended"][1]} m。', '',
        '各场景均使用 order 80–200、stride 5；所有区域提取和 GT 归属条件保持参考报告设置。', '',
        '本报告的 recommended 是进入验证前固定的 scene 3 方案，不表示验证后仍应无条件采用。', '',
        '## 本次结果的含义', '',
        '本报告只展示预先冻结配置在下表场景上的实际指标；不预设放宽门限是否有收益。'
        '如果据此再次选参，这些场景就不能继续充当该选择的未见测试集。', '',
        '保守 P = correct / predicted；未知历史身份预测也计入分母，'
        '不表示它们已被证实为错误。保守 F1 = 2*correct/(predicted+has_true_history)。', '',
        '## 各场景结果（合并 t-1/t-2/t-3）', '',
        '| split | scene | 存在正确历史查询数 | 原正确数 | 新正确数 | 原 Recall | 新 Recall | 原保守 P | 新保守 P | 新明确错误 | 新未知历史身份 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split, scene in report['scenes']:
        a = next(r for r in rows if r['split'] == split and r['scene'] == scene
                 and r['configuration'] == 'baseline' and r['slot'] == 'all')
        b = next(r for r in rows if r['split'] == split and r['scene'] == scene
                 and r['configuration'] == 'recommended' and r['slot'] == 'all')
        lines.append(f'| {split} | {scene} | {a["has_true_history"]} | {a["correct"]} | '
            f'{b["correct"]} | {percent(a["recall"])} | {percent(b["recall"])} | '
            f'{percent(a["precision_lower_bound"])} | {percent(b["precision_lower_bound"])} | '
            f'{b["wrong"]} | {b["unscorable_prediction"]} |')
    lines += ['', '## 汇总与消融', '',
        '| 范围 | 配置 | 正确/存在正确历史 | Recall | 条件 P | 保守 P | 保守 F1 | Y 排除 | 中心距离追加排除 | 明确错误 | 未知预测 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['aggregates']:
        if r['slot'] != 'all':
            continue
        lines.append(f'| {r["scope"]} | {r["configuration"]} | {r["correct"]}/{r["has_true_history"]} | '
            f'{percent(r["recall"])} | {percent(r["precision"])} | '
            f'{percent(r["precision_lower_bound"])} | {percent(r["f1_lower_bound"])} | '
            f'{r["y_missed_true"]} | {r["center_missed_true"]} | {r["wrong"]} | {r["unscorable_prediction"]} |')
    lines += ['', '## 分历史间隔（全部外部场景）', '',
        '| 历史帧 | 配置 | 正确/存在正确历史 | Recall | 条件 P | 保守 P | 保守 F1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in report['aggregates']:
        if r['scope'] != 'all_external' or r['slot'] == 'all' or r['configuration'] not in ('baseline', 'recommended'):
            continue
        lines.append(f'| t-{r["slot"]} | {r["configuration"]} | {r["correct"]}/{r["has_true_history"]} | '
            f'{percent(r["recall"])} | {percent(r["precision"])} | '
            f'{percent(r["precision_lower_bound"])} | {percent(r["f1_lower_bound"])} |')
    lines += ['', '## 解释与限制', ''] + [f'- {note}' for note in report['notes']]
    lines += ['', '## 复现', '', '```bash',
        'source use_radarocc5060.sh',
        'python tools/temporal_vis/validate_region_matching_gates.py \\',
        '  --workers 2 \\',
        '  --output-dir outputs/region_matching_benchmark/cross_scene_validation_new', '```']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if report['config'].get('coordinate_convention') == 'kradar_physical_v2':
        path = output/'summary.md'
        path.write_text(path.read_text().replace('scene 3', '参考训练场景').replace('六个外部场景', '本次验证场景'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [f'{s}/{c}' for s, c in report['scenes']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    for ax, metric, title in zip(axes, ('recall', 'f1_lower_bound'), ('Recall', 'Conservative F1')):
        for config, offset in [('baseline', -.18), ('recommended', .18)]:
            vals = [next(r for r in rows if r['split'] == split and r['scene'] == scene
                and r['configuration'] == config and r['slot'] == 'all')[metric] for split, scene in report['scenes']]
            ax.bar(x+offset, [100*v if v is not None else np.nan for v in vals], width=.36, label=config)
        ax.set_xticks(x, labels); ax.set_ylim(0, 105); ax.set_ylabel('%'); ax.set_title(title)
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'cross_scene_comparison.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-report', type=Path,
        default=Path('outputs/region_matching_benchmark/scene3_p97_xy20_loose_gt/report.json'))
    p.add_argument('--ann-files', nargs='+', type=Path, default=[
        Path(f'data/annotations/kradar_dict_{s}_official_temporal_doppler8.pkl') for s in ('val', 'test')])
    p.add_argument('--exclude-scenes', nargs='+', default=['3'])
    p.add_argument('--y-gates', nargs=3, type=float, default=[4, 5, 3])
    p.add_argument('--center-gates', nargs=3, type=float, default=[4, 12, 16])
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Use a new output directory')
    if args.workers < 1 or not all(np.isfinite(v) and v > 0 for v in args.y_gates+args.center_gates):
        p.error('Positive workers and finite positive gates required')
    reference = json.loads(args.reference_report.read_text())
    base = reference['config']
    gates = dict(baseline=(base['y_gates'], base['center_gates']),
                 y_only=(args.y_gates, base['center_gates']),
                 center_only=(base['y_gates'], args.center_gates),
                 recommended=(args.y_gates, args.center_gates),
                 wide_control=([20., 20., 20.], [100., 100., 100.]))
    tasks, scene_list, seen = [], [], set()
    for annotation in args.ann_files:
        split = annotation.stem.replace('kradar_dict_', '').replace('_official_temporal_doppler8', '')
        infos = va.read_infos(annotation)
        scenes = sorted({str(i.get('scene_id', i.get('scene_token'))) for i in infos}, key=int)
        for scene in scenes:
            if scene in args.exclude_scenes:
                continue
            if scene in seen:
                raise ValueError(f'Scene appears in more than one annotation: {scene}')
            seen.add(scene)
            config = dict(base, ann_file=str(annotation), scenes=[scene])
            tasks.append((split, scene, config, gates, str(args.output_dir)))
            scene_list.append((split, scene))
    if not tasks:
        raise ValueError('No external scenes')
    args.output_dir.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(validate_scene, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            part = future.result()
            rows.extend(part)
            print(f'COMPLETED {futures[future]}', flush=True)
    rows.sort(key=lambda r: (r['split'], int(r['scene']), r['configuration'], r['slot']))
    aggregates = []
    for scope in ['all_external']+sorted({s for s, _ in scene_list}):
        for name in gates:
            for slot in ('1', '2', '3', 'all'):
                selected = [r for r in rows if r['configuration'] == name and r['slot'] == slot
                            and (scope == 'all_external' or r['split'] == scope)]
                aggregates.append(dict(scope=scope, configuration=name, slot=slot, **aggregate(selected)))
    report = dict(reference_report=str(args.reference_report), config=base, gates=gates,
        scenes=scene_list, excluded_scenes=args.exclude_scenes, fixed_weights=[1., 0.],
        baseline_source='scene3 reference report; only ann_file and scene replaced',
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [Path(__file__), args.reference_report]+args.ann_files},
        per_scene=rows, aggregates=aggregates,
        notes=['Parameters remain frozen; no gate/weight search on these scenes.',
            'All other extraction, XY merge, GT labeling and frame selection settings match the scene3 reference.',
            'The val/test labels denote source annotation splits, not new training or a leaderboard submission.',
            'Loose XY GT labels (purity=0, minimum=1) evaluate projected region association, not point purity or occupancy.',
            'Conditional precision ignores historical predictions with unknown GT ID.',
            'Conservative precision includes unknown historical predictions in denominator; these are not proven false.',
            'Only labeled current regions contribute to GT metrics; unlabeled background association quality is unknown.',
            'Counts are region/history queries and correlated within scenes, not independent physical objects.',
            'A scene with zero true-history queries has undefined recall; it is not a zero-accuracy result.'])
    write_csv(args.output_dir/'per_scene_metrics.csv', rows)
    write_csv(args.output_dir/'aggregate_metrics.csv', aggregates)
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    render(args.output_dir, report)
    print('Validation complete:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
