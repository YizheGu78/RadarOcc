#!/usr/bin/env python3
"""CPU-only query-pruning audit: circular Doppler, ego compensation, GT evaluation.

This does not edit RadarOcc inputs or train anything. It estimates query budgets
and saves original-NPZ-order masks. Stationary consistency is NOT true velocity.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as va
from tools.temporal_vis import vis_p1_reflection_regions as vis
from tools.temporal_vis.neighborhood_motion import ego_velocity, load_doppler_table
from tools.temporal_vis.doppler_query_pruning import (
    UNCERTAIN, STATIC_CONSISTENT, NONSTATIC_EVIDENCE, circular_peaks,
    fft_period, query_masks, static_consistency)

METHODS = ('doppler_heuristic', 'alias_guarded', 'spatial_same_budget')
COUNTS = ('n_tokens', 'n_kept', 'n_pruned', 'n_eval_roi', 'n_roi_kept',
          'n_uncertain', 'n_static_consistent', 'n_nonstatic_evidence',
          'n_alias_ambiguous_static', 'n_gt_union', 'n_gt_union_kept',
          'n_gt_boxes', 'n_gt_boxes_nonempty', 'n_gt_boxes_empty_before',
          'n_gt_boxes_kept', 'n_gt_boxes_all_pruned')


def rooted(path):
    return path if path.is_absolute() else ROOT/path


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def rates(row):
    def ratio(a, b):
        return row[a]/row[b] if row[b] else None
    return dict(reduction_rate=ratio('n_pruned', 'n_tokens'),
                kept_rate=ratio('n_kept', 'n_tokens'),
                roi_kept_rate=ratio('n_roi_kept', 'n_eval_roi'),
                gt_token_retention=ratio('n_gt_union_kept', 'n_gt_union'),
                gt_box_coverage=ratio('n_gt_boxes_kept', 'n_gt_boxes_nonempty'))


def causal_velocity(rows, position, extrinsic, max_dt=1., max_speed=70.):
    """Use the preceding actual annotation entry, not preceding sampled order."""
    if position == 0:
        return None, dict(ego_reason='no_previous_frame', previous_order=None, ego_dt_sec=None)
    current, previous = rows[position], rows[position-1]
    details = dict(previous_order=int(previous['order_in_scene']), ego_dt_sec=None)
    if not all(r.get('ego_pose_valid', False) for r in (current, previous)):
        return None, dict(**details, ego_reason='invalid_pose_flag')
    if not all(r.get('timestamp_valid', True) for r in (current, previous)):
        return None, dict(**details, ego_reason='invalid_timestamp_flag')
    try:
        dt = float(current['timestamp_sec'])-float(previous['timestamp_sec'])
        details['ego_dt_sec'] = dt
        if not np.isfinite(dt) or not 0 < dt <= max_dt:
            return None, dict(**details, ego_reason='invalid_or_large_time_gap')
        velocity = ego_velocity(np.asarray(current['ego_pose']), np.asarray(previous['ego_pose']), extrinsic, dt)
        if np.linalg.norm(velocity) > max_speed:
            return None, dict(**details, ego_reason='ego_speed_exceeds_guard')
    except (KeyError, TypeError, ValueError) as error:
        return None, dict(**details, ego_reason='invalid_pose_or_timestamp', ego_error=str(error))
    return velocity, dict(**details, ego_reason='ok', ego_vx_mps=float(velocity[0]),
                          ego_vy_mps=float(velocity[1]), ego_vz_mps=float(velocity[2]))


def read_frame(path, geometry):
    n = 175*250
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data['power_val'])
        if features.ndim != 2 or features.shape[0] != 8 or features.shape[1] < n:
            raise ValueError(f'{path}: expected raw Doppler8 [8,N], N >= {n}')
        range_ids = np.repeat(np.arange(175), 250)
        if 'range_ind' in data and not np.array_equal(np.asarray(data['range_ind']).reshape(-1)[:n], range_ids):
            raise ValueError(f'{path}: range grouping disagrees with the model P0 input')
        coords = np.column_stack((range_ids, np.asarray(data['azimuth_ind']).reshape(-1)[:n],
                                  np.asarray(data['elevation_ind']).reshape(-1)[:n]))
        features = features[:, :n].T.astype(np.float64)
    return coords, geometry.to_xyz(coords), features


def gt_masks_for_frame(xyz, gt, roi, space):
    if gt is None:
        return np.zeros((0, len(xyz)), bool)
    return np.asarray([roi & (vis.gt_memberships(xyz, np.asarray(b)[None], gt['radar_to_lidar'], space) >= 0)
                       for b in gt['boxes']], dtype=bool).reshape(-1, len(xyz))


def evaluate_mask(keep, gt_masks):
    """Evaluation only. Overlapping GTs remain independent boxes; union points dedup."""
    before = gt_masks.sum(axis=1)
    after = (gt_masks & keep).sum(axis=1)
    union = gt_masks.any(axis=0)
    return dict(n_gt_union=int(union.sum()), n_gt_union_kept=int((union & keep).sum()),
                n_gt_boxes=len(before), n_gt_boxes_nonempty=int((before > 0).sum()),
                n_gt_boxes_empty_before=int((before == 0).sum()),
                n_gt_boxes_kept=int(((before > 0) & (after > 0)).sum()),
                n_gt_boxes_all_pruned=int(((before > 0) & (after == 0)).sum())), before, after


def render(xyz, state, keep, gt, path, title, max_points=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    hulls = [] if gt is None else [c[:, :2][ConvexHull(c[:, :2]).vertices] for c in gt['corners']]

    def scatter(ax, mask, color, size=1.5, alpha=.65):
        ids = np.flatnonzero(mask)
        if max_points and len(ids) > max_points:
            ids = ids[np.linspace(0, len(ids)-1, max_points, dtype=int)]
        ax.scatter(xyz[ids, 0], xyz[ids, 1], c=color, s=size, alpha=alpha, rasterized=True)

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            if col < 2:
                eligible = np.ones(len(xyz), bool) if col == 0 else keep
                for cls, color in [(UNCERTAIN, '#8c8c8c'), (STATIC_CONSISTENT, '#e6a326'),
                                   (NONSTATIC_EVIDENCE, '#167ac6')]:
                    scatter(ax, eligible & (state == cls), color)
            else:
                scatter(ax, ~keep, '#d62728', size=2)
            for hull in hulls:
                ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor='#00924b', linewidth=1.1))
            ax.set_xlim(0, 85)
            ax.set_ylim((-60, 60) if row == 0 else (-15, 20))
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X forward (m)')
            ax.set_ylabel('Y left (m)')
            ax.grid(alpha=.15)
            ax.set_title([f'All P0 tokens: {len(xyz):,}', f'Kept temporal queries: {int(keep.sum()):,}',
                          f'Pruned temporal queries: {int((~keep).sum()):,}'][col])
    legend = [Line2D([], [], marker='o', color=c, linestyle='', label=s) for c, s in
              [('#167ac6', 'Non-static Doppler evidence'), ('#e6a326', 'Static-consistent (aliases unresolved)'),
               ('#8c8c8c', 'Uncertain: kept'), ('#d62728', 'Pruned from temporal queries only'),
               ('#00924b', 'GT footprint: evaluation only')]]
    fig.legend(handles=legend, loc='outside lower center', ncol=3, fontsize=9)
    fig.suptitle(title+'\nCounts use ALL 43,750 P0 tokens; plots show XY windows. Original model tokens are not deleted.'
                 + ('\nDisplay is subsampled; statistics and saved masks use all tokens.' if max_points else ''), fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def aggregate(rows, across_scenes=False):
    keys = ('doppler_sign', 'threshold_mps', 'method') if across_scenes else ('scene', 'doppler_sign', 'threshold_mps', 'method')
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in groups.items():
        record = dict(zip(keys, key), frames=len(group))
        record.update({k: sum(row[k] for row in group) for k in COUNTS})
        record.update(rates(record))
        output.append(record)
    return output


def percent(value):
    return '—' if value is None else f'{value:.2%}'


def report_files(output, summary, all_summary, frames, args, manifest):
    write_csv(output/'summary_by_scene.csv', summary)
    write_csv(output/'summary_all_scenes.csv', all_summary)
    lines = ['# Doppler 时序 Query 减少率离线检查', '',
             '只检查时序 Query 预算，不修改原 token、历史候选、模型权重或训练代码。',
             'Doppler 循环统计与自车补偿之后的残差，表示与静止假设的模周期一致程度，不是真实三维速度。', '',
             '## 默认可视化阈值下的结果', '',
             '| 场景 | 符号 | 帧数 | 启发式 Query 减少率 | 别名保护减少率 | GT 点保留率 | 同预算空间采样 GT 点保留率 | GT 框被剪空/原非空框 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        if row['method'] != 'doppler_heuristic' or not np.isclose(row['threshold_mps'], args.visual_threshold):
            continue
        match = lambda name: next(r for r in summary if all(r[k] == row[k] for k in
                    ('scene', 'doppler_sign', 'threshold_mps')) and r['method'] == name)
        guard, spatial = match('alias_guarded'), match('spatial_same_budget')
        lines.append(f'| {row["scene"]} | {row["doppler_sign"]:+d} | {row["frames"]} | '
                     f'{percent(row["reduction_rate"])} | {percent(guard["reduction_rate"])} | '
                     f'{percent(row["gt_token_retention"])} | {percent(spatial["gt_token_retention"])} | '
                     f'{row["n_gt_boxes_all_pruned"]}/{row["n_gt_boxes_nonempty"]} |')
    lines += ['', '## 如何理解', '',
              f'- 阈值扫描：{args.thresholds} m/s；图片使用 {args.visual_threshold} m/s。',
              f'- 静止一致点按 range 分层保留约 {args.static_keep_ratio:.0%}（各 range 向上取整）；不可靠峰、无可用自车速度的点全部保留。',
              '- doppler_heuristic：削减静止一致点，但保留静止点配额；可能误剪折叠后的运动目标。',
              f'- alias_guarded：在世界径向速度绝对值不超过 {args.max_target_radial_mps:g} m/s 的假设下，只削减没有非静止别名候选的静止一致点。',
              '- 速度范围较窄、别名候选多时，alias_guarded 很可能无法减少 Query。它不是经过验证的真实静止分类器，也无法排除横向运动。',
              '- spatial_same_budget：按 range 分层、角度排序选取代表点，与启发式保留完全相同的 Query 数，仅作覆盖率对照。',
              '- 两种 Doppler 符号独立报告；没有按 GT 选择符号。应由独立静止场景/传感器约定确认符号。',
              '- Query 减少率分母为完整 P0 输入 43,750；GT 统计在指定评估 ROI 内进行，空 GT 框不作为剪空框。',
              '- GT 框覆盖率要求原本有点的框在保留后至少还有一个点，不表示目标检测或 Occupancy IoU。',
              '- 当前不执行邻域搜索、Learned KNN 或 Attention，因此减少率不是整模型实测加速比。', '',
              '## 输出', '',
              '- frame_metrics.csv：逐场景、逐帧、逐符号、逐阈值、逐方法的真实计数与比例。',
              '- gt_per_box.csv：每个 GT 框原有/保留/取消 Query 数及是否被剪空。',
              '- summary_by_scene.csv、summary_all_scenes.csv：按计数汇总；跨帧 GT 框按观测次数计。',
              '- frame_diagnostics.csv：自车速度、时间间隔、峰可靠性、跨边界峰等。',
              '- scene*/sign_*/order*.png：启发式策略保留/取消 token 的总览和细节图。',
              '- scene*/masks/order*.npz：所有阈值和方法的完整布尔 mask，按原 NPZ 前 175×250 点顺序保存。',
              '- masks 中 keep_bits 用 np.unpackbits(..., axis=-1, count=n_tokens) 恢复；轴为 [sign, threshold, method, point]。',
              '- 在线模型按 power 排序过 token，后续接入训练时必须映射顺序；不能直接套用本离线 NPZ 顺序的 mask。',
              '- index.html：逐帧可视化入口。', '']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    page = ['<!doctype html><meta charset="utf-8"><title>Doppler query audit</title>',
            '<style>body{font:15px sans-serif;max-width:1500px;margin:24px auto}img{width:100%;max-width:1200px}details{margin:16px 0}</style>',
            '<h1>Doppler 时序 Query 离线检查</h1>',
            '<p>蓝色：非静止 Doppler 证据；金色：静止一致；灰色：不确定且保留；红色：取消时序 Query；绿色：GT 框。</p>',
            '<p>所有原始 token 仍保留在单帧路径。静止一致不代表已解开速度混叠。两种速度符号没有用 GT 优选。</p>',
            '<p><a href="summary.md">统计说明</a> · <a href="frame_metrics.csv">逐帧统计</a> · <a href="gt_per_box.csv">逐 GT 框统计</a></p>']
    for scene in args.scenes:
        page.append(f'<h2>Scene {html.escape(str(scene))}</h2>')
        for frame in frames:
            if frame['scene'] != scene:
                continue
            images = ''.join(f'<p>{html.escape(label)}</p><a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}"></a>'
                             for label, path in frame['images'].items())
            page.append(f'<details><summary>order {frame["order"]} / radar {frame["radar_frame_id"]}</summary>{images}</details>')
    (output/'index.html').write_text('\n'.join(page), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', default=['1', '2', '3', '15', '22'])
    p.add_argument('--ann-files', nargs='+', type=Path, default=[
        Path(f'data/annotations/kradar_dict_{s}_official_temporal_doppler8.pkl') for s in ('val', 'test')])
    p.add_argument('--start-order', type=int, default=80)
    p.add_argument('--end-order', type=int, default=200)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--frames-per-scene', type=int, default=25)
    p.add_argument('--thresholds', nargs='+', type=float, default=[.1, .2, .3, .5])
    p.add_argument('--visual-threshold', type=float, default=.2)
    p.add_argument('--static-keep-ratio', type=float, default=.25)
    p.add_argument('--max-peak-std', type=float, default=.2)
    p.add_argument('--min-concentration', type=float, default=.8)
    p.add_argument('--max-target-radial-mps', type=float, default=30.)
    p.add_argument('--doppler-sign', choices=('both', '1', '-1'), default='both')
    p.add_argument('--max-ego-dt', type=float, default=1.)
    p.add_argument('--max-ego-speed', type=float, default=70.)
    p.add_argument('--doppler-bins', type=Path, default=Path('data/K-Radar-official-meta/resources/arr_doppler.mat'))
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--extrinsic', type=Path)
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--gt-space', choices=('xy', 'xyz'), default='xy')
    p.add_argument('--no-gt', action='store_true')
    p.add_argument('--eval-x-limits', nargs=2, type=float, default=[0, 85])
    p.add_argument('--eval-y-limits', nargs=2, type=float, default=[-60, 60])
    p.add_argument('--eval-z-limits', nargs=2, type=float, default=[-5, 8])
    p.add_argument('--plot-max-points', type=int, default=0, help='0 draws every token; statistics always use all')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/doppler_query_audit/five_scenes_25frames'))
    args = p.parse_args()
    if min(args.stride, args.frames_per_scene) < 1 or args.start_order > args.end_order or args.plot_max_points < 0:
        p.error('Require positive stride/frame count, ordered bounds, nonnegative plot limit')
    args.scenes = [vis.scene_key(s) for s in args.scenes]
    if len(set(args.scenes)) != len(args.scenes) or len(set(args.thresholds)) != len(args.thresholds):
        p.error('Scenes and thresholds must not repeat')
    table = load_doppler_table(rooted(args.doppler_bins))
    period = fft_period(table)
    if not all(np.isfinite(t) and 0 < t < period/2 for t in args.thresholds):
        p.error('Every threshold must be finite and within (0, period/2)')
    visual = [i for i, t in enumerate(args.thresholds) if np.isclose(t, args.visual_threshold)]
    if len(visual) != 1 or not 0 <= args.static_keep_ratio <= 1:
        p.error('visual-threshold must occur once in thresholds; static-keep-ratio must be in [0,1]')
    if not all(np.isfinite(v) and v > 0 for v in (args.max_ego_dt, args.max_ego_speed, args.max_target_radial_mps)):
        p.error('Positive finite time/speed guards required')
    if not 0 < args.max_peak_std < period/2 or not 0 < args.min_concentration <= 1:
        p.error('Invalid peak reliability limits')
    limits = np.array([args.eval_x_limits, args.eval_y_limits, args.eval_z_limits])
    if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        p.error('Each evaluation ROI limit must be finite and increasing')
    output = rooted(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error('Use a new or empty output directory')
    inventory = [(rooted(path), vis.annotation_scenes(rooted(path))) for path in args.ann_files]
    tasks = []
    for scene in args.scenes:
        matches = [(path, groups[scene]) for path, groups in inventory if scene in groups]
        if len(matches) != 1:
            p.error(f'Scene {scene}: expected exactly one source split, found {len(matches)}')
        path, rows = matches[0]
        if len({int(r['order_in_scene']) for r in rows}) != len(rows):
            p.error(f'Scene {scene}: duplicate orders')
        positions = [i for i, r in enumerate(rows) if args.start_order <= int(r['order_in_scene']) <= args.end_order
                     and (int(r['order_in_scene'])-args.start_order) % args.stride == 0][:args.frames_per_scene]
        if len(positions) != args.frames_per_scene:
            p.error(f'Scene {scene}: need {args.frames_per_scene} frames, selected only {len(positions)}')
        tasks.append((scene, path, rows, positions))
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output/'matplotlib_cache'))
    geometry = va.RadarBinGeometry.from_file(rooted(args.bins))
    extrinsic = va.geometry_module.radar_to_lidar_default() if args.extrinsic is None else np.load(rooted(args.extrinsic), allow_pickle=False)
    extrinsic = va.geometry_module.validate_transform(extrinsic)
    signs = [1, -1] if args.doppler_sign == 'both' else [int(args.doppler_sign)]
    metrics, boxes, diagnostics, frames = [], [], [], []
    for scene, annotation, rows, positions in tasks:
        for number, position in enumerate(positions):
            info = rows[position]
            order = int(info['order_in_scene'])
            path = rooted(Path(info.get('sparse_radar_path') or info['radar_path']))
            coords, xyz, features = read_frame(path, geometry)
            peaks = circular_peaks(features, table, args.max_peak_std, args.min_concentration)
            velocity, ego_details = causal_velocity(rows, position, extrinsic, args.max_ego_dt, args.max_ego_speed)
            roi = ((xyz >= limits[:, 0]) & (xyz < limits[:, 1])).all(1)
            packed, states, residuals, alias_counts, images = [], [], [], [], {}
            base = dict(scene=scene, order=order, radar_frame_id=info.get('radar_frame_id'),
                        occ_frame_idx=info.get('occ_frame_idx'), annotation=str(annotation))
            # Predictions for all signs/thresholds are created before any GT is read.
            predictions = []
            for sign in signs:
                consistency = static_consistency(xyz, peaks, velocity, sign, args.max_target_radial_mps)
                variants = [query_masks(coords, consistency, t, args.static_keep_ratio) for t in args.thresholds]
                predictions.append((sign, consistency, variants))
            gt = None if args.no_gt else vis.load_gt(info, scene, rooted(args.bbox_root), extrinsic)
            gt_masks = gt_masks_for_frame(xyz, gt, roi, args.gt_space)
            for sign, consistency, variants in predictions:
                diagnostics.append(dict(**base, doppler_sign=sign, **ego_details,
                    n_tokens=len(xyz), n_peak_reliable=int(peaks['reliable'].sum()),
                    n_invalid_peak_input=int((~peaks['valid_input']).sum()),
                    n_seam_crossing=int(peaks['seam_crossing'].sum()),
                    n_seam_crossing_reliable=int((peaks['seam_crossing'] & peaks['reliable']).sum()),
                    n_linear_false_near_zero=int(((np.abs(peaks['linear_mean_mps']) <= args.visual_threshold)
                        & (np.abs(peaks['folded_mps']) > args.visual_threshold) & peaks['reliable']).sum())))
                per_threshold, per_state = [], []
                for threshold, masks in zip(args.thresholds, variants):
                    per_method = []
                    for method in METHODS:
                        keep = masks[method]
                        truth, before, after = evaluate_mask(keep, gt_masks)
                        row = dict(**base, doppler_sign=sign, threshold_mps=threshold, method=method,
                            n_tokens=len(keep), n_kept=int(keep.sum()), n_pruned=int((~keep).sum()),
                            n_eval_roi=int(roi.sum()), n_roi_kept=int((roi & keep).sum()),
                            n_uncertain=int((masks['state'] == UNCERTAIN).sum()),
                            n_static_consistent=int(masks['static_consistent'].sum()),
                            n_nonstatic_evidence=int((masks['state'] == NONSTATIC_EVIDENCE).sum()),
                            n_alias_ambiguous_static=int(masks['alias_ambiguous_static'].sum()), **truth)
                        row.update(rates(row))
                        metrics.append(row)
                        for j, (old, new) in enumerate(zip(before, after)):
                            boxes.append(dict(**base, doppler_sign=sign, threshold_mps=threshold, method=method,
                                gt_index=j, gt_id=str(gt['ids'][j]), duplicate_gt_id=bool(gt['duplicate_ids'][j]),
                                n_original=int(old), n_kept=int(new), n_pruned=int(old-new),
                                retention_rate=float(new/old) if old else None, all_pruned=bool(old > 0 and new == 0)))
                        per_method.append(np.packbits(keep))
                    per_threshold.append(np.stack(per_method))
                    per_state.append(masks['state'])
                packed.append(np.stack(per_threshold))
                states.append(np.stack(per_state))
                residuals.append(consistency['residual_mps'].astype(np.float32))
                alias_counts.append(consistency['alias_candidate_count'])
                if not args.no_plots:
                    label = 'sign_pos' if sign == 1 else 'sign_neg'
                    relative = Path(f'scene{scene}')/label/f'order{order:05d}.png'
                    (output/relative).parent.mkdir(parents=True, exist_ok=True)
                    masks = variants[visual[0]]
                    keep = masks['doppler_heuristic']
                    render(xyz, masks['state'], keep, gt, output/relative,
                        f'Scene {scene} order {order} | sign {sign:+d} | threshold {args.visual_threshold:g} m/s | '
                        f'heuristic reduction {1-keep.mean():.1%} | alias-guarded reduction {1-masks["alias_guarded"].mean():.1%}',
                        args.plot_max_points)
                    images[label] = str(relative)
            relative_mask = Path(f'scene{scene}')/'masks'/f'order{order:05d}.npz'
            (output/relative_mask).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output/relative_mask, xyz=xyz.astype(np.float32), coords_rae=coords.astype(np.int16),
                original_npz_point_id=np.arange(len(xyz), dtype=np.int32), n_tokens=np.array(len(xyz)),
                signs=np.array(signs), thresholds_mps=np.array(args.thresholds), methods=np.array(METHODS),
                keep_bits=np.stack(packed), state=np.stack(states), eval_roi_bits=np.packbits(roi),
                folded_radial_mps=peaks['folded_mps'].astype(np.float32),
                circular_std_mps=peaks['circular_std_mps'].astype(np.float32),
                concentration=peaks['concentration'].astype(np.float32), peak_reliable=peaks['reliable'],
                static_residual_mps=np.stack(residuals), alias_candidate_count=np.stack(alias_counts),
                source_npz=np.array(str(path)), ego_velocity_mps=np.full(3, np.nan) if velocity is None else velocity)
            frames.append(dict(**base, source_npz=str(path), masks=str(relative_mask), images=images))
            print(f'scene={scene} {number+1}/{len(positions)} order={order} ego={ego_details["ego_reason"]} '
                  + ' '.join(f'sign{s:+d} reduce={1-v[visual[0]]["doppler_heuristic"].mean():.1%}' for s, _, v in predictions), flush=True)
        # Save completed-scene tables as well, so a later plotting/data error loses no completed scene.
        write_csv(output/'frame_metrics.csv', metrics)
        write_csv(output/'gt_per_box.csv', boxes)
        write_csv(output/'frame_diagnostics.csv', diagnostics)
    manifest = dict(config=vars(args), period_mps=period, doppler_table_mps=table.tolist(),
        radar_to_lidar=extrinsic.tolist(), frames=frames, gt_used_for_prediction=False,
        no_model_training=True, original_tokens_modified=False, alias_resolution_performed=False,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       [Path(__file__), Path(__file__).with_name('doppler_query_pruning.py'),
                        Path(__file__).with_name('neighborhood_motion.py'), rooted(args.bins), rooted(args.doppler_bins)]
                       + [path for path, _ in inventory]},
        notes=['All masks index original NPZ order, not model power-sorted P0 order.',
               'Two signs are sensitivity hypotheses, not GT-selected calibration.',
               'Folded static consistency does not determine true target speed or foreground identity.',
               'Selected val/test scenes are exploratory diagnostics, not independent threshold validation.'])
    report_files(output, aggregate(metrics), aggregate(metrics, True), frames, args, manifest)
    print('Saved audit:', output, flush=True)


if __name__ == '__main__':
    main()
