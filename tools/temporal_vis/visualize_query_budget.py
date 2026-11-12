#!/usr/bin/env python3
"""Before training: inspect the exact shared Q64/Q32 policy on five scenes."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal.core.query_budget import TemporalQueryBudget, causal_ego_velocity
from temporal.core.ego_alignment import RadarBinGeometry, radar_to_lidar_default, add_coordinate_argument
from tools.temporal_vis import vis_p1_reflection_regions as vis
from tools.temporal_vis.audit_doppler_queries import (rooted, read_frame, gt_masks_for_frame,
                                                     evaluate_mask, write_csv)

CONFIG = 'projects/configs/baselines/RadarOcc_Small_5060_temporal_query_budget.py'
POLICY = dict(doppler_sign='both_consensus', threshold_mps=.2, max_peak_std=.2,
              min_concentration=.8, class_fractions=[.5, .25, .25], angular_cells=[8, 4],
              doppler_bins='data/K-Radar-official-meta/resources/arr_doppler.mat',
              max_ego_dt=1., max_ego_speed=70.)


def source_hashes():
    files = [str(p.relative_to(ROOT)) for folder in ('temporal', 'projects/configs/baselines')
             for p in (ROOT/folder).rglob('*.py')]
    files += [str(Path(__file__).relative_to(ROOT)), 'run_temporal_query_budget.sh',
              POLICY['doppler_bins'], 'data/K-Radar-official-meta/resources/info_arr.mat']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}


def render(xyz, state, masks, gt, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), layout='constrained')
    hulls = [c[:, :2][ConvexHull(c[:, :2]).vertices] for c in gt['corners']]
    for row, (budget, keep) in enumerate(masks.items()):
        for col in range(3):
            ax = axes[row, col]
            if col == 2:
                ids = np.flatnonzero(~keep)
                ax.scatter(xyz[ids, 0], xyz[ids, 1], s=1.4, c='#d62728', alpha=.5, rasterized=True)
            else:
                eligible = np.ones(len(xyz), bool) if col == 0 else keep
                for cls, color, label in [(2, '#167ac6', 'Motion evidence'), (0, '#808080', 'Uncertain'),
                                           (1, '#e6a326', 'Static-consistent')]:
                    ids = np.flatnonzero(eligible & (state == cls))
                    ax.scatter(xyz[ids, 0], xyz[ids, 1], s=2 if col else 1, c=color, alpha=.65,
                               rasterized=True, label=label)
            for hull in hulls:
                ax.add_patch(Polygon(hull, closed=True, fill=False, edgecolor='#00924b', lw=1.2))
            ax.set(xlim=(0, 85), ylim=(-25, 25), xlabel='X forward (m)', ylabel='Y left (m)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.15)
            ax.set_title([f'Original: {len(xyz):,} tokens',
                          f'Q{budget}: {keep.sum():,} queries kept ({1-keep.mean():.1%} removed)',
                          f'Q{budget}: {(~keep).sum():,} temporal queries removed'][col])
        axes[row, 0].legend(fontsize=8)
    fig.suptitle(title+'\nGreen: GT (evaluation only). All single-frame tokens and all historical candidates remain.\n'
                 'Blue/gold are Doppler hypotheses, not foreground/background labels; all plotted points are drawn.', fontsize=11)
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_coordinate_argument(p)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/query_budget/five_scenes_64_32'))
    p.add_argument('--scenes', nargs='+', default=['1', '2', '3', '15', '22'])
    p.add_argument('--start-order', type=int, default=80)
    p.add_argument('--end-order', type=int, default=200)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--frames-per-scene', type=int, default=25)
    p.add_argument('--ann-files', nargs='+', type=Path, default=[
        Path(f'data/annotations/kradar_dict_{s}_official_temporal_doppler8.pkl') for s in ('val', 'test')])
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--plot-workers', type=int, default=2)
    args = p.parse_args()
    if args.stride <= 0 or args.frames_per_scene <= 0 or args.plot_workers <= 0:
        p.error('Positive stride and frame count required')
    out = rooted(args.output_dir)
    if out.exists() and any(out.iterdir()):
        p.error('Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out/'matplotlib_cache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(out/'cache'))
    hashes = source_hashes()
    renderer = None if args.no_plots else ProcessPoolExecutor(args.plot_workers, mp_context=get_context('spawn'))
    pending = []
    geometry = RadarBinGeometry.from_file(ROOT/'data/K-Radar-official-meta/resources/info_arr.mat', coordinate_convention=args.coordinate_convention)
    extrinsic = radar_to_lidar_default()
    selectors = {k: TemporalQueryBudget(per_range=k, **POLICY) for k in (64, 32)}
    inventory = [(rooted(path), vis.annotation_scenes(rooted(path))) for path in args.ann_files]
    metrics, boxes, frames = [], [], []
    for scene in map(vis.scene_key, args.scenes):
        sources = [(path, groups[scene]) for path, groups in inventory if scene in groups]
        if len(sources) != 1:
            p.error(f'Expected one source for scene {scene}')
        annotation, rows = sources[0]
        selected = [i for i, r in enumerate(rows) if args.start_order <= int(r['order_in_scene']) <= args.end_order
                    and (int(r['order_in_scene'])-args.start_order) % args.stride == 0][:args.frames_per_scene]
        if len(selected) != args.frames_per_scene:
            p.error(f'Scene {scene}: not enough frames')
        for i in selected:
            info = rows[i]
            coords, xyz, features = read_frame(rooted(Path(info.get('sparse_radar_path') or info['radar_path'])), geometry)
            # Same last-four-entry sequence as TemporalKRadarDataset. No future
            # frame, GT input, or preceding sampled-frame approximation.
            sequence = rows[max(0, i-3):i+1]
            poses = np.asarray([r['ego_pose'] for r in sequence])
            times = np.asarray([r.get('timestamp_sec', np.nan) for r in sequence], float)
            velocity, reason = causal_ego_velocity(poses, times, np.ones(len(sequence), bool), extrinsic)
            predictions = {k: selector.select_frame(features, coords, xyz, velocity) for k, selector in selectors.items()}
            gt = vis.load_gt(info, scene, rooted(args.bbox_root), extrinsic)
            roi = ((xyz >= [0, -60, -5]) & (xyz < [85, 60, 8])).all(1)
            memberships = gt_masks_for_frame(xyz, gt, roi, 'xy')
            order = int(info['order_in_scene'])
            base = dict(scene=scene, order=order, annotation=str(annotation), ego_reason=reason)
            masks = {}
            for k, (keep, detail) in predictions.items():
                expected = len(np.unique(coords[:, 0]))*k
                if int(keep.sum()) != expected:
                    raise AssertionError('Exact per-range query budget failed')
                # Confirm invariance to the model's per-range power ordering.
                permutation = np.lexsort((np.arange(len(coords)), -features[:, 2], coords[:, 0]))
                reordered, _ = selectors[k].select_frame(features[permutation], coords[permutation], xyz[permutation], velocity)
                restored = np.zeros_like(keep)
                restored[permutation] = reordered
                if not np.array_equal(restored, keep):
                    raise AssertionError('Offline / model token ordering changes selected observations')
                truth, before, after = evaluate_mask(keep, memberships)
                row = dict(**base, per_range=k, n_tokens=len(keep), n_kept=int(keep.sum()),
                           reduction_rate=float(1-keep.mean()), **truth)
                row.update(gt_token_retention=truth['n_gt_union_kept']/truth['n_gt_union'] if truth['n_gt_union'] else None,
                    **{f'n_kept_class_{cls}': int((keep & (detail['state'] == cls)).sum()) for cls in (0, 1, 2)})
                metrics.append(row)
                for j, (old, new) in enumerate(zip(before, after)):
                    boxes.append(dict(**base, per_range=k, gt_index=j, gt_id=str(gt['ids'][j]),
                        n_original=int(old), n_kept=int(new), retention=float(new/old) if old else None,
                        all_pruned=bool(old and not new)))
                masks[k] = keep
            dest = Path(f'scene{scene}')/f'order{order:05d}'
            (out/dest.parent).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out/dest.with_suffix('.npz'), coords_rae=coords, xyz=xyz.astype(np.float32),
                state=detail['state'], keep_Q64=masks[64], keep_Q32=masks[32],
                original_npz_point_id=np.arange(len(coords)), token_order=np.array('original_npz'))
            if not args.no_plots:
                pending.append(renderer.submit(render, xyz, detail['state'], masks, gt,
                    out/dest.with_suffix('.png'), f'Scene {scene} order {order} | both-sign consensus'))
                if len(pending) >= 2*args.plot_workers:
                    pending.pop(0).result()
            frames.append(dict(**base, image=str(dest.with_suffix('.png')), mask=str(dest.with_suffix('.npz'))))
            print(f'Scene {scene} order {order}: Q64={masks[64].sum()}, Q32={masks[32].sum()}', flush=True)
        write_csv(out/'frame_metrics.csv', metrics)
        write_csv(out/'gt_per_box.csv', boxes)
    for future in pending:
        future.result()
    if renderer is not None:
        renderer.shutdown()
    if source_hashes() != hashes:
        raise RuntimeError('Source changed during visualization; rerun in a new directory')
    summary = []
    for scene in list(map(vis.scene_key, args.scenes))+['ALL']:
        for k in (64, 32):
            group = [r for r in metrics if r['per_range'] == k and (scene == 'ALL' or r['scene'] == scene)]
            sums = {name: sum(r[name] for r in group) for name in ('n_tokens', 'n_kept', 'n_gt_union',
                'n_gt_union_kept', 'n_gt_boxes_nonempty', 'n_gt_boxes_all_pruned')}
            summary.append(dict(scene=scene, per_range=k, frames=len(group), **sums,
                reduction_rate=1-sums['n_kept']/sums['n_tokens'],
                gt_token_retention=sums['n_gt_union_kept']/sums['n_gt_union'] if sums['n_gt_union'] else None))
    write_csv(out/'summary.csv', summary)
    page = ['<!doctype html><meta charset="utf-8"><title>Q64 / Q32 review</title>',
            '<style>body{font:16px sans-serif;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>',
            '<h1>时序 Query 筛选：每 range 保留 64 / 32</h1>',
            '<p>蓝：两种符号均有运动证据；金：两种符号均与静止一致；灰：不确定。红：取消时序 Query；绿：GT。</p>',
            '<p>不确定点也受预算限制。单帧 token 和历史候选完整保留。本页只检查 Query 覆盖，不代表选点器关联正确或 IoU 提升。</p>',
            '<p><a href="summary.csv">汇总</a> · <a href="frame_metrics.csv">逐帧</a> · <a href="gt_per_box.csv">逐 GT 框</a></p>',
            '<table><tr><th>场景</th><th>每 range</th><th>减少率</th><th>GT 点保留率</th><th>被剪空 / 原非空 GT 框</th></tr>']
    for row in summary:
        retention = 'N/A' if row['gt_token_retention'] is None else f"{row['gt_token_retention']:.2%}"
        page.append(f'<tr><td>{row["scene"]}</td><td>{row["per_range"]}</td><td>{row["reduction_rate"]:.2%}</td>'
                    f'<td>{retention}</td><td>{row["n_gt_boxes_all_pruned"]}/{row["n_gt_boxes_nonempty"]}</td></tr>')
    page.append('</table>')
    for frame in frames:
        image = html.escape(frame['image'])
        page.append(f'<details><summary>Scene {frame["scene"]} order {frame["order"]}</summary>'
                    f'<a href="{image}"><img loading="lazy" src="{image}"></a></details>')
    (out/'index.html').write_text('\n'.join(page))
    manifest = dict(complete=True, plots_generated=not args.no_plots, budgets=[64, 32], policy=POLICY,
        source_sha256=hashes, frames=frames, gt_used_for_selection=False, training_started=False,
        order_invariance_verified=True, config=vars(args))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    print('Review visualization:', out/'index.html')


if __name__ == '__main__':
    main()
