#!/usr/bin/env python3
"""Compare GT-free P0 vs post-Range-Attention P1 radar reflection proposals.

Uses one frozen single-frame RadarOcc checkpoint. All modes operate on identical
P0 token locations and identical region extraction settings. The score-proxy
control (P0-feature vs P1-feature) uses the same robust-score rule,
while feature values and distributions differ. It is not a learned detector. GT is used for evaluation only.
Also saves all_frames_box_power.csv: one row per frame and individual GT/proposal
box, with original-power mean/min/max and separate support/inside/outside-GT subsets.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as alignment
from tools.temporal_vis import vis_p1_reflection_regions as oldvis
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions
from tools.temporal_vis.region_power_statistics import frame_power_statistics, write_power_statistics


def robust_descriptor_score(descriptor, num_ranges=175, k=250):
    """Dimensionless, per-range robust activation norm; no Doppler bins as score.

    This is a diagnostic activation proxy, not a calibrated power or object score.
    """
    if descriptor.shape != (num_ranges*k, 5):
        raise ValueError(f'Expected {num_ranges*k}x5 continuous descriptors, got {descriptor.shape}')
    x = np.asarray(descriptor, dtype=np.float64).reshape(num_ranges, k, 5)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite continuous descriptor features")
    med = np.median(x, axis=1, keepdims=True)
    deviation = np.abs(x-med)
    mad = np.median(deviation, axis=1, keepdims=True)
    # Reject numerical-constant channels rather than manufacturing saliency.
    # A zero MAD alone does not imply constant input (e.g. one sparse return).
    tolerance = 1e-8*np.maximum(1.0, np.abs(med))
    dynamic = deviation.max(axis=1, keepdims=True) > tolerance
    scale = np.where(mad > tolerance, 1.4826*mad,
                     deviation.mean(axis=1, keepdims=True))
    denom = np.where(dynamic, np.maximum(scale, tolerance), 1.0)
    z = np.where(dynamic, (x-med)/denom, 0.0)
    return np.sqrt(np.mean(np.clip(np.abs(z), 0, 20)**2, axis=-1)).reshape(-1)


def load_frozen_range_module(config_path, checkpoint):
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    import projects.occ_plugin  # register original RadarOcc modules

    if not torch.cuda.is_available():
        raise RuntimeError('P1 Range Attention requires a CUDA GPU and radarocc5060 environment')
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    cfg = Config.fromfile(str(config_path))
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint_data = load_checkpoint(model, str(checkpoint), map_location='cpu')
    # load_checkpoint defaults to non-strict loading: ensure this is not an
    # accidentally incompatible checkpoint with randomly initialized RA.
    raw_weights = checkpoint_data.get('state_dict', checkpoint_data)
    weights = {k.removeprefix('module.'): v for k, v in raw_weights.items()}
    loaded_keys = set(weights)
    required = {k for k in model.state_dict() if k.startswith('pts_middle_encoder.range_attn.')}
    missing = sorted(required - loaded_keys)
    if not required or missing:
        raise RuntimeError(f'Checkpoint is missing {len(missing)} of {len(required)} '
                           f'Range Attention parameter/buffer keys: {missing[:5]}')
    mismatched = [k for k in required if weights[k].shape != model.state_dict()[k].shape]
    if mismatched:
        raise RuntimeError(f'Checkpoint has incompatible Range Attention shapes: {mismatched[:5]}')
    model = model.cuda().eval()
    if not hasattr(model.pts_middle_encoder, 'range_attn'):
        raise TypeError('Configuration must build RadarEncV8small with .range_attn')
    return model


def extract_p0_p1(model, path, geometry):
    import torch
    n = 175*250
    with np.load(path, allow_pickle=False) as f:
        if not all(key in f for key in ('power_val', 'azimuth_ind', 'elevation_ind')):
            raise KeyError(f'NPZ missing expected sparse radar channels: {path}')
        channels = np.asarray(f['power_val'])
        az = np.asarray(f['azimuth_ind']).reshape(-1)
        el = np.asarray(f['elevation_ind']).reshape(-1)
        ra = (np.asarray(f['range_ind']).reshape(-1) if 'range_ind' in f
              else np.repeat(np.arange(175), 250))
        if channels.ndim != 2 or channels.shape[0] != 8 or channels.shape[1] < n or len(az) < n or len(el) < n or len(ra) < n:
            raise ValueError(f'Unexpected radar NPZ shape: power={channels.shape}, az={len(az)}')
        rdr = dict(power_val=torch.as_tensor(channels[:, :n][None], dtype=torch.float32, device='cuda'),
                   azimuth_ind=torch.as_tensor(az[:n][None], dtype=torch.long, device='cuda'),
                   elevation_ind=torch.as_tensor(el[:n][None], dtype=torch.long, device='cuda'),
                   range_ind=torch.as_tensor(ra[:n][None], dtype=torch.long, device='cuda'))
    with torch.inference_mode():
        p0, coors, bs = model._prepare_p1_frame(rdr)
        assert bs == 1 and p0.shape == (n, 8) and coors.shape == (n, 4)
        torch.cuda.synchronize()
        start = time.perf_counter()
        p1 = model.pts_middle_encoder.range_attn(p0, coors[:, 1], coors[:, 3])
        if p1.shape != (n, 8):
            raise ValueError(f'Unexpected P1 shape: {tuple(p1.shape)}')
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        p0 = p0.detach().cpu().numpy()
        p1 = p1.detach().cpu().numpy()
        coords = coors[:, [2, 3, 1]].cpu().numpy()  # [R,A,E] physical-bin lookup
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError(f'Radar P0/P1 produced NaN/Inf on {path}')
    xyz = geometry.to_xyz(coords)
    # RA consumes log10(power) for channels [0,1,2,6,7].  Use the same
    # transform as the pre-attention feature-score control.  Zero power is
    # handled as 0 only when the model configuration supports that convention.
    continuous = p0[:, [0, 1, 2, 6, 7]]
    if np.any(continuous < 0):
        raise ValueError('P0 continuous power must be nonnegative for log10 encoding')
    if np.any(continuous == 0) and not model.pts_middle_encoder.range_attn.allow_zero_power:
        raise ValueError('P0 has zero power but the model disallows zero-power log encoding')
    safe_continuous = np.where(continuous == 0, 1., continuous)
    pre_attention_input = np.log10(safe_continuous)
    pre_attention_input[continuous == 0] = 0.
    scores = dict(P0_power=p0[:, 2].copy(),
                  P0_descriptor=robust_descriptor_score(pre_attention_input),
                  P1_range_attention=robust_descriptor_score(p1[:, :5]))
    return xyz, scores, elapsed


def gt_statistics(result, xyz, gt, gt_space="xy", purity=.6, min_gt_points=3):
    report = oldvis.gt_diagnostics(result, xyz, gt, gt_space=gt_space,
                                   purity=purity, min_gt_points=min_gt_points)
    diagnostics = list(report['region_diagnostics'].values())
    labeled = [d for d in diagnostics if d['reason'] == 'labeled']
    accepted = sum(r['accepted'] for r in result['regions'])
    # Only count objects with enough returns inside the proposal analysis ROI.
    # In XY evaluation, an out-of-height return can still overlap a footprint.
    all_labels = np.asarray(report['point_labels'])[result['analysis_mask']]
    gt_ids = np.asarray(gt['ids']).astype(str)
    object_rows = [j for j in range(len(gt_ids)) if (all_labels == j).sum() >= min_gt_points
                   and not gt['duplicate_ids'][j]]
    detected = set(report['region_labels'].values())
    unique_detectable = {gt_ids[j] for j in object_rows}
    obj_hit = len(unique_detectable & detected)
    stats = dict(accepted=accepted, accepted_gt_labeled=len(labeled),
                gt_labeled_fraction=len(labeled)/accepted if accepted else 0.0,
                gt_detectable_objects=len(unique_detectable),
                gt_detected_objects=obj_hit,
                gt_object_coverage=obj_hit/len(unique_detectable) if unique_detectable else None,
                mean_dominant_fraction=float(np.mean([d['dominant_fraction'] for d in labeled])) if labeled else None,
                mean_dominant_fraction_all=float(np.mean([d['dominant_fraction'] for d in diagnostics])) if diagnostics else None,
                box_fraction_all_tokens=result['stats']['box_fraction_all_p1'],
                num_support_tokens=result['stats']['n_support'])
    for reason in ('no_support', 'no_gt_overlap', 'duplicate_id_only', 'ambiguous_dominant',
                   'insufficient_gt_points', 'low_purity'):
        stats[f'gt_rejected_{reason}'] = sum(d['reason'] == reason for d in diagnostics)
    return stats


def render_comparison(results, xyz, scores, output, title, gt=None, rgb_path=None, gt_space='xy'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from scipy.spatial import ConvexHull

    if rgb_path is None:
        fig, axes = plt.subplots(3, 3, figsize=(21, 14), layout='constrained')
    else:
        from PIL import Image
        fig = plt.figure(figsize=(21, 18), layout='constrained')
        grid = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1])
        rgb_ax = fig.add_subplot(grid[0, :])
        with Image.open(rgb_path) as rgb_img:
            rgb_ax.imshow(rgb_img.convert('RGB'))
        rgb_ax.axis('off')
        rgb_ax.set_title(f'RGB reference (not used to propose regions): {Path(rgb_path).name}')
        axes = np.array([[fig.add_subplot(grid[row+1, col]) for col in range(3)]
                         for row in range(3)])
    labels = [('P0_power', 'Before RA: original power'),
              ('P0_descriptor', 'Before RA: robust descriptor score'),
              ('P1_range_attention', 'After RA: robust descriptor score')]
    shown = np.linspace(0, len(xyz)-1, min(12000, len(xyz)), dtype=int)
    gt_hulls = []
    gt_points = np.array([], dtype=int)
    if gt is not None:
        point_labels = oldvis.gt_memberships(xyz, gt['boxes'], gt['radar_to_lidar'], gt_space)
        roi = results['P0_power']['analysis_mask']
        gt_points = np.flatnonzero(roi & (point_labels >= 0))
        for corners in gt['corners']:
            xy = corners[:, :2]
            gt_hulls.append(xy[ConvexHull(xy).vertices])
    for col, (key, label) in enumerate(labels):
        res, score = results[key], scores[key]
        ax = axes[0,col]
        art = ax.scatter(xyz[shown,0], xyz[shown,1], c=score[shown], s=1.5,
                         cmap='viridis', rasterized=True, vmin=float(np.quantile(score, .02)),
                         vmax=float(np.quantile(score, .99)))
        fig.colorbar(art, ax=ax, shrink=.65)
        ax.set_title(label + '\nScalar score (not calibrated radar power)' if key != 'P0_power'
                     else label + '\nOriginal channel 2 power')
        accepted = [r for r in res['regions'] if r['accepted']]
        supports = (np.unique(np.concatenate([r['support_point_ids'] for r in accepted])).astype(int)
                    if accepted else np.array([], dtype=int))
        for row in (1,2):
            a = axes[row,col]
            a.scatter(xyz[shown,0], xyz[shown,1], c='#bbbbbb', s=.8, alpha=.35, rasterized=True)
            # Never downsample the proposal evidence or the GT-contained returns.
            a.scatter(xyz[gt_points,0], xyz[gt_points,1], facecolors='none',
                      edgecolors='#00985a', s=16, linewidths=.7, rasterized=True)
            a.scatter(xyz[supports,0], xyz[supports,1], c='#dc651a', s=3,
                      alpha=.8, rasterized=True)
            for region in accepted:
                low, high = region['box_min'], region['box_max']
                a.add_patch(Rectangle(low[:2], *(high[:2]-low[:2]), fill=False,
                                      edgecolor='#dc651a', linewidth=1))
                if row == 2:
                    a.text(low[0], high[1], f'R{region["id"]}', fontsize=6,
                           color='#91400c', clip_on=True)
            if gt is not None:
                for hull in gt_hulls:
                    a.add_patch(Polygon(hull, closed=True, fill=False,
                                        edgecolor='#00985a', linewidth=1.2))
            a.set_title(f'{"Full ROI" if row == 1 else "Forward-area detail"}: '
                        f'{len(accepted)} accepted regions')
        for row, ax in enumerate(axes[:,col]):
            ax.set_xlim(0, 85 if row != 2 else 60)
            ax.set_ylim((-60,60) if row != 2 else (-10,10))
            ax.set_xlabel('X forward (m)')
            ax.set_ylabel('Y left (m)')
            ax.grid(alpha=.12)
            ax.set_aspect('equal', adjustable='box')
    fig.suptitle(title + '\nOrange: proposed boxes + actual support points | Green: GT footprints + contained returns'
                 '\nGray: subsampled context only; orange/green evidence points are not downsampled. GT is evaluation only.'
                 '\nP0 descriptor and P1 descriptor use identical per-range robust scoring.', fontsize=13)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=Path, default=Path('projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--ann-file', type=Path, help='Temporal split PKL; auto-locate if omitted')
    p.add_argument('--scene', default='3')
    p.add_argument('--start-order', type=int, default=80)
    p.add_argument('--end-order', type=int, default=200)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--show-gt', action='store_true')
    p.add_argument('--gt-space', choices=('xy', 'xyz'), default='xy',
                   help='Evaluate 2D proposals against GT footprints (xy) or full cuboids (xyz)')
    p.add_argument('--gt-purity', type=float, default=.6,
                   help='Minimum fraction of proposal support belonging to its dominant GT object')
    p.add_argument('--min-gt-points', type=int, default=3,
                   help='Minimum unique-object returns for GT labeling and ROI detectability')
    p.add_argument('--power-space', choices=('xy', 'xyz'),
                   help='Box inclusion for per-box power statistics; defaults to --gt-space. '
                        'xy uses footprints; xyz uses full cuboids. Does not change proposals or overlays.')
    p.add_argument('--with-rgb', action='store_true', help='Include paired RGB reference when available')
    p.add_argument('--camera-dir', type=Path, help='RGB directory, if scene has no camera metadata')
    p.add_argument('--gif', action='store_true', help='Generate animation from the saved comparison panels')
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/p0_vs_p1_range_boxes'))
    args = p.parse_args()
    if args.stride <= 0 or args.max_frames < 0 or args.start_order > args.end_order:
        p.error('Require stride > 0, max-frames >= 0 and start-order <= end-order')
    if not 0 <= args.gt_purity <= 1 or args.min_gt_points < 1:
        p.error('Require 0 <= gt-purity <= 1 and min-gt-points >= 1')
    if args.power_space is None:
        args.power_space = args.gt_space
    root = ROOT
    resolved = lambda x: x if x.is_absolute() else root / x
    args.output_dir = resolved(args.output_dir)
    ann, scene = oldvis.load_scene(root, args.ann_file, args.scene)
    chosen = [i for i in scene if args.start_order <= int(i['order_in_scene']) <= args.end_order
              and (int(i['order_in_scene'])-args.start_order) % args.stride == 0]
    if args.max_frames:
        chosen = chosen[:args.max_frames]
    if not chosen:
        raise ValueError('No selected frames; check scene/orders and annotation split')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose an empty --output-dir to avoid mixing results')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_frozen_range_module(resolved(args.config), resolved(args.checkpoint))
    geometry = alignment.RadarBinGeometry.from_file(resolved(args.bins))
    cfg = RegionConfig(seed_percentile=97, seed_mad=3, merge_space='xy',
                       xy_box_gap=(.5,.5), box_size=(6,3,3), max_box_size=(20,6,6))
    extrinsic = alignment.geometry_module.radar_to_lidar_default()
    rows = []
    power_box_rows, power_frame_rows = [], []
    frame_images = []
    for index, info in enumerate(chosen):
        order = int(info['order_in_scene'])
        path = resolved(Path(info.get('sparse_radar_path') or info['radar_path']))
        xyz, scores, secs = extract_p0_p1(model, path, geometry)
        results = {key:extract_regions(xyz, value, cfg) for key,value in scores.items()}
        gt = oldvis.load_gt(info, args.scene, resolved(args.bbox_root), extrinsic) if args.show_gt else None
        box_power, frame_power = frame_power_statistics(results, xyz, scores['P0_power'],
                                                       gt=gt, space=args.power_space)
        power_context = dict(scene=args.scene, order=order,
                             radar_frame_id=info.get('radar_frame_id'),
                             occ_frame_idx=info.get('occ_frame_idx'), radar_path=str(path))
        power_box_rows.extend(dict(**power_context, **r) for r in box_power)
        power_frame_rows.extend(dict(**power_context, **r) for r in frame_power)
        out_img = args.output_dir / f'scene{args.scene}_order{order:05d}.png'
        rgb_path = None
        if args.with_rgb:
            camera_dir = oldvis.camera_directory(root, str(args.scene), args.camera_dir)
            normalized = dict(info)
            if info.get('cams'):
                normalized['cams'] = {k: {**v, 'data_path': str(resolved(Path(v['data_path'])))}
                                      for k, v in info['cams'].items()}
            rgb_path, _ = alignment.resolve_rgb(normalized, scene, camera_dir=camera_dir)
        render_comparison(results, xyz, scores, out_img,
                          f'Scene {args.scene} order {order} | Range Attention {secs*1000:.1f} ms',
                          gt, rgb_path=rgb_path, gt_space=args.gt_space)
        frame_images.append(out_img)
        for key,res in results.items():
            row = dict(scene=args.scene, order=order, method=key,
                       accepted=res['stats']['n_accepted'],
                       box_fraction_all_tokens=res['stats']['box_fraction_all_p1'],
                       num_support_tokens=res['stats']['n_support'],
                       range_attention_ms=secs*1000,
                       image=out_img.name)
            if gt is not None:
                row.update(gt_statistics(res, xyz, gt, gt_space=args.gt_space,
                                         purity=args.gt_purity, min_gt_points=args.min_gt_points))
            rows.append(row)
        print(f'{index+1}/{len(chosen)} order={order} '
              + ' '.join(f'{k}:{v["stats"]["n_accepted"]}' for k,v in results.items())
              + f' RA={secs*1000:.1f}ms', flush=True)
    columns = list(dict.fromkeys(c for r in rows for c in r))
    with (args.output_dir/'frame_metrics.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for name in ('P0_power','P0_descriptor','P1_range_attention'):
        sample=[r for r in rows if r['method']==name]
        record=dict(method=name, frames=len(sample))
        for key in ('accepted', 'box_fraction_all_tokens','num_support_tokens',
                    'gt_labeled_fraction','gt_object_coverage','mean_dominant_fraction',
                    'mean_dominant_fraction_all'):
            vals=[r[key] for r in sample if key in r and r[key] is not None]
            record[key]=float(np.mean(vals)) if vals else None
        if args.show_gt:
            labeled = sum(r['accepted_gt_labeled'] for r in sample)
            accepted = sum(r['accepted'] for r in sample)
            detectable = sum(r['gt_detectable_objects'] for r in sample)
            detected = sum(r['gt_detected_objects'] for r in sample)
            record.update(total_accepted=accepted, total_gt_labeled=labeled,
                          total_gt_detectable_objects=detectable, total_gt_detected_objects=detected,
                          micro_gt_labeled_fraction=labeled/accepted if accepted else None,
                          micro_gt_object_coverage=detected/detectable if detectable else None)
        summary.append(record)
    cols=list(summary[0])
    with (args.output_dir/'summary.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=cols);writer.writeheader();writer.writerows(summary)
    write_power_statistics(args.output_dir, power_box_rows, power_frame_rows)
    (args.output_dir/'experiment.json').write_text(json.dumps(dict(
        experiment='P0 power vs P0 feature-score control vs P1 Range-Attention feature-score',
        checkpoint=str(resolved(args.checkpoint)), config=str(resolved(args.config)),
        resolved_ann_file=str(ann), scene=str(args.scene),
        selected_orders=[int(info['order_in_scene']) for info in chosen],
        bbox_root=str(resolved(args.bbox_root)),
        gt_only_for_evaluation=True, no_model_training=True, samples=len(chosen),
        score='per-range robust activation RMS; MAD scale with mean-absolute-deviation fallback for nonconstant zero-MAD channels; P0 on log10 pre-attention channels, P1 on learned channels, excluding Doppler indices; proxy not calibrated object confidence',
        region_config=cfg.__dict__, gt_space=args.gt_space if args.show_gt else None,
        gt_purity=args.gt_purity if args.show_gt else None,
        min_gt_points=args.min_gt_points if args.show_gt else None,
        power_statistics=dict(
            primary_table='all_frames_box_power.csv',
            row_key='scene/order/method/box_index/space; recurring or duplicate GT IDs are not merged',
            source='Original NPZ power_val[2], linear stored scale, same values for all methods; not calibrated dB',
            space=args.power_space, roi='Same analysis ROI as proposal extraction',
            box='All ROI returns inside the box in the selected space',
            support='Actual proposal support returns, reported separately',
            box_inside_gt='Proposal box returns inside the union of ALL GT boxes',
            box_outside_gt='Proposal box returns outside the union of ALL GT boxes; not necessarily noise',
            overlaps='Per-box CSV can repeat returns in overlapping boxes; frame and summary means deduplicate within each method/frame/point_set',
            mean_power='In the primary table: arithmetic mean of original power in this single box only',
            min_power='Observed minimum original power in this single box; blank for no returns',
            max_power='Observed maximum original power in this single box; blank for no returns',
            pooled_reference='power_summary.csv alone pools point sums and counts across frames; primary per-box rows stay separate',
            macro_box_mean_power='Equal-weight average of nonempty individual box means; separate from pooled point mean'),
        metric_definitions=dict(
            gt_labeled_fraction='Accepted proposals passing the configured dominant-object purity and count thresholds / accepted proposals',
            gt_object_coverage='Distinct detected GT IDs / distinct unambiguous GT IDs with at least min_gt_points returns in analysis ROI',
            mean_dominant_fraction='Mean dominant-object support fraction over GT-labeled proposals only',
            mean_dominant_fraction_all='Mean dominant-object support fraction over ALL accepted proposals, including unlabeled proposals',
            summary='Unprefixed ratios are per-frame macro averages; micro_* ratios pool frame counts; objects recurring across frames count once per frame'),
        interpretation='Matched scene and xyz. P0-to-P1 feature comparison measures score-dependent proposal changes, not proof of sidelobe suppression.'
    ),indent=2,default=str)+'\n')
    if args.gif:
        from PIL import Image
        frames=[]
        for frame_path in frame_images:
            with Image.open(frame_path) as im:
                copy=im.convert('RGB')
                copy.thumbnail((1280,960))
                frames.append(copy.quantize(colors=128))
        frames[0].save(args.output_dir/'comparison.gif', save_all=True,
                       append_images=frames[1:], duration=200, loop=0)
    print('Visualization and metrics:',args.output_dir.resolve(),flush=True)
    print('Per-frame, per-box power table:', (args.output_dir/'all_frames_box_power.csv').resolve(), flush=True)

if __name__=='__main__':
    main()
