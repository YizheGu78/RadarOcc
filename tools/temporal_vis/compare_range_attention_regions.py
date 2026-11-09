#!/usr/bin/env python3
"""Compare GT-free P0 vs post-Range-Attention P1 radar reflection proposals.

Uses one frozen single-frame RadarOcc checkpoint. All modes operate on identical
P0 token locations and identical region extraction settings. The score-proxy
control (P0-feature vs P1-feature) uses the same robust-score rule,
while feature values and distributions differ. It is not a learned detector. GT is used for evaluation only.
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
    mad = np.median(np.abs(x-med), axis=1, keepdims=True)
    # Reject numerical-constant channels rather than manufacturing saliency.
    dynamic = mad > (1e-8*np.maximum(1.0, np.abs(med)))
    denom = np.where(dynamic, 1.4826*mad, 1.0)
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
    loaded_keys = {k.removeprefix('module.') for k in raw_weights}
    required = {k for k in model.state_dict() if k.startswith('pts_middle_encoder.range_attn.')}
    missing = sorted(required - loaded_keys)
    if not required or missing:
        raise RuntimeError(f'Checkpoint is missing {len(missing)} of {len(required)} '
                           f'Range Attention parameter/buffer keys: {missing[:5]}')
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


def gt_statistics(result, xyz, gt, gt_space="xy"):
    report = oldvis.gt_diagnostics(result, xyz, gt, gt_space=gt_space)
    diagnostics = list(report['region_diagnostics'].values())
    labeled = [d for d in diagnostics if d['reason'] == 'labeled']
    accepted = sum(r['accepted'] for r in result['regions'])
    all_labels = np.asarray(report['point_labels'])
    gt_ids = np.asarray(gt['ids']).astype(str)
    object_rows = [j for j in range(len(gt_ids)) if (all_labels == j).sum() >= 3
                   and not gt['duplicate_ids'][j]]
    detected = set(report['region_labels'].values())
    unique_detectable = {gt_ids[j] for j in object_rows}
    obj_hit = len(unique_detectable & detected)
    return dict(accepted=accepted, accepted_gt_labeled=len(labeled),
                gt_labeled_fraction=len(labeled)/accepted if accepted else 0.0,
                gt_detectable_objects=len(unique_detectable),
                gt_object_coverage=obj_hit/len(unique_detectable) if unique_detectable else None,
                mean_dominant_fraction=float(np.mean([d['dominant_fraction'] for d in labeled])) if labeled else None,
                box_fraction_all_tokens=result['stats']['box_fraction_all_p1'],
                num_support_tokens=result['stats']['n_support'])


def render_comparison(results, xyz, scores, output, title, gt=None, rgb_path=None):
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
    labels = [('P0_power', 'P0 original power'),
              ('P0_descriptor', 'P0 robust descriptor score'),
              ('P1_range_attention', 'P1 Range Attention descriptor score')]
    shown = np.linspace(0, len(xyz)-1, min(12000, len(xyz)), dtype=int)
    gt_hulls = []
    if gt is not None:
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
        for row in (1,2):
            a = axes[row,col]
            a.scatter(xyz[shown,0], xyz[shown,1], c='#bbbbbb', s=.8, alpha=.35, rasterized=True)
            accepted = [r for r in res['regions'] if r['accepted']]
            for region in accepted:
                low, high = region['box_min'], region['box_max']
                a.add_patch(Rectangle(low[:2], *(high[:2]-low[:2]), fill=False,
                                      edgecolor='#dc651a', linewidth=1))
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
    fig.suptitle(title + '\nOrange: GT-free proposed boxes | Green: GT XY footprints (evaluation only)'
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
    p.add_argument('--with-rgb', action='store_true', help='Include paired RGB reference when available')
    p.add_argument('--camera-dir', type=Path, help='RGB directory, if scene has no camera metadata')
    p.add_argument('--gif', action='store_true', help='Generate animation from the saved comparison panels')
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/p0_vs_p1_range_boxes'))
    args = p.parse_args()
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
    frame_images = []
    for index, info in enumerate(chosen):
        order = int(info['order_in_scene'])
        path = resolved(Path(info.get('sparse_radar_path') or info['radar_path']))
        xyz, scores, secs = extract_p0_p1(model, path, geometry)
        results = {key:extract_regions(xyz, value, cfg) for key,value in scores.items()}
        gt = oldvis.load_gt(info, args.scene, resolved(args.bbox_root), extrinsic) if args.show_gt else None
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
                          gt, rgb_path=rgb_path)
        frame_images.append(out_img)
        for key,res in results.items():
            row = dict(scene=args.scene, order=order, method=key,
                       accepted=res['stats']['n_accepted'],
                       box_fraction_all_tokens=res['stats']['box_fraction_all_p1'],
                       num_support_tokens=res['stats']['n_support'],
                       range_attention_ms=secs*1000,
                       image=out_img.name)
            if gt is not None:
                row.update(gt_statistics(res, xyz, gt, gt_space=args.gt_space))
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
                    'gt_labeled_fraction','gt_object_coverage','mean_dominant_fraction'):
            vals=[r[key] for r in sample if key in r and r[key] is not None]
            record[key]=float(np.mean(vals)) if vals else None
        summary.append(record)
    cols=list(summary[0])
    with (args.output_dir/'summary.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=cols);writer.writeheader();writer.writerows(summary)
    (args.output_dir/'experiment.json').write_text(json.dumps(dict(
        experiment='P0 power vs P0 feature-score control vs P1 Range-Attention feature-score',
        checkpoint=str(resolved(args.checkpoint)), config=str(resolved(args.config)),
        gt_only_for_evaluation=True, no_model_training=True, samples=len(chosen),
        score='per-range robust activation RMS; P0 on log10 pre-attention channels, P1 on learned channels, excluding Doppler indices; proxy not calibrated object confidence',
        region_config=cfg.__dict__, gt_space=args.gt_space if args.show_gt else None,
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

if __name__=='__main__':
    main()