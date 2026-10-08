#!/usr/bin/env python3
"""Render P1 before/after ego alignment without a checkpoint or CUDA."""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("radar_alignment_geometry", ROOT / "temporal/core/ego_alignment.py")
geometry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_module)
RadarBinGeometry = geometry_module.RadarBinGeometry
align_to_current = geometry_module.align_to_current


def load_p1(path):
    # Mirror RadarOcc_small._prepare_p1_frame: first 175 ranges * 250
    # candidates. k == original_k, so sorting changes order, not membership.
    with np.load(path, allow_pickle=False) as data:
        power = np.asarray(data['power_val'])
        count = 175 * 250
        if power.ndim != 2 or power.shape[0] < 3 or power.shape[1] < count:
            raise ValueError(f"{path}: expected [C,N] with N >= {count}")
        elevation = np.asarray(data['elevation_ind']).reshape(-1)[:count]
        azimuth = np.asarray(data['azimuth_ind']).reshape(-1)[:count]
        if len(elevation) != count or len(azimuth) != count:
            raise ValueError(f"{path}: truncated angular indices")
        ranges = np.repeat(np.arange(175), 250)
        coords = np.stack((ranges, azimuth, elevation), axis=-1)
        return coords, power[2, :count].copy()


def render(raw, aligned, mask, offsets, output, title, scores=None, max_points=2000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if raw.shape != aligned.shape or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError('Expected matching [T,N,3] raw/aligned coordinates')
    if len(mask) != len(raw) or len(offsets) != len(raw):
        raise ValueError('Mask/offset length disagrees with T')
    colors = plt.get_cmap('tab10')
    # Exactly the same selected point IDs are used on both sides.
    selected = []
    for slot in range(len(raw)):
        if scores is None:
            ids = np.linspace(0, len(raw[slot]) - 1, min(max_points, len(raw[slot])), dtype=int)
        else:
            ids = np.argsort(np.nan_to_num(scores[slot], nan=-np.inf), kind='stable')[-max_points:]
        selected.append(ids)
    fig = plt.figure(figsize=(16, 12))
    axes = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),
            fig.add_subplot(2, 2, 3, projection='3d'), fig.add_subplot(2, 2, 4, projection='3d')]
    visible = []
    for slot in range(len(raw)):
        if not mask[slot]:
            continue
        color = colors(slot % 10)
        label = 'current t' if offsets[slot] == 0 else f't{int(offsets[slot]):+d}'
        for column, points in enumerate((raw, aligned)):
            xyz = points[slot, selected[slot]]
            visible.append(xyz)
            axes[column].scatter(xyz[:, 0], xyz[:, 1], s=3, alpha=.5, color=color, label=label)
            axes[column + 2].scatter(*xyz.T, s=2, alpha=.5, color=color, label=label)
    if not visible:
        raise ValueError('No valid frames to render')
    combined = np.concatenate(visible)
    lo, hi = combined.min(0) - 1, combined.max(0) + 1
    for i, ax in enumerate(axes):
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel('X forward (m)'); ax.set_ylabel('Y left (m)')
        ax.set_title(('Before alignment' if i % 2 == 0 else 'After alignment in current radar frame') + (' | BEV' if i < 2 else ' | 3D'))
        if i < 2:
            ax.set_aspect('equal'); ax.grid(alpha=.2)
        else:
            ax.set_zlim(lo[2], hi[2]); ax.set_zlabel('Z up (m)'); ax.view_init(elev=25, azim=-65)
        ax.legend(markerscale=3, loc='upper right')
    fig.suptitle(title + '\nSame point IDs, axis limits and viewpoints on both sides; invalid history excluded')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', type=Path, default=ROOT / 'data/annotations/kradar_dict_val_official_temporal_doppler8.pkl')
    parser.add_argument('--bins', type=Path, help='Actual bin NPZ or K-Radar info_arr.mat; required unless --debug-file')
    parser.add_argument('--extrinsic', type=Path, help='Optional 4x4 T_lidar_from_radar NPY; default matches existing RadarOcc convention')
    parser.add_argument('--azimuth-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--elevation-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--scene', default='3')
    parser.add_argument('--current-order', type=int, default=80, help='order_in_scene, not radar filename ID')
    parser.add_argument('--frame-nums', type=int, default=4)
    parser.add_argument('--max-points', type=int, default=2000, help='Display only; geometry is computed for all P1 points')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/temporal_alignment')
    parser.add_argument('--debug-file', type=Path, help='Render a plugin-recorded alignment NPZ instead of loading metadata')
    args = parser.parse_args()
    if args.frame_nums < 1 or args.max_points < 1:
        parser.error('frame-nums and max-points must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_file:
        with np.load(args.debug_file, allow_pickle=False) as d:
            render(d['coords_raw_xyz'][0], d['coords_aligned_xyz'][0], d['valid_mask'][0], d['frame_offsets'][0],
                   args.output_dir / (args.debug_file.stem + '.png'), args.debug_file.stem, max_points=args.max_points)
        return
    if args.bins is None:
        parser.error('--bins is required; no approximate bin spacing is assumed')
    with args.ann_file.open('rb') as stream:
        data = pickle.load(stream)
    infos = data['infos'] if isinstance(data, dict) else data
    scene = sorted((i for i in infos if str(i.get('scene_id', i.get('scene_token'))) == args.scene), key=lambda i: i['order_in_scene'])
    positions = [p for p, i in enumerate(scene) if i['order_in_scene'] == args.current_order]
    if len(positions) != 1:
        raise ValueError(f'Scene {args.scene}/order {args.current_order} absent or duplicated in this split; available scenes: {sorted({str(i.get("scene_id", i.get("scene_token"))) for i in infos})}')
    pos = positions[0]
    slots = list(range(pos - args.frame_nums + 1, pos + 1))
    mask = np.asarray([p >= 0 for p in slots])
    sequence = [scene[max(p, 0)] for p in slots]
    for info, valid in zip(sequence, mask):
        if valid and not info.get('ego_pose_valid', False):
            raise ValueError('Invalid ego pose in selected sequence')
    geometry = RadarBinGeometry.from_file(args.bins, azimuth_sign=args.azimuth_sign, elevation_sign=args.elevation_sign)
    loaded = [load_p1(i.get('sparse_radar_path') or i['radar_path']) for i in sequence]
    raw = geometry.to_xyz(np.stack([x[0] for x in loaded]))
    poses = np.stack([i['ego_pose'] for i in sequence])
    extrinsic = np.load(args.extrinsic, allow_pickle=False) if args.extrinsic else None
    aligned, transforms = align_to_current(raw[None], poses[None], extrinsic, mask[None])
    offsets = np.arange(-args.frame_nums + 1, 1)
    prefix = args.output_dir / f'scene{args.scene}_order{args.current_order:05d}_T{args.frame_nums}'
    np.savez_compressed(str(prefix) + '.npz', coords_raw_xyz=raw[None], coords_aligned_xyz=aligned,
                        transforms=transforms, valid_mask=mask[None], frame_offsets=offsets[None], ego_poses=poses[None])
    summary = dict(scene=args.scene, current_order=args.current_order, orders=[i['order_in_scene'] for i in sequence],
                   radar_frame_ids=[i.get('radar_frame_id') for i in sequence], valid_mask=mask.tolist(),
                   bins=str(args.bins), azimuth_sign=args.azimuth_sign, elevation_sign=args.elevation_sign,
                   radar_to_lidar=(extrinsic if extrinsic is not None else geometry_module.radar_to_lidar_default()).tolist(),
                   transforms=transforms[0].tolist(), points_per_frame=int(raw.shape[1]))
    Path(str(prefix) + '.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    render(raw, aligned[0], mask, offsets, str(prefix) + '.png', f'Scene {args.scene}, current order {args.current_order}',
           scores=np.stack([x[1] for x in loaded]), max_points=args.max_points)
    print(f'Saved {prefix}.png/.npz/.json; no model inference or training performed')


if __name__ == '__main__':
    main()
