#!/usr/bin/env python3
"""Render P1 before/after ego alignment without a checkpoint or CUDA."""
import argparse
import importlib.util
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("radar_alignment_geometry", ROOT / "temporal/core/ego_alignment.py")
geometry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_module)
RadarBinGeometry = geometry_module.RadarBinGeometry
align_to_current = geometry_module.align_to_current


def read_infos(path):
    with Path(path).open('rb') as stream:
        data = pickle.load(stream)
    return data['infos'] if isinstance(data, dict) else data


def scene_infos(infos, scene_id):
    return sorted((i for i in infos if str(i.get('scene_id', i.get('scene_token'))) == str(scene_id)),
                  key=lambda i: i['order_in_scene'] if 'order_in_scene' in i else i.get('timestamp', 0))


def resolve_rgb(info, reference_scene, camera_dir=None, camera_offset=0, rgb_image=None, camera_name=None):
    """Use stored association, or the repository's scene-order camera mapping."""
    if rgb_image:
        path = Path(rgb_image).expanduser().resolve()
        mapping = dict(method='explicit_image', path=str(path))
    elif info.get('cams'):
        cams = info['cams']
        if camera_name:
            if camera_name not in cams:
                raise ValueError(f'Camera {camera_name} absent; choices: {list(cams)}')
            name = camera_name
        elif len(cams) == 1:
            name = next(iter(cams))
        else:
            raise ValueError(f'Multiple cameras: specify --camera-name from {list(cams)}')
        path = Path(cams[name]['data_path']).expanduser().resolve()
        mapping = dict(method='metadata', camera=name, path=str(path))
    else:
        if camera_dir is None:
            raise ValueError('Metadata has no camera path. Supply --camera-dir for this scene, or --rgb-image.')
        directory = Path(camera_dir).expanduser().resolve()
        def natural_key(path):
            return tuple((0, int(s)) if s.isdigit() else (1, s.lower()) for s in re.split(r'(\d+)', path.name))
        files = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in
                        {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}), key=natural_key)
        if not files:
            raise FileNotFoundError(f'No RGB images in {directory}')
        if camera_offset == 0 and len(files) != len(reference_scene):
            raise ValueError(f'RGB count {len(files)} != reference scene count {len(reference_scene)}. '
                             'For a subset split, pass --camera-reference-ann-file with the complete scene; '
                             'use --camera-offset only for a known sequence offset. No filename-ID matching is performed.')
        def identifier(x):
            value = x.get('lidar_token', x.get('token'))
            return str(value) if value is not None else (str(x.get('scene_id', x.get('scene_token'))), x['order_in_scene'])
        positions = [p for p, item in enumerate(reference_scene) if identifier(item) == identifier(info)]
        if len(positions) != 1:
            raise ValueError('Current frame absent or duplicated in camera reference annotation')
        ordinal = positions[0] + camera_offset
        if ordinal < 0 or ordinal >= len(files):
            raise IndexError(f'Camera ordinal {ordinal} outside 0..{len(files)-1}')
        path = files[ordinal]
        mapping = dict(method='scene_order', scene_ordinal=positions[0], camera_ordinal=ordinal,
                       camera_offset=camera_offset, camera_count=len(files), reference_count=len(reference_scene), path=str(path))
    if not path.is_file():
        raise FileNotFoundError(f'RGB image not found: {path}')
    return path, mapping


def show_rgb(ax, path):
    from PIL import Image
    with Image.open(path) as image:
        ax.imshow(np.asarray(image.convert('RGB')))
    ax.set_title(f'Current-frame RGB reference | {Path(path).name}')
    ax.axis('off')


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


def render(raw, aligned, mask, offsets, output, title, scores=None, max_points=2000, rgb_path=None):
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
    fig = plt.figure(figsize=(16, 16 if rgb_path else 12))
    grid = fig.add_gridspec(3 if rgb_path else 2, 2)
    start = 1 if rgb_path else 0
    if rgb_path:
        show_rgb(fig.add_subplot(grid[0, :]), rgb_path)
    axes = [fig.add_subplot(grid[start, 0]), fig.add_subplot(grid[start, 1]),
            fig.add_subplot(grid[start + 1, 0], projection='3d'), fig.add_subplot(grid[start + 1, 1], projection='3d')]
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
    lo[:2] = np.minimum(lo[:2], 0)
    hi[:2] = np.maximum(hi[:2], 0)
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


def render_power(raw, scores, mask, offsets, output, title, max_points=2000, rgb_path=None):
    """Per-frame local BEV, common stored-power scale, no extra logarithm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cols = 2
    rows = (len(raw) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 5 * rows + (4 if rgb_path else 0)), layout='constrained')
    grid = fig.add_gridspec(rows + bool(rgb_path), cols + 1, width_ratios=[1, 1, .045])
    start = int(bool(rgb_path))
    if rgb_path:
        show_rgb(fig.add_subplot(grid[0, :cols]), rgb_path)
    selected = [np.argsort(np.nan_to_num(score, nan=-np.inf), kind='stable')[-max_points:] for score in scores]
    finite_values = np.concatenate([scores[t, selected[t]][np.isfinite(scores[t, selected[t]])]
                                    for t in range(len(raw)) if mask[t]])
    if not finite_values.size:
        raise ValueError('No finite power scores to plot')
    vmin, vmax = float(finite_values.min()), float(finite_values.max())
    norm = Normalize(vmin, vmax if vmax > vmin else vmin + 1)
    points = np.concatenate([raw[t, selected[t]] for t in range(len(raw)) if mask[t]])
    lo, hi = points.min(0) - 1, points.max(0) + 1
    lo[:2] = np.minimum(lo[:2], 0)
    hi[:2] = np.maximum(hi[:2], 0)
    axes = []
    artist = None
    for t in range(rows * cols):
        ax = fig.add_subplot(grid[start + t // cols, t % cols])
        if t >= len(raw):
            ax.axis('off'); continue
        axes.append(ax)
        label = 'current t' if offsets[t] == 0 else f't{int(offsets[t]):+d}'
        if mask[t]:
            ids = selected[t]
            artist = ax.scatter(raw[t, ids, 0], raw[t, ids, 1], c=scores[t, ids], cmap='viridis', norm=norm, s=5)
            ax.set_title(f'{label} | local BEV ({len(ids)} returns)')
        else:
            ax.set_title(f'{label} | invalid history excluded')
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
        ax.set_xlabel('X forward (m)'); ax.set_ylabel('Y left (m)'); ax.grid(alpha=.2)
    colorbar_ax = fig.add_subplot(grid[start:, cols])
    fig.colorbar(artist, cax=colorbar_ax, label='Stored power channel 2 (shared scale)')
    fig.suptitle(title + '\nSingle-frame local BEV return strength')
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
    parser.add_argument('--camera-dir', type=Path, help='RGB directory for the selected scene; natural-order association')
    parser.add_argument('--camera-reference-ann-file', type=Path, help='Full-scene annotation when the chosen split is a subset')
    parser.add_argument('--camera-offset', type=int, default=0, help='Known ordinal offset, never a radar filename offset')
    parser.add_argument('--camera-name', help='Camera key if metadata contains multiple cameras')
    parser.add_argument('--rgb-image', type=Path, help='Explicit current RGB; also supported with --debug-file')
    parser.add_argument('--no-rgb', action='store_true', help='Explicitly render radar-only panels')
    args = parser.parse_args()
    if args.frame_nums < 1 or args.max_points < 1:
        parser.error('frame-nums and max-points must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_file:
        if not args.no_rgb and args.rgb_image is None:
            parser.error('--debug-file requires --rgb-image, or --no-rgb; debug dumps do not store frame identity')
        rgb_path = None
        if not args.no_rgb:
            rgb_path, _ = resolve_rgb({}, [], rgb_image=args.rgb_image)
        with np.load(args.debug_file, allow_pickle=False) as d:
            scores = d['power_scores'][0] if 'power_scores' in d else None
            render(d['coords_raw_xyz'][0], d['coords_aligned_xyz'][0], d['valid_mask'][0], d['frame_offsets'][0],
                   args.output_dir / (args.debug_file.stem + '.png'), args.debug_file.stem, scores=scores,
                   max_points=args.max_points, rgb_path=rgb_path)
            if scores is not None:
                render_power(d['coords_raw_xyz'][0], scores, d['valid_mask'][0], d['frame_offsets'][0],
                             args.output_dir / (args.debug_file.stem + '_power.png'), args.debug_file.stem,
                             max_points=args.max_points, rgb_path=rgb_path)
        return
    if args.bins is None:
        parser.error('--bins is required; no approximate bin spacing is assumed')
    infos = read_infos(args.ann_file)
    scene = scene_infos(infos, args.scene)
    positions = [p for p, i in enumerate(scene) if i['order_in_scene'] == args.current_order]
    if len(positions) != 1:
        raise ValueError(f'Scene {args.scene}/order {args.current_order} absent or duplicated in this split; available scenes: {sorted({str(i.get("scene_id", i.get("scene_token"))) for i in infos})}')
    pos = positions[0]
    rgb_path, rgb_mapping = None, None
    if not args.no_rgb:
        reference = scene_infos(read_infos(args.camera_reference_ann_file), args.scene) if args.camera_reference_ann_file else scene
        rgb_path, rgb_mapping = resolve_rgb(scene[pos], reference, args.camera_dir, args.camera_offset,
                                            args.rgb_image, args.camera_name)
        print(f'RGB association: {rgb_mapping}')
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
                        transforms=transforms, valid_mask=mask[None], frame_offsets=offsets[None], ego_poses=poses[None],
                        power_scores=np.stack([x[1] for x in loaded])[None])
    summary = dict(scene=args.scene, current_order=args.current_order, orders=[i['order_in_scene'] for i in sequence],
                   radar_frame_ids=[i.get('radar_frame_id') for i in sequence], valid_mask=mask.tolist(),
                   bins=str(args.bins), azimuth_sign=args.azimuth_sign, elevation_sign=args.elevation_sign,
                   radar_to_lidar=(extrinsic if extrinsic is not None else geometry_module.radar_to_lidar_default()).tolist(),
                   transforms=transforms[0].tolist(), points_per_frame=int(raw.shape[1]), rgb=rgb_mapping)
    Path(str(prefix) + '.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    render(raw, aligned[0], mask, offsets, str(prefix) + '.png', f'Scene {args.scene}, current order {args.current_order}',
           scores=np.stack([x[1] for x in loaded]), max_points=args.max_points, rgb_path=rgb_path)
    render_power(raw, np.stack([x[1] for x in loaded]), mask, offsets, str(prefix) + '_power.png',
                 f'Scene {args.scene}, current order {args.current_order}', max_points=args.max_points, rgb_path=rgb_path)
    print(f'Saved {prefix}.png/_power.png/.npz/.json; no model inference or training performed')


if __name__ == '__main__':
    main()
