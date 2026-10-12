#!/usr/bin/env python3
"""Inspect near-origin masking, without modifying NPZs or running a model."""
import argparse
import csv
import html
import importlib.util
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ego_mask_geometry', ROOT / 'temporal/core/ego_alignment.py')
geometry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_module)


def natural_key(value):
    return [(0, int(v)) if v.isdigit() else (1, v.lower())
            for v in re.split(r'(\d+)', str(value))]


def read_infos(path):
    with path.open('rb') as stream:
        data = pickle.load(stream)
    if isinstance(data, dict):
        data = data.get('infos', data.get('data_list'))
    if not isinstance(data, list):
        raise ValueError(f'{path}: expected annotation info list')
    return data


def choose_frames(infos, requested_order=None):
    groups = {}
    for info in infos:
        scene = info.get('scene_id', info.get('scene_token'))
        if scene is None:
            raise ValueError('Annotation is missing scene_id/scene_token')
        groups.setdefault(str(scene), []).append(info)
    selected = []
    for scene in sorted(groups, key=natural_key):
        rows = sorted(groups[scene], key=lambda i: i.get('order_in_scene', i.get('timestamp', 0)))
        matches = ([i for i in rows if i.get('order_in_scene') == requested_order]
                   if requested_order is not None else [])
        if len(matches) > 1:
            raise ValueError(f'Scene {scene}: duplicated order {requested_order}')
        info = matches[0] if matches else rows[len(rows)//2]
        method = 'requested_order' if matches else 'middle_of_split_scene'
        selected.append((scene, info, len(rows), method))
    return selected


def resolve_radar(info, repo_root):
    value = info.get('sparse_radar_path', info.get('radar_path'))
    if not value:
        raise ValueError('Annotation is missing radar NPZ path')
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [repo_root/path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Radar NPZ missing: {value}')


def read_radar(path, geometry):
    with np.load(path, allow_pickle=False) as data:
        power = np.asarray(data['power_val'], dtype=np.float64).copy()
        coords = np.stack([np.asarray(data[k]).reshape(-1) for k in
                           ('range_ind', 'azimuth_ind', 'elevation_ind')], axis=-1)
    if power.shape != (8, len(coords)) or not np.isfinite(power).all():
        raise ValueError(f'{path}: expected finite power_val [8,N]')
    if (power[[0, 1, 2, 6, 7]] < 0).any():
        raise ValueError(f'{path}: negative power/statistic channel')
    return geometry.to_xyz(coords), power, coords


def zero_ego_power(xyz, power, radius_m):
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('radius_m must be finite and positive')
    # Raw local radar XY, before any ego-pose transform. All elevations included.
    mask = np.hypot(xyz[:, 0], xyz[:, 1]) <= radius_m
    after = power.copy()
    after[np.ix_([0, 1, 2, 6, 7], np.flatnonzero(mask))] = 0
    return after, mask


def raster(xyz, scores, bounds, resolution):
    x0, x1, y0, y1 = bounds
    shape = (int(np.ceil((y1-y0)/resolution)), int(np.ceil((x1-x0)/resolution)))
    grid = np.zeros(shape)
    ix = np.floor((xyz[:, 0]-x0)/resolution).astype(int)
    iy = np.floor((xyz[:, 1]-y0)/resolution).astype(int)
    good = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    np.maximum.at(grid, (iy[good], ix[good]), np.log10(1+scores[good]))
    return grid


def render(xyz, before, after, radius, path, title, zoom_m, resolution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle

    xmax = max(51.2, float(xyz[:, 0].max()) + 1)
    ymax = max(25.6, float(np.abs(xyz[:, 1]).max()) + 1)
    full = (-3, xmax, -ymax, ymax)
    near = (-zoom_m, zoom_m, -zoom_m, zoom_m)
    norm = Normalize(0, max(float(np.log10(1+before).max()), 1))
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#eeeeee')
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for row, bounds in enumerate((full, near)):
        for col, scores in enumerate((before, after)):
            ax = axes[row, col]
            grid = raster(xyz, scores, bounds, resolution if row == 0 else min(resolution, .05))
            im = ax.imshow(np.ma.masked_equal(grid, 0), origin='lower', extent=bounds,
                           cmap=cmap, norm=norm, interpolation='nearest', aspect='equal')
            ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor='cyan', linewidth=1.5))
            ax.plot(0, 0, '+', color='cyan', markersize=6)
            ax.set(xlabel='Radar X (m)', ylabel='Radar Y (m)',
                   title=('Original power' if col == 0 else f'Power zeroed inside {radius:g} m')
                         + (' | full view' if row == 0 else ' | near-origin view'))
    fig.colorbar(im, ax=axes, label='log10(1 + power_val[channel]); same scale on both sides', shrink=.75)
    fig.suptitle(title + '\nCyan circle: raw radar XY mask; grey: zero/no return', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=ROOT)
    p.add_argument('--ann-dir', type=Path, default=Path('data/annotations'))
    p.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--radii-m', type=float, nargs='+', default=[1.0], help='Candidate XY radii; not confirmed vehicle bounds')
    p.add_argument('--order', type=int, help='Prefer this order_in_scene; fall back to scene middle if absent')
    p.add_argument('--power-channel', type=int, choices=[0, 1, 2, 6, 7], default=2)
    p.add_argument('--zoom-m', type=float, default=8.0)
    p.add_argument('--resolution-m', type=float, default=.1)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/ego_power_mask_audit'))
    args = p.parse_args()
    if (not all(np.isfinite(r) and r > 0 for r in args.radii_m)
            or not np.isfinite(args.zoom_m) or args.zoom_m <= max(args.radii_m)
            or not np.isfinite(args.resolution_m) or args.resolution_m <= 0):
        p.error('Require positive finite radii/resolution and zoom-m greater than all radii')
    args.repo_root = args.repo_root.resolve()
    def rooted(path):
        return path if path.is_absolute() else args.repo_root/path
    output = rooted(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = geometry_module.RadarBinGeometry.from_file(rooted(args.bins))
    rows = []
    for split in args.splits:
        ann = rooted(args.ann_dir)/f'kradar_dict_{split}_official_temporal_doppler8.pkl'
        infos = read_infos(ann)
        selected = choose_frames(infos, args.order)
        print(f'{split}: {len(infos)} annotation frames, {len(selected)} scenes')
        directory = output/split
        directory.mkdir(exist_ok=True)
        for scene, info, count, method in selected:
            path = resolve_radar(info, args.repo_root)
            xyz, power, indices = read_radar(path, geometry)
            token = str(info.get('lidar_token', info.get('token', path.stem)))
            for radius in args.radii_m:
                after, mask = zero_ego_power(xyz, power, radius)
                filename = f'scene{scene}_r{radius:g}m.png'
                destination = directory/filename
                render(xyz, power[args.power_channel], after[args.power_channel], radius,
                       destination, f'{split} | scene {scene} | token {token} | channel {args.power_channel}',
                       args.zoom_m, args.resolution_m)
                record = dict(split=split, scene=scene, token=token,
                    order_in_scene=info.get('order_in_scene'), scene_split_frames=count,
                    selection=method, radius_m=radius, power_channel=args.power_channel,
                    total_returns=len(mask), masked_returns=int(mask.sum()),
                    masked_fraction=float(mask.mean()), zero_range_returns=int((indices[:, 0] == 0).sum()),
                    masked_max_range_m=(float(np.linalg.norm(xyz[mask], axis=-1).max()) if mask.any() else 0),
                    radar_path=str(path), annotation=str(ann), image=str(destination))
                rows.append(record)
                print(f'  scene {scene}, r={radius:g}m: zeroed {mask.sum()}/{len(mask)} -> {destination}')
    if not rows:
        raise ValueError('No scene frames found')
    with (output/'manifest.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps(dict(
        bins=str(rooted(args.bins)), mask='hypot(raw_radar_x, raw_radar_y) <= radius_m',
        zeroed_channels=[0, 1, 2, 6, 7], preserved_doppler_id_channels=[3, 4, 5],
        scope='all NPZ returns, before P1 range cropping; one sampled frame per split/scene',
        source_files_modified=False, frames=rows), indent=2))
    cards = []
    for row in rows:
        relative = html.escape(str(Path(row['image']).relative_to(output)), quote=True)
        label = html.escape(f"{row['split']} | scene {row['scene']} | radius {row['radius_m']:g} m | token {row['token']}")
        cards.append(f'<figure><figcaption>{label}</figcaption><a href="{relative}"><img loading="lazy" src="{relative}"></a></figure>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Ego power mask audit</title>'
        '<style>body{font-family:sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:16px}'
        'figure{margin:0}img{width:100%}figcaption{padding:8px}</style>'
        '<h1>Original / zeroed power</h1><p>Candidate boundaries; one sampled frame per split/scene. Click to enlarge.</p>'
        '<main>' + ''.join(cards) + '</main>')
    print(f'Done: {output}; source NPZs untouched')


if __name__ == '__main__':
    main()
