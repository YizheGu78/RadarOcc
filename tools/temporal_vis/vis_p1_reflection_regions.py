#!/usr/bin/env python3
"""Inspect GT-free P1 reflection regions with associated RGB reference panels.

No model, checkpoint, CUDA, GT boxes, target IDs, or learned selector is used.
This is SINGLE-FRAME proposal inspection, not temporal tracking or fusion.
"""
import argparse
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as alignment
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions, npz_payload


def read_p1(path):
    """Mirror P1 membership, retaining original NPZ order and requiring Doppler8."""
    count = 175 * 250
    with np.load(path, allow_pickle=False) as data:
        power = np.asarray(data['power_val'])
        if power.ndim != 2 or power.shape[0] != 8 or power.shape[1] < count:
            raise ValueError(f'{path}: expected Doppler8 power_val [8,N], N >= {count}')
        azimuth = np.asarray(data['azimuth_ind']).reshape(-1)[:count]
        elevation = np.asarray(data['elevation_ind']).reshape(-1)[:count]
        if len(azimuth) != count or len(elevation) != count:
            raise ValueError(f'{path}: truncated P1 angular indices')
        coords = np.stack([np.repeat(np.arange(175), 250), azimuth, elevation], axis=-1)
        return coords, power[2, :count].copy()


def project_points(xyz, projection):
    projection = np.asarray(projection, float)
    if projection.shape != (3, 4) or not np.isfinite(projection).all() or np.linalg.matrix_rank(projection) < 3:
        raise ValueError('Projection must be a finite rank-3 [3,4] radar-to-pixel matrix')
    homogeneous = np.c_[xyz, np.ones(len(xyz))] @ projection.T
    depth = homogeneous[:, 2]
    uv = np.full((len(xyz), 2), np.nan)
    front = depth > 1e-6
    uv[front] = homogeneous[front, :2] / depth[front, None]
    return uv, depth


def project_box_edges(low, high, projection):
    """Clip edges at the camera near plane; never connect through negative depth."""
    bits = np.array(list(itertools.product((0, 1), repeat=3)))
    corners = np.where(bits, np.asarray(high), np.asarray(low))
    q = np.c_[corners, np.ones(8)] @ projection.T
    segments = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(bits[i] != bits[j]) != 1:
                continue
            a, b = q[i].copy(), q[j].copy()
            near = 1e-3
            if a[2] < near and b[2] < near:
                continue
            if a[2] < near:
                a += (b - a) * ((near - a[2]) / (b[2] - a[2]))
            elif b[2] < near:
                b += (a - b) * ((near - b[2]) / (a[2] - b[2]))
            segments.append(np.stack([a[:2] / a[2], b[:2] / b[2]]))
    return segments


def region_color(region):
    import matplotlib.pyplot as plt
    return plt.get_cmap('tab20')(region['id'] % 20) if region['accepted'] else (.5, .5, .5, .7)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def render(result, xyz, power, destination, title, rgb_path=None, projection=None,
           rgb_crop_width=0, max_points=10000, max_boxes=40,
           detail_x_limits=(0., 60.), detail_y_limits=(-10., 10.), power_scale='log'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    config = result['config']
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), layout='constrained')
    camera, intensity, seeds, before, boxes, detail = axes.ravel()
    selected_regions = sorted(result['regions'], key=lambda r: (not r['accepted'], r['id']))[:max_boxes]
    if rgb_path:
        with Image.open(rgb_path) as image:
            image = image.convert('RGB')
            if rgb_crop_width > 0:
                image = image.crop((0, 0, min(rgb_crop_width, image.width), image.height))
            camera.imshow(image)
            width, height = image.size
        if projection is not None:
            for region in selected_regions:
                if not region['accepted']:
                    continue
                color = region_color(region)
                for segment in project_box_edges(region['box_min'], region['box_max'], projection):
                    camera.plot(segment[:, 0], segment[:, 1], color=color, lw=1.1)
                uv, depth = project_points(region['center'][None], projection)
                if depth[0] > 0 and 0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height:
                    camera.text(*uv[0], f'R{region["id"]}', color=color, fontsize=9,
                                bbox=dict(facecolor='black', alpha=.4, pad=1))
        camera.set_xlim(-.5, width - .5)
        camera.set_ylim(height - .5, -.5)
        camera.set_title('RGB reference' + (' + projected proposal boxes' if projection is not None else ' (no projection calibration supplied)')
                         + '\n' + Path(rgb_path).name)
    else:
        camera.text(.5, .5, 'RGB disabled (--no-rgb)', ha='center', va='center', transform=camera.transAxes)
        camera.set_title('RGB reference')
    camera.axis('off')
    extent = [config['x_limits'][0], config['x_limits'][0] + result['raw_bev'].shape[0] * config['xy_resolution'],
              config['y_limits'][0], config['y_limits'][0] + result['raw_bev'].shape[1] * config['xy_resolution']]
    raster = np.where(np.isfinite(result['raw_bev']), result['raw_bev'], np.nan)
    finite = power[result['analysis_mask']]
    if power_scale == 'log':
        raster = np.sign(raster) * np.log10(1 + np.abs(raster))
        finite = np.sign(finite) * np.log10(1 + np.abs(finite))
    image = np.ma.masked_invalid(raster).T
    if len(finite):
        low, high = np.percentile(finite, [2, 99.5])
        artist = intensity.imshow(image, origin='lower', extent=extent, cmap='viridis',
            vmin=low, vmax=max(high, low + 1e-8), interpolation='nearest')
        fig.colorbar(artist, ax=intensity, fraction=.035,
                     label=('Signed log10(1+|power|)' if power_scale == 'log' else 'Stored power channel 2')
                     + ' | per-frame scale')
    intensity.set_title('P1 BEV power | all analysis-ROI returns\nDisplay compression does not change proposal extraction')
    seeds.imshow(result['support_bev'].T.astype(float), origin='lower', extent=extent,
                 cmap='Greys', vmin=0, vmax=2, interpolation='nearest')
    sx, sy = np.nonzero(result['seed_bev'])
    resolution = config['xy_resolution']
    seeds.scatter(config['x_limits'][0] + (sx + .5) * resolution,
                  config['y_limits'][0] + (sy + .5) * resolution,
                  color='orange', s=5, alpha=.6, label='High-threshold cells')
    peaks = result['peak_cells']
    if len(peaks):
        seeds.scatter(config['x_limits'][0] + (peaks[:, 0] + .5) * resolution,
                      config['y_limits'][0] + (peaks[:, 1] + .5) * resolution,
                      color='red', marker='+', s=40, label='Region seeds')
    seeds.set_title('Support cells + high-threshold cells + selected seeds\nClustering keeps Z layers separate')
    seeds.legend(fontsize=8, loc='upper right')
    ids = np.flatnonzero(result['analysis_mask'])
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
    def draw_regions(ax, regions, prefix, x_limits, y_limits, fine=False, show_height=False):
        background = ids[(xyz[ids, 0] >= x_limits[0]) & (xyz[ids, 0] <= x_limits[1])
                         & (xyz[ids, 1] >= y_limits[0]) & (xyz[ids, 1] <= y_limits[1])]
        ax.scatter(xyz[background, 0], xyz[background, 1], color='.8', s=1, alpha=.4)
        visible = [r for r in regions if r['box_min'][0] <= x_limits[1] and r['box_max'][0] >= x_limits[0]
                   and r['box_min'][1] <= y_limits[1] and r['box_max'][1] >= y_limits[0]]
        selected = sorted(visible, key=lambda r: (not r['accepted'], r['id']))[:max_boxes]
        label_positions = []
        for region in selected:
            color = region_color(region)
            members = region['support_point_ids']
            ax.scatter(xyz[members, 0], xyz[members, 1], color=color, s=6, alpha=.8)
            for low, high, style in ((region['observed_min'], region['observed_max'], '--'),
                                      (region['box_min'], region['box_max'], '-')):
                span = np.maximum(high[:2] - low[:2], .05)
                ax.add_patch(Rectangle(low[:2], *span, fill=False, edgecolor=color, linestyle=style, lw=1.2))
            label = f'{prefix}{region["id"]}'
            if not fine:
                label += f' n={region["source_seed_count"]}'
            if show_height:
                label += f' z={region["center"][2]:.1f}'
            # Keep labels inside zoom panels even when a box crosses a crop edge.
            position = np.clip(region['center'][:2], [x_limits[0], y_limits[0]], [x_limits[1], y_limits[1]])
            overlap = sum(np.linalg.norm(position - previous) < config['xy_resolution']
                          for previous in label_positions)
            label_positions.append(position)
            ax.annotate(label, xy=position, xytext=(2, 3 + 10 * overlap), textcoords='offset points',
                        color=color, fontsize=7, clip_on=True,
                        bbox=dict(facecolor='white', alpha=.7, pad=.3))
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        return len(selected), len(visible)

    fine_drawn, fine_visible = draw_regions(before, result['fine_regions'], 'S', config['x_limits'], config['y_limits'], fine=True)
    drawn, visible = draw_regions(boxes, result['regions'], 'R', config['x_limits'], config['y_limits'])
    zoom_drawn, zoom_visible = draw_regions(detail, result['regions'], 'R', detail_x_limits, detail_y_limits, show_height=True)
    box_label = 'Support-enclosing box' if config['merge_regions'] else 'Fixed-size box (legacy)'
    for ax in (before, boxes):
        ax.legend(handles=[Line2D([0], [0], color='black', ls='--', label='Observed support bounds'),
                           Line2D([0], [0], color='black', label=box_label if ax is boxes else 'Original fixed-size box'),
                           Line2D([0], [0], color='.5', label='Low-confidence proposal')], fontsize=7, loc='upper right')
    stats = result['stats']
    before.set_title(f'Before: {stats["n_fine_regions"]} seed regions | original fixed boxes\n'
                     f'Showing {fine_drawn}/{fine_visible}; S IDs identify reflection seeds')
    boxes.set_title(f'After: {stats["n_fine_regions"]} seeds -> {stats["n_regions"]} groups | '
                    f'{stats["n_accepted"]} accepted\n'
                    f'Showing {drawn}/{visible}; n = source seeds; gray = low confidence')
    detail.set_title(f'Forward-area detail | showing {zoom_drawn}/{zoom_visible} groups\n'
                     f'Accepted boxes contain {stats["n_box_points"]} P1 returns in full ROI')
    for ax in (intensity, seeds):
        ax.set_xlim(config['x_limits'])
        ax.set_ylim(config['y_limits'])
    for ax in (intensity, seeds, before, boxes, detail):
        ax.set_aspect('equal')
        ax.set_xlabel('X forward (m)')
        ax.set_ylabel('Y left (m)')
        ax.grid(alpha=.15)
    fig.suptitle(title + '\nGT-free, single-frame reflection groups; RGB is reference only; IDs are not tracks', fontsize=13)
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def camera_directory(root, scene, explicit):
    if explicit:
        return explicit if explicit.is_absolute() else root / explicit
    candidates = [root / 'data/K-Radar-RGB/K-Radar/K-Radar-RGB' / scene / 'images_rb_switched',
                  root / 'data/K-Radar-RGB' / scene / 'images_rb_switched']
    existing = [p for p in candidates if p.is_dir()]
    if len(existing) > 1:
        raise ValueError('Multiple RGB directories found; specify --camera-dir')
    return existing[0] if existing else None


def scene_key(value):
    """Metadata may store scene 3 as integer 3, '3', or '03'."""
    text = str(value)
    return str(int(text)) if text.isdigit() else text


def annotation_scenes(path):
    groups = {}
    for info in alignment.read_infos(path):
        key = scene_key(info.get('scene_id', info.get('scene_token')))
        groups.setdefault(key, []).append(info)
    for key, rows in groups.items():
        if any('order_in_scene' not in row for row in rows):
            raise ValueError(f'{path}: scene {key} lacks order_in_scene; use a temporal frame-level PKL')
        rows.sort(key=lambda row: int(row['order_in_scene']))
    return groups


def format_inventory(inventory):
    lines = []
    for path, groups in inventory:
        entries = ', '.join(f'{key} (n={len(rows)}, orders={rows[0]["order_in_scene"]}..{rows[-1]["order_in_scene"]})'
                            for key, rows in sorted(groups.items()))
        lines.append(f'{path}: {entries or "no scenes"}')
    return '\n'.join(lines)


def load_scene(root, ann_file, requested_scene, list_only=False):
    """Locate one full temporal split. Never merge independently numbered splits."""
    key = scene_key(requested_scene)
    if ann_file is not None:
        path = ann_file if ann_file.is_absolute() else root / ann_file
        paths = [path]
    else:
        paths = [root / 'data/annotations' / f'kradar_dict_{split}_official_temporal_doppler8.pkl'
                 for split in ('train', 'val', 'test')]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise FileNotFoundError('No full train/val/test temporal PKLs under data/annotations; pass --ann-file')
    inventory = [(path, annotation_scenes(path)) for path in paths]
    if list_only:
        print(format_inventory(inventory), flush=True)
        return None, None
    matches = [(path, groups[key]) for path, groups in inventory if key in groups]
    if not matches:
        raise ValueError(f'Scene {key} is absent from the selected annotation(s).\n'
                         + format_inventory(inventory)
                         + '\nOmit --ann-file to auto-locate the scene in full train/val/test PKLs, '
                         'or use --list-scenes to inspect the available orders.')
    if len(matches) > 1:
        raise ValueError(f'Scene {key} occurs in multiple splits; pass --ann-file explicitly. '
                         'Their order_in_scene values may be split-local and cannot be merged safely.\n'
                         + format_inventory(inventory))
    path, rows = matches[0]
    print(f'Using annotation: {path} | scene={key} frames={len(rows)} '
          f'orders={rows[0]["order_in_scene"]}..{rows[-1]["order_in_scene"]}', flush=True)
    return path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--ann-file', type=Path, help='Explicit temporal PKL; omit to locate scene in full train/val/test PKLs')
    parser.add_argument('--list-scenes', action='store_true', help='List annotation scenes/order ranges and exit without loading radar/RGB')
    parser.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    parser.add_argument('--scene', default='3')
    parser.add_argument('--start-order', type=int, default=80)
    parser.add_argument('--end-order', type=int, default=200)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=0, help='0 means all selected orders')
    parser.add_argument('--azimuth-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--elevation-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--xy-resolution', type=float, default=.5)
    parser.add_argument('--z-resolution', type=float, default=1.)
    parser.add_argument('--x-limits', type=float, nargs=2, default=[0., 85.])
    parser.add_argument('--y-limits', type=float, nargs=2, default=[-60., 60.])
    parser.add_argument('--z-limits', type=float, nargs=2, default=[-5., 8.])
    parser.add_argument('--range-band-m', type=float, default=8.)
    parser.add_argument('--seed-percentile', type=float, default=95.)
    parser.add_argument('--support-percentile', type=float, default=80.)
    parser.add_argument('--seed-mad', type=float, default=3.)
    parser.add_argument('--support-mad', type=float, default=.5)
    parser.add_argument('--peak-separation-m', type=float, default=1.5)
    parser.add_argument('--max-seeds', type=int, default=128)
    parser.add_argument('--grow-half-size-m', type=float, nargs=3, default=[4., 2., 2.])
    parser.add_argument('--box-size-m', type=float, nargs=3, default=[6., 3., 3.], help='Minimum XYZ box size; exact fixed size with --no-merge')
    parser.add_argument('--no-merge', action='store_true', help='Reproduce the original per-seed fixed-box extraction')
    parser.add_argument('--merge-gap-m', type=float, nargs=3, default=[1.5, 1., 1.], help='Maximum XYZ separation of support cell centers for linking')
    parser.add_argument('--merge-min-contact-cells', type=int, default=2, help='Minimum distinct closest-boundary cells on EACH side of a link')
    parser.add_argument('--max-box-size-m', type=float, nargs=3, default=[12., 4., 4.], help='Maximum XYZ box size for accepted merged groups')
    parser.add_argument('--box-padding-m', type=float, nargs=3, default=[.5, .25, .5], help='XYZ padding on each side of support bounds')
    parser.add_argument('--min-points', type=int, default=6)
    parser.add_argument('--min-cells', type=int, default=3)
    parser.add_argument('--max-aspect', type=float, default=8.)
    parser.add_argument('--camera-dir', type=Path)
    parser.add_argument('--camera-reference-ann-file', type=Path, help='Full scene PKL for ordinal RGB matching when using a subset')
    parser.add_argument('--camera-offset', type=int, default=0, help='Known camera sequence ordinal offset, never a radar filename offset')
    parser.add_argument('--camera-name')
    parser.add_argument('--rgb-image', type=Path, help='Explicit RGB for a single selected frame')
    parser.add_argument('--no-rgb', action='store_true')
    parser.add_argument('--projection-npy', type=Path, help='Optional calibrated [3,4] radar-XYZ to pixel matrix matching this image/crop')
    parser.add_argument('--rgb-crop-width', type=int, default=0, help='0 keeps the full RGB image; positive keeps left columns')
    parser.add_argument('--max-points', type=int, default=10000, help='Background scatter display only; never limits extraction')
    parser.add_argument('--max-draw-boxes', type=int, default=40, help='Drawing only; all proposals are exported')
    parser.add_argument('--detail-x-limits', type=float, nargs=2, default=[0., 60.])
    parser.add_argument('--detail-y-limits', type=float, nargs=2, default=[-10., 10.])
    parser.add_argument('--power-scale', choices=('log', 'linear'), default='log', help='Display only; extraction always uses original power')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--fps', type=float, default=5.)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/p1_reflection_regions'))
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing an existing diagnostic run')
    args = parser.parse_args()
    if (args.end_order < args.start_order or args.stride < 1 or args.max_frames < 0
            or min(args.max_points, args.max_draw_boxes) < 1 or args.rgb_crop_width < 0
            or not np.isfinite(args.fps) or args.fps <= 0
            or any(not np.isfinite(v).all() or v[0] >= v[1] for v in (args.detail_x_limits, args.detail_y_limits))):
        parser.error('Check order range, stride, frame/display counts, crop and fps')
    if args.no_rgb and (args.projection_npy or args.rgb_image or args.camera_dir):
        parser.error('--no-rgb cannot be combined with RGB image/directory/projection options')
    try:
        config = RegionConfig(xy_resolution=args.xy_resolution, z_resolution=args.z_resolution,
            x_limits=tuple(args.x_limits), y_limits=tuple(args.y_limits), z_limits=tuple(args.z_limits),
            range_band_m=args.range_band_m, seed_percentile=args.seed_percentile,
            support_percentile=args.support_percentile, seed_mad=args.seed_mad, support_mad=args.support_mad,
            peak_separation_m=args.peak_separation_m, max_seed_count=args.max_seeds,
            grow_half_size=tuple(args.grow_half_size_m), box_size=tuple(args.box_size_m),
            merge_regions=not args.no_merge, merge_gap=tuple(args.merge_gap_m),
            merge_min_contact_cells=args.merge_min_contact_cells,
            max_box_size=tuple(args.max_box_size_m), box_padding=tuple(args.box_padding_m),
            min_points=args.min_points, min_cells=args.min_cells, max_aspect=args.max_aspect).validate()
    except ValueError as exc:
        parser.error(str(exc))
    root = args.repo_root.resolve()
    rooted = lambda p: p if p.is_absolute() else root / p
    ann_file, scene = load_scene(root, args.ann_file, args.scene, args.list_scenes)
    if args.list_scenes:
        return
    args.scene = scene_key(args.scene)
    bins = rooted(args.bins)
    orders = [int(i['order_in_scene']) for i in scene]
    if len(set(orders)) != len(orders):
        raise ValueError('Duplicate scene/order records in annotation')
    selected = [i for i in scene if args.start_order <= int(i['order_in_scene']) <= args.end_order
                and (int(i['order_in_scene']) - args.start_order) % args.stride == 0]
    if args.max_frames:
        selected = selected[:args.max_frames]
    if not selected:
        raise ValueError(f'No selected orders for scene {args.scene} in {ann_file}. '
                         f'Requested orders {args.start_order}..{args.end_order}, stride={args.stride}; '
                         f'available orders {min(orders)}..{max(orders)} (n={len(orders)}). '
                         'Use --list-scenes; orders are annotation order_in_scene, not radar filename IDs.')
    if args.rgb_image and len(selected) != 1:
        parser.error('--rgb-image requires exactly one selected frame')
    reference = annotation_scenes(rooted(args.camera_reference_ann_file)).get(args.scene, []) if args.camera_reference_ann_file else scene
    camera_dir = None if args.no_rgb else camera_directory(root, str(args.scene), args.camera_dir)
    projection = None
    if args.projection_npy:
        projection = np.load(rooted(args.projection_npy), allow_pickle=False)
        project_points(np.zeros((0, 3)), projection)
    geometry = alignment.RadarBinGeometry.from_file(bins, azimuth_sign=args.azimuth_sign, elevation_sign=args.elevation_sign)
    output = rooted(args.output_dir)
    if output.is_dir() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output directory is not empty: {output}; use a new --output-dir or --overwrite')
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    csv_rows = []
    config_dict = {**vars(args), 'resolved_ann_file': str(ann_file), 'algorithm': config.__dict__}
    print(f'Scene {args.scene}: {len(selected)} selected frames; no GT/model/CUDA', flush=True)
    for index, info in enumerate(selected):
        load_start = time.perf_counter()
        radar_path = rooted(Path(info.get('sparse_radar_path') or info['radar_path']))
        coords, power = read_p1(radar_path)
        xyz = geometry.to_xyz(coords)
        load_seconds = time.perf_counter() - load_start
        start = time.perf_counter()
        result = extract_regions(xyz, power, config)
        region_seconds = time.perf_counter() - start
        rgb_path, mapping = None, None
        if not args.no_rgb:
            normalized = dict(info)
            if info.get('cams'):
                normalized['cams'] = {k: {**v, 'data_path': str(rooted(Path(v['data_path'])))} for k, v in info['cams'].items()}
            rgb_path, mapping = alignment.resolve_rgb(normalized, reference, camera_dir=camera_dir,
                camera_offset=args.camera_offset, rgb_image=rooted(args.rgb_image) if args.rgb_image else None,
                camera_name=args.camera_name)
        order = int(info['order_in_scene'])
        stem = f'scene{args.scene}_order{order:05d}'
        image_path = output / f'{stem}.png'
        render(result, xyz, power, image_path, f'Scene {args.scene} | order {order} | P1 175 x 250 returns',
               rgb_path, projection, args.rgb_crop_width, args.max_points, args.max_draw_boxes,
               args.detail_x_limits, args.detail_y_limits, args.power_scale)
        np.savez_compressed(output / f'{stem}.npz', **npz_payload(result, xyz, power, coords))
        regions = [{k: v for k, v in r.items() if k not in ('support_point_ids', 'box_point_ids')} for r in result['regions']]
        fine_regions = [{k: v for k, v in r.items() if k not in ('support_point_ids', 'box_point_ids')} for r in result['fine_regions']]
        record = dict(scene=str(args.scene), order=order, radar_frame_id=info.get('radar_frame_id'),
            radar_path=str(radar_path), image=image_path.name, arrays=f'{stem}.npz', rgb=mapping,
            load_seconds=load_seconds, region_seconds=region_seconds,
            stats=result['stats'], thresholds=result['thresholds'], regions=regions, fine_regions=fine_regions)
        (output / f'{stem}.json').write_text(json.dumps(record, indent=2, default=json_default, allow_nan=False) + '\n', encoding='utf-8')
        frames.append({k: v for k, v in record.items() if k not in ('regions', 'fine_regions', 'thresholds')})
        csv_rows.append(dict(scene=str(args.scene), order=order, **result['stats'],
            load_seconds=load_seconds, region_seconds=region_seconds, image=image_path.name))
        print(f'[{index+1}/{len(selected)}] order={order} seeds={result["stats"]["n_fine_regions"]} '
              f'groups={result["stats"]["n_regions"]} recovered_support={result["stats"]["n_recovered_support"]} '
              f'accepted={result["stats"]["n_accepted"]} boxed_points={result["stats"]["n_box_points"]} '
              f'region_seconds={region_seconds:.3f} image={image_path}', flush=True)
    with (output / 'frame_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest = dict(model_run=False, gt_used_for_proposals=False, source_data_modified=False,
        proposal_ids='frame-local, not tracks; no temporal association yet',
        point_ids='Original NPZ first 175*250 indices, not GPU power-sorted positions',
        box_fraction='Fraction of all P1 returns in accepted boxes; NOT object recall',
        schema_version=2,
        note='Geometry-based reflection groups, not verified objects. Merged boxes enclose support; '
             'size limits and long-chain flags are heuristics. No temporal size stabilization yet. '
             '--no-merge retains original per-seed fixed boxes.',
        config=json.loads(json.dumps(config_dict, default=str)), bins_sha256=hashlib.sha256(bins.read_bytes()).hexdigest(),
        frames=frames)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=json_default, allow_nan=False) + '\n', encoding='utf-8')
    cards = ''.join(f'<figure><figcaption>Scene {html.escape(str(r["scene"]))}, order {r["order"]}</figcaption>'
        f'<a href="{html.escape(r["image"], quote=True)}"><img loading="lazy" src="{html.escape(r["image"], quote=True)}"></a></figure>' for r in frames)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P1 reflection proposals</title>'
        '<style>body{font:16px sans-serif;background:#eee;margin:20px}main{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:16px}figure{margin:0}img{width:100%}</style>'
        '<h1>P1 reflection proposals</h1><p>GT-free, single-frame proposals. RGB is reference only. '
        'Proposal IDs are not tracks. Open an image for details.</p><main>' + cards + '</main>', encoding='utf-8')
    if args.gif:
        from PIL import Image
        images = []
        for frame in frames:
            with Image.open(output / frame['image']) as image:
                image.thumbnail((1280, 900))
                images.append(image.convert('RGB').quantize(colors=128))
        images[0].save(output / 'regions.gif', save_all=True, append_images=images[1:],
                       duration=max(1, round(1000 / args.fps)), loop=0)
    print(f'Gallery: {output / "index.html"}\nSummary: {output / "frame_summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
