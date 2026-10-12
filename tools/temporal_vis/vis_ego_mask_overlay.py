#!/usr/bin/env python3
"""Compare T-frame ego-aligned overlays before/after local self-region masking."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import vis_ego_power_mask as audit
import vis_alignment as alignment


def history_for(rows, current, frame_nums):
    pos = next(i for i, row in enumerate(rows) if row is current)
    positions = list(range(pos-frame_nums+1, pos+1))
    history = [rows[max(0, i)] for i in positions]
    valid = np.array([i >= 0 and bool(row.get('ego_pose_valid', True))
                      for i, row in zip(positions, history)])
    if not valid[-1]:
        raise ValueError('Selected current frame has invalid ego pose')
    order = lambda row: int(row['order_in_scene']) if 'order_in_scene' in row else next(i for i, r in enumerate(rows) if r is row)
    offsets = np.array([order(row)-order(current) for row in history])
    return history, valid, offsets


def selected_ids(scores, xyz, max_points, max_radius):
    # Strongest original returns, plus all mask-region returns so even weak
    # near-origin echoes are visible. Never reselect after power is zeroed.
    strongest = np.argsort(scores, kind='stable')[-max_points:]
    local = np.flatnonzero(np.hypot(xyz[:, 0], xyz[:, 1]) <= max_radius)
    return np.union1d(strongest, local)


def render(points, scores, ids, masks, valid, offsets, radius, path, title, zoom, transforms, rgb=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 18 if rgb else 15), layout='constrained')
    shift = int(rgb is not None)
    grid = fig.add_gridspec(3+shift, 2)
    if rgb:
        alignment.show_rgb(fig.add_subplot(grid[0, :]), rgb)
    axes = [[fig.add_subplot(grid[row+shift, col], projection='3d' if row == 2 else None)
             for col in range(2)] for row in range(3)]
    visible = np.concatenate([points[t, ids[t]] for t in range(len(valid)) if valid[t]])
    lo, hi = visible.min(axis=0)-1, visible.max(axis=0)+1
    colors = plt.get_cmap('tab10')
    for frame in range(len(valid)):
        if not valid[frame]:
            continue
        original = ids[frame][scores[frame, ids[frame]] > 0]
        kept = original[~masks[frame, original]]
        color = colors(frame % 10)
        label = 'current t' if frame == len(valid)-1 else f't{offsets[frame]:+d}'
        for col, use in enumerate((original, kept)):
            p = points[frame, use]
            for row in (0, 1):
                axes[row][col].scatter(p[:, 0], p[:, 1], s=8 if row == 0 else 14,
                                      color=color, alpha=.8, label=label)
            axes[2][col].scatter(p[:, 0], p[:, 1], p[:, 2], s=6, color=color, alpha=.8, label=label)
    for col in range(2):
        heading = 'Original T-frame overlay' if col == 0 else f'Local power zeroed (r={radius:g}m), then aligned'
        for row in (0, 1):
            ax = axes[row][col]
            ax.set(xlabel='Current radar X forward (m)', ylabel='Y left (m)',
                   title=heading + (' | BEV' if row == 0 else ' | near-origin BEV'))
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim((lo[0], hi[0]) if row == 0 else (-zoom, zoom))
            ax.set_ylim((lo[1], hi[1]) if row == 0 else (-zoom, zoom))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            # Current-frame boundary only. History boundaries are transformed
            # into current coordinates and therefore drawn at their own poses.
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.stack((radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)), axis=-1)
            # history origins can be recovered from raw->aligned stored below
            for t, transform in enumerate(transforms):
                if valid[t]:
                    boundary = circle @ transform[:3, :3].T + transform[:3, 3]
                    ax.plot(boundary[:, 0], boundary[:, 1], '--', color=colors(t%10), alpha=.6, linewidth=1)
        ax = axes[2][col]
        ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=heading+' | 3D',
               xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        ax.view_init(elev=25, azim=-65)
    fig.suptitle(title+'\nSame original return IDs and views; colors = frames; dashed boundaries = per-frame mask regions', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=audit.ROOT)
    p.add_argument('--ann-dir', type=Path, default=Path('data/annotations'))
    p.add_argument('--splits', nargs='+', choices=['train','val','test'], default=['train','val','test'])
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--radii-m', type=float, nargs='+', default=[.5,1,2])
    p.add_argument('--frame-nums', type=int, default=4)
    p.add_argument('--max-points', type=int, default=2000)
    p.add_argument('--order', type=int)
    p.add_argument('--zoom-m', type=float, default=10)
    p.add_argument('--camera-root', type=Path, help='Root containing SCENE/images_rb_switched')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/ego_mask_overlay_audit'))
    args = p.parse_args()
    if (args.frame_nums < 1 or args.max_points < 1 or
            not all(np.isfinite(r) and r > 0 for r in args.radii_m) or
            not np.isfinite(args.zoom_m) or args.zoom_m <= max(args.radii_m)):
        p.error('Require positive frame/point counts, finite radii, and zoom greater than radius')
    root = args.repo_root.resolve()
    rooted = lambda path: path if path.is_absolute() else root/path
    out = rooted(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    geometry = alignment.RadarBinGeometry.from_file(rooted(args.bins))
    datasets, references = {}, {}
    for split in ['train','val','test']:
        ann = rooted(args.ann_dir)/f'kradar_dict_{split}_official_temporal_doppler8.pkl'
        if split in args.splits or (args.camera_root and ann.exists()):
            infos = audit.read_infos(ann)
            datasets[split] = infos
            for info in infos:
                scene = str(info.get('scene_id', info.get('scene_token')))
                key = str(info.get('lidar_token', info.get('radar_path', info.get('sparse_radar_path'))))
                references.setdefault(scene, {})[key] = info
    records = []
    for split in args.splits:
        for scene, current, scene_count, method in audit.choose_frames(datasets[split], args.order):
            rows = sorted([i for i in datasets[split] if str(i.get('scene_id', i.get('scene_token'))) == scene],
                          key=lambda i:i.get('order_in_scene', i.get('timestamp',0)))
            history, valid, offsets = history_for(rows, current, args.frame_nums)
            xyz, powers, source = [], [], []
            for info in history:
                source_path = audit.resolve_radar(info, root)
                local, power, indices = audit.read_radar(source_path, geometry)
                crop = indices[:, 0] < 175  # Small P1's 175 range bins.
                xyz.append(local[crop]); powers.append(power[:, crop]); source.append(str(source_path))
            raw = np.stack(xyz)
            scores = np.stack([power[2] for power in powers])
            poses = np.stack([np.asarray(i['ego_pose']) for i in history])
            aligned, transforms = alignment.align_to_current(raw[None], poses[None], valid_mask=valid[None])
            aligned = aligned[0]
            ids = [selected_ids(scores[t], raw[t], args.max_points, max(args.radii_m)) for t in range(args.frame_nums)]
            rgb, rgb_mapping = None, None
            if args.camera_root:
                try:
                    ref = sorted(references[scene].values(), key=lambda i:i.get('order_in_scene', i.get('timestamp',0)))
                    rgb, rgb_mapping = alignment.resolve_rgb(current, ref,
                        camera_dir=rooted(args.camera_root)/scene/'images_rb_switched')
                except (ValueError, FileNotFoundError, IndexError) as exc:
                    rgb_mapping = dict(error=str(exc))
                    print(f'Warning: {split}/scene{scene}: RGB omitted: {exc}')
            directory = out/split
            directory.mkdir(exist_ok=True)
            for radius in args.radii_m:
                masks = np.stack([audit.zero_ego_power(raw[t], powers[t], radius)[1] for t in range(args.frame_nums)])
                destination = directory/f'scene{scene}_T{args.frame_nums}_r{radius:g}m.png'
                render(aligned, scores, ids, masks, valid, offsets, radius, destination,
                       f'{split} | scene {scene} | current order {current.get("order_in_scene")} | T={args.frame_nums}',
                       args.zoom_m, transforms[0], rgb)
                records.append(dict(split=split,scene=scene,radius_m=radius,image=str(destination),
                    current_token=str(current.get('lidar_token')),selection=method,scene_split_frames=scene_count,
                    orders=[i.get('order_in_scene') for i in history],valid=valid.tolist(),offsets=offsets.tolist(),
                    masked_per_frame=masks.sum(axis=1).tolist(),source_npzs=source,rgb=rgb_mapping))
                print(f'Saved {destination}; masked/frame={masks.sum(axis=1).tolist()}')
    (out/'manifest.json').write_text(json.dumps(dict(source_files_modified=False,
        mask='raw-frame hypot(x,y)<=radius; power channels 0,1,2,6,7 zeroed; zero-power returns not plotted',frames=records),indent=2))
    cards = []
    for r in records:
        src = html.escape(str(Path(r['image']).relative_to(out)),quote=True)
        cards.append(f'<figure><figcaption>{html.escape(r["split"])} scene {html.escape(r["scene"])} | {r["radius_m"]:g}m</figcaption><a href="{src}"><img loading="lazy" src="{src}"></a></figure>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Temporal ego mask overlay</title>'
        '<style>main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:15px}img{width:100%}figure{margin:0}</style>'
        '<h1>Original / masked ego-aligned T-frame overlays</h1><main>'+''.join(cards)+'</main>')
    print(f'Gallery: {out/"index.html"}')


if __name__ == '__main__':
    main()
