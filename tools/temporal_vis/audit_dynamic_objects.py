#!/usr/bin/env python3
"""BBox-proxy same-object KNN recall and paired occupancy trail diagnostics.

No torch/CUDA dependency. See README_dynamic_objects.md for conventions.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as va


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def inside(points, box):
    """Processed LiDAR box: xyz bottom centre, full dimensions, yaw radians."""
    x, y, z, dx, dy, dz, yaw = box
    rel = points - [x, y, z]
    c, s = np.cos(yaw), np.sin(yaw)
    return ((np.abs(c * rel[:, 0] + s * rel[:, 1]) <= dx / 2 + 1e-6)
            & (np.abs(-s * rel[:, 0] + c * rel[:, 1]) <= dy / 2 + 1e-6)
            & (rel[:, 2] >= -1e-6) & (rel[:, 2] <= dz + 1e-6))


def memberships(points, boxes):
    # Overlapping boxes are ambiguous: exclude from query and same-object truth.
    labels = np.full(len(points), -1, dtype=int)
    count = np.zeros(len(points), dtype=int)
    for i, box in enumerate(boxes):
        hit = inside(points, box)
        labels[hit] = i
        count += hit
    labels[count > 1] = -2
    return labels


def ratio(n, d):
    return float(n / d) if d else None


def speed_band(speed):
    if speed is None:
        return 'unknown'
    return '0_0.5' if speed < .5 else '0.5_2' if speed < 2 else '2_5' if speed < 5 else '5_inf'


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}; check selection and data coverage')
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class Dataset:
    def __init__(self, args):
        self.args = args
        self.frames = va.scene_infos(va.read_infos(args.ann_file), args.scene)
        self.by_order = {int(f['order_in_scene']): f for f in self.frames}
        self.orders = sorted(self.by_order)
        self.positions = {order: i for i, order in enumerate(self.orders)}
        if len(self.by_order) != len(self.frames):
            raise ValueError('Duplicate scene/order metadata')
        self.ext = (np.load(args.extrinsic, allow_pickle=False) if args.extrinsic
                    else va.geometry_module.radar_to_lidar_default())
        va.geometry_module.validate_transform(self.ext, 'radar_to_lidar')
        self.geometry = va.RadarBinGeometry.from_file(args.bins)

    @lru_cache(maxsize=32)
    def annotation(self, order):
        f = self.by_order[order]
        p = self.args.bbox_root / str(self.args.scene) / 'bbox'
        i = int(f['occ_frame_idx'])
        boxes = np.load(p / f'bbox{i}.npy', allow_pickle=False).reshape(-1, 7)
        ids = np.load(p / f'bbox_token{i}.npy', allow_pickle=False).reshape(-1).astype(str)
        cats = np.load(p / f'object_category{i}.npy', allow_pickle=False).reshape(-1)
        if not (len(boxes) == len(ids) == len(cats)) or len(set(ids)) != len(ids):
            raise ValueError(f'Inconsistent or duplicate object IDs: order {order}')
        if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
            raise ValueError(f'Invalid box geometry: order {order}')
        return boxes, ids, cats

    def pose(self, order):
        f = self.by_order[order]
        if not f.get('ego_pose_valid', False):
            raise ValueError(f'Invalid ego pose: {order}')
        return va.geometry_module.validate_transform(np.asarray(f['ego_pose']), 'ego_pose')

    def time(self, order):
        f = self.by_order[order]
        if not f.get('timestamp_valid', False) or not np.isfinite(f['timestamp_sec']):
            raise ValueError(f'Real timestamp required: {order}')
        return float(f['timestamp_sec'])

    @lru_cache(maxsize=16)
    def p1(self, order):
        f = self.by_order[order]
        coords, _ = va.load_p1(f.get('sparse_radar_path') or f['radar_path'])
        xyz = self.geometry.to_xyz(coords)
        eligible = np.isfinite(xyz).all(1)
        eligible &= np.hypot(xyz[:, 0], xyz[:, 1]) > self.args.ego_mask_radius
        xyz = xyz[eligible]
        boxes, _, _ = self.annotation(order)
        labels = memberships(transform(xyz, self.ext), boxes)
        return xyz, labels

    def speed(self, order, token):
        times, centers = [], []
        for j in range(order - self.args.speed_window, order + self.args.speed_window + 1):
            if j not in self.by_order:
                continue
            boxes, ids, _ = self.annotation(j)
            idx = np.flatnonzero(ids == token)
            if not len(idx):
                continue
            center = boxes[idx[0], :3].copy()
            center[2] += boxes[idx[0], 5] / 2
            centers.append(transform(center[None], self.pose(j))[0])
            times.append(self.time(j))
        if len(times) < 3 or np.ptp(times) <= 0:
            return None, None
        t = np.asarray(times) - self.time(order)
        design = np.column_stack([t, np.ones(len(t))])
        fit = np.linalg.lstsq(design, np.asarray(centers), rcond=None)[0]
        rmse = float(np.sqrt(np.mean(np.sum((design @ fit - centers) ** 2, axis=1))))
        return (float(np.linalg.norm(fit[0, :2])) if rmse <= self.args.max_speed_rmse else None), rmse

    def selected(self):
        return [j for j in sorted(self.by_order) if self.args.start_order <= j <= self.args.end_order][::self.args.stride][:self.args.max_frames]

    def histories(self, order):
        # Default mirrors TemporalKradarDataset: previous entries in this split.
        if self.args.history_mode == 'dataset':
            pos = self.positions[order]
            return [(order - self.orders[pos-slot], self.orders[pos-slot])
                    for slot in self.args.gaps if pos >= slot]
        return [(gap, order-gap) for gap in self.args.gaps if order-gap in self.by_order]


def matching_counts(query, history, history_labels, object_index, radius, k):
    same = history_labels == object_index
    result = dict(n_queries=len(query), n_history_same=int(same.sum()),
                  n_eligible=0, n_no_history=0, n_outside_radius=0,
                  n_topk_miss=0, n_success=0, n_selected=0, n_selected_same=0)
    if not same.any():
        result['n_no_history'] = len(query)
        return result
    result['n_eligible'] = len(query)
    if not len(query):
        return result
    nearest = cKDTree(history[same]).query(query, k=1)[0]
    d, indices = cKDTree(history).query(query, k=np.arange(1, k + 1), distance_upper_bound=radius)
    valid = np.isfinite(d) & (indices < len(history))
    picked_same = valid & (history_labels[np.minimum(indices, len(history) - 1)] == object_index)
    success = picked_same.any(axis=1)
    outside = nearest >= radius  # scipy query uses an exclusive upper bound
    result.update(n_outside_radius=int(outside.sum()),
                  n_topk_miss=int((~outside & ~success).sum()), n_success=int(success.sum()),
                  n_selected=int(valid.sum()), n_selected_same=int(picked_same.sum()))
    assert result['n_success'] + result['n_topk_miss'] + result['n_outside_radius'] == len(query)
    return result


def match(args, data):
    rows = []
    for order in data.selected():
        current, labels = data.p1(order)
        boxes, ids, cats = data.annotation(order)
        for gap, hist in data.histories(order):
            dt = data.time(order) - data.time(hist)
            if dt <= 0:
                raise ValueError('Nonpositive history time interval')
            history, hist_labels = data.p1(hist)
            _, hist_ids, _ = data.annotation(hist)
            matrix = np.linalg.inv(data.pose(order) @ data.ext) @ (data.pose(hist) @ data.ext)
            history = transform(history, matrix)
            for i, token in enumerate(ids):
                speed, rmse = data.speed(order, token)
                same_idx = np.flatnonzero(hist_ids == token)
                # -999 cannot collide with background (-1) or ambiguous (-2).
                counts = matching_counts(current[labels == i], history, hist_labels,
                                         int(same_idx[0]) if len(same_idx) else -999,
                                         args.radius, args.k)
                rows.append(dict(scene=args.scene, order=order, occ_frame_idx=data.by_order[order]['occ_frame_idx'],
                                 object_id=token, category=int(cats[i]), history_order=hist, frame_gap=gap,
                                 delta_t_sec=dt, speed_mps=speed, speed_fit_rmse=rmse, speed_band=speed_band(speed),
                                 history_box_present=bool(len(same_idx)), radius_m=args.radius, k=args.k,
                                 **counts, same_object_recall=ratio(counts['n_success'], counts['n_eligible']),
                                 selected_same_precision=ratio(counts['n_selected_same'], counts['n_selected'])))
        print(f'matching order={order}: {len(ids)} objects', flush=True)
    write_csv(args.output_dir / 'matching_per_object_history.csv', rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['speed_band'], row['frame_gap'])].append(row)
    summary = []
    for (band, gap), group in sorted(groups.items()):
        totals = {key: sum(r[key] for r in group) for key in rows[0] if key.startswith('n_')}
        summary.append(dict(speed_band=band, frame_gap=gap, object_history_pairs=len(group),
                            delta_t_min=min(r['delta_t_sec'] for r in group), delta_t_max=max(r['delta_t_sec'] for r in group),
                            **totals, same_object_recall=ratio(totals['n_success'], totals['n_eligible']),
                            selected_same_precision=ratio(totals['n_selected_same'], totals['n_selected'])))
    write_csv(args.output_dir / 'matching_by_speed_gap.csv', summary)


def dense(path, shape, sparse_order='zyx'):
    a = np.load(path, allow_pickle=False)
    if a.shape == tuple(shape):
        return a.astype(np.int64)
    if a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f'{path}: expected dense XYZ grid or sparse integer [z,y,x,class]')
    xyz = a[:, [2, 1, 0] if sparse_order == 'zyx' else [0, 1, 2]].astype(int)
    if np.any(xyz < 0) or np.any(xyz >= shape):
        raise ValueError(f'{path}: out-of-grid indices; check grid shape and coordinate convention')
    out = np.zeros(shape, dtype=np.int64)
    out[tuple(xyz.T)] = a[:, 3].astype(int)
    return out


def pred_path(root, info):
    return root / str(info['scene_id']) / str(info.get('lidar_token', info['token'])) / 'pred_c.npy'


def plot_overlay(path, points, gt, single, knn, trail, current):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    roi = trail | current
    if not roi.any():
        plt.close(fig)
        return
    lo, hi = points[roi, :2].min(0) - 2, points[roi, :2].max(0) + 2
    local = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
    for ax, occupied, title in zip(axes, [gt, single, knn], ['GT', 'Single frame', 'Temporal KNN']):
        for mask, color, label in [(trail, '#eab676', 'Trail evaluation region'),
                                   (current, '#84bbdf', 'Current box'),
                                   (occupied & roi, 'black', 'Occupied in target/trail'),
                                   (occupied & trail, '#d62728', 'Occupied in trail')]:
            xy = np.unique(points[mask & local, :2], axis=0)
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c=color, label=label)
        ax.set(title=f'{title} | trail occupied={int((occupied & trail).sum())}/{int(trail.sum())}',
               xlabel='Grid X (m)', ylabel='Grid Y (m)', aspect='equal')
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def occupancy(args, data):
    matching = {}
    if args.matching_csv:
        with args.matching_csv.open() as stream:
            for row in csv.DictReader(stream):
                key = (int(row['scene']), int(row['order']), row['object_id'], int(row['frame_gap']))
                if key in matching:
                    raise ValueError('Duplicate matching CSV join key')
                matching[key] = row
    shape = np.asarray(args.grid_shape)
    bounds = np.asarray(args.pc_range)
    axes = [bounds[i] + (np.arange(shape[i]) + .5) * (bounds[i + 3] - bounds[i]) / shape[i] for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    lidar_points = points if args.grid_frame == 'lidar' else transform(points, data.ext)
    radar_points = transform(lidar_points, np.linalg.inv(data.ext))
    fov = (np.abs(np.degrees(np.arctan2(radar_points[:, 1], radar_points[:, 0]))) <= args.fov_deg / 2) & (radar_points[:, 0] >= 0)
    rows, skipped, plots = [], [], 0
    for order in data.selected():
        info = data.by_order[order]
        paths = [pred_path(args.single_root, info), pred_path(args.knn_root, info)]
        if not all(p.exists() for p in paths):
            if not args.skip_missing:
                raise FileNotFoundError(f'Missing paired predictions: {paths}')
            skipped.append(dict(order=order, reason='missing_prediction', paths=[str(p) for p in paths])); continue
        gt = dense(info['occ_path'], shape, sparse_order=args.gt_order).ravel()
        single, knn = [dense(p, shape).ravel() for p in paths]
        valid = fov & (gt != 255)
        if args.valid_mask_template:
            mask_path = args.valid_mask_template.format(scene=args.scene, order=order, occ_frame_idx=info['occ_frame_idx'], token=info['token'])
            mask = np.load(mask_path, allow_pickle=False)
            if mask.shape != tuple(shape) or not np.isin(mask, [0, 1]).all():
                raise ValueError('Valid mask must be binary dense XYZ with matching grid shape')
            valid &= mask.ravel().astype(bool)
        gt_occ = (gt != 0) & (gt != 255)
        s_occ = (single != 0) & (single != 255)
        k_occ = (knn != 0) & (knn != 255)
        boxes, ids, _ = data.annotation(order)
        for i, token in enumerate(ids):
            current = inside(lidar_points, boxes[i])
            target = current & gt_occ & valid
            speed, rmse = data.speed(order, token)
            union = np.zeros(len(points), dtype=bool)
            hist_records = []
            for gap, hist in data.histories(order):
                hboxes, hids, _ = data.annotation(hist)
                idx = np.flatnonzero(hids == token)
                if not len(idx):
                    continue
                # Test current grid centres in historical LiDAR frame: handles rotation.
                historical_points = transform(lidar_points, np.linalg.inv(data.pose(hist)) @ data.pose(order))
                past = inside(historical_points, hboxes[idx[0]])
                union |= past
                hist_records.append((gap, hist, past))
            if not hist_records:
                continue
            for gap, hist, past in hist_records + [(0, None, union)]:
                trail = past & ~current & valid & (gt == 0)
                n = int(trail.sum()); sf = int((s_occ & trail).sum()); kf = int((k_occ & trail).sum())
                keys = [(int(args.scene), order, token, g) for g, _, _ in hist_records] if gap == 0 else [(int(args.scene), order, token, gap)]
                joined = [matching[key] for key in keys if key in matching]
                eligible = sum(int(r['n_eligible']) for r in joined)
                successes = sum(int(r['n_success']) for r in joined)
                row = dict(scene=args.scene, order=order, object_id=token, frame_gap=gap,
                           region='history_union' if gap == 0 else 'history_box', history_order=hist,
                           delta_t_sec=data.time(order)-data.time(hist) if hist is not None else None,
                           speed_mps=speed, speed_fit_rmse=rmse, speed_band=speed_band(speed),
                           matching_pairs_expected=len(keys), matching_pairs_joined=len(joined),
                           n_eligible=eligible, n_success=successes,
                           same_object_recall=ratio(successes, eligible) if len(joined)==len(keys) else None,
                           n_trail_free=n, single_trail_fp=sf, knn_trail_fp=kf,
                           single_tfpr=ratio(sf,n), knn_tfpr=ratio(kf,n), delta_tfpr=ratio(kf-sf,n),
                           n_target_occupied=int(target.sum()), single_target_tp=int((target & s_occ).sum()),
                           knn_target_tp=int((target & k_occ).sum()))
                row['single_target_recall'] = ratio(row['single_target_tp'], row['n_target_occupied'])
                row['knn_target_recall'] = ratio(row['knn_target_tp'], row['n_target_occupied'])
                rows.append(row)
                if gap == 0 and plots < args.max_plots:
                    plot_overlay(args.output_dir / f'order{order}_object{ i}_occupancy.png', points, gt_occ & valid, s_occ & valid, k_occ & valid, trail, current)
                    plots += 1
        print(f'occupancy order={order}', flush=True)
    (args.output_dir / 'skipped_frames.json').write_text(json.dumps(skipped, indent=2))
    write_csv(args.output_dir / 'occupancy_per_object.csv', rows)
    summary = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['region'], row['speed_band'], row['frame_gap'])].append(row)
    for (region, band, gap), group in sorted(groups.items()):
        totals = {key: sum(r[key] for r in group) for key in ['n_trail_free','single_trail_fp','knn_trail_fp','n_target_occupied','single_target_tp','knn_target_tp']}
        paired = [r for r in group if r['same_object_recall'] is not None and r['delta_tfpr'] is not None]
        correlation = None
        if len(paired) >= 3:
            x = [1-r['same_object_recall'] for r in paired]; y = [r['delta_tfpr'] for r in paired]
            if np.std(x) > 0 and np.std(y) > 0:
                correlation = float(np.corrcoef(x,y)[0,1])
        summary.append(dict(region=region, speed_band=band, frame_gap=gap, object_pairs=len(group), **totals,
                            single_tfpr=ratio(totals['single_trail_fp'],totals['n_trail_free']),
                            knn_tfpr=ratio(totals['knn_trail_fp'],totals['n_trail_free']),
                            delta_tfpr=ratio(totals['knn_trail_fp']-totals['single_trail_fp'],totals['n_trail_free']),
                            single_target_recall=ratio(totals['single_target_tp'],totals['n_target_occupied']),
                            knn_target_recall=ratio(totals['knn_target_tp'],totals['n_target_occupied']),
                            correlation_pairs=len(paired), miss_vs_delta_tfpr_pearson=correlation))
    write_csv(args.output_dir / 'occupancy_summary.csv', summary)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['match','occupancy'])
    p.add_argument('--ann-file', type=Path, default=Path('data/annotations/kradar_dict_test_official_temporal_doppler8.pkl'))
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--extrinsic', type=Path)
    p.add_argument('--scene', default='3')
    p.add_argument('--start-order', type=int, default=80)
    p.add_argument('--end-order', type=int, default=200)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--max-frames', type=int, default=25)
    p.add_argument('--gaps', type=int, nargs='+', default=[1,2,3])
    p.add_argument('--history-mode', choices=['dataset', 'scene-order'], default='dataset',
                   help='gaps are preceding split slots (model default) or exact scene-order gaps')
    p.add_argument('--radius', type=float, default=2.)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--ego-mask-radius', type=float, default=0.)
    p.add_argument('--speed-window', type=int, default=2, help='Half window in scene orders; fits world XY speed')
    p.add_argument('--max-speed-rmse', type=float, default=.5)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--single-root', type=Path)
    p.add_argument('--knn-root', type=Path)
    p.add_argument('--matching-csv', type=Path)
    p.add_argument('--grid-shape', type=int, nargs=3, default=[128,128,14])
    p.add_argument('--pc-range', type=float, nargs=6, default=[0,-25.6,-2.6,51.2,25.6,3.])
    p.add_argument('--grid-frame', choices=['lidar','radar'], default='lidar')
    p.add_argument('--gt-order', choices=['xyz','zyx'], default='xyz')
    p.add_argument('--fov-deg', type=float, default=107.)
    p.add_argument('--valid-mask-template', help='Binary XYZ .npy path with {scene}/{order}/{occ_frame_idx}/{token} placeholders')
    p.add_argument('--skip-missing', action='store_true')
    p.add_argument('--max-plots', type=int, default=6)
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.radius <= 0 or args.k < 1 or args.stride < 1 or args.max_frames < 1 or min(args.gaps) < 1 or len(set(args.gaps)) != len(args.gaps):
        p.error('Positive radius/k/stride/max-frames and unique positive gaps required')
    if args.end_order < args.start_order or args.speed_window < 1 or args.max_speed_rmse < 0 or args.ego_mask_radius < 0 or min(args.grid_shape) <= 0 or not 0 < args.fov_deg <= 360 or args.max_plots < 0:
        p.error('Invalid selection, geometry or speed-fit settings')
    if np.any(np.asarray(args.pc_range[3:]) <= args.pc_range[:3]):
        p.error('pc-range upper bounds must exceed lower bounds')
    if args.experiment == 'occupancy' and (args.single_root is None or args.knn_root is None):
        p.error('occupancy requires --single-root and --knn-root')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = Dataset(args)
    if not data.selected():
        p.error('No selected frames')
    (match if args.experiment == 'match' else occupancy)(args, data)
    metadata = {key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    metadata.update(box_truth='Processed expanded bbox proxy; overlapping P1 membership excluded',
                    free_space='valid-mask restricted benchmark empty' if args.valid_mask_template else 'benchmark empty within geometric FoV; not observed-free ground truth',
                    speed='local least-squares world XY geometric-centre speed; unknown if <3 observations or high RMSE',
                    history='dataset slots or exact order gaps per history_mode; frame_gap always actual order difference',
                    interpretation='association only; shared voxels/objects/frames are correlated samples')
    (args.output_dir / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
