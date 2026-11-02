#!/usr/bin/env python3
"""GT-only evaluation of GT-free P1 reflection region matching (no learned KNN).

The matcher sees region geometry and ego poses only. GT bbox tokens are read
AFTER matching to score output; they never influence proposals or matching.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import vis_alignment as va
from tools.temporal_vis.audit_dynamic_objects import memberships, transform
from tools.temporal_vis.p1_reflection_regions import RegionConfig, extract_regions

WEIGHTS = ((1., 0.), (.9, .1), (.75, .25), (.5, .5), (.25, .75), (0., 1.))


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def label_regions(regions, point_labels, object_ids, purity=0.6, min_gt_points=3):
    """Assign an unambiguous GT ID to a proposal using its SUPPORT points.

    Denominator includes unlabeled/ambiguous support points, avoiding inflated
    purity. Duplicate GT object IDs are always ambiguous.
    """
    object_ids = np.asarray(object_ids).astype(str)
    _, inverse, counts = np.unique(object_ids, return_inverse=True, return_counts=True)
    is_unique_id = counts[inverse] == 1
    result = {}
    for region in regions:
        if not region['accepted']:
            continue
        indices = np.asarray(region['support_point_ids'], dtype=np.int64)
        if not len(indices):
            continue
        labels = np.asarray(point_labels)[indices]
        labels = labels[(labels >= 0) & (labels < len(object_ids))]
        labels = labels[is_unique_id[labels]]
        if not len(labels):
            continue
        values, frequencies = np.unique(labels, return_counts=True)
        winner = int(values[np.argmax(frequencies)])
        n = int(frequencies.max())
        if n >= min_gt_points and n / len(indices) >= purity:
            result[int(region['id'])] = str(object_ids[winner])
    return result


def aligned_box(center, dimensions, matrix):
    """History radar AABB -> conservative current radar AABB under rotation."""
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    rot = np.asarray(matrix[:3, :3], dtype=np.float64)
    result_center = rot @ center + matrix[:3, 3]
    result_extent = np.abs(rot) @ dimensions
    return result_center, result_extent


def select_match(current, histories, slot, w_center, w_size, y_gates, center_gates,
                 size_scale=0.5):
    """Geometry-only nearest cost after hard Y/3D gates; no GT or tracking ID."""
    if slot not in (1, 2, 3):
        raise ValueError('History slot must be 1, 2 or 3')
    if not histories:
        return None
    centers = np.asarray([h['aligned_center'] for h in histories], dtype=np.float64)
    sizes = np.asarray([h['aligned_size'] for h in histories], dtype=np.float64)
    current_center = np.asarray(current['center'], dtype=np.float64)
    current_size = np.asarray(current['box_size'], dtype=np.float64)
    y_diff = np.abs(centers[:, 1] - current_center[1])
    distances = np.linalg.norm(centers - current_center, axis=1)
    allowed = (y_diff <= y_gates[slot-1] + 1e-10) & (distances <= center_gates[slot-1] + 1e-10)
    if not allowed.any():
        return None
    size_error = np.mean(np.abs(np.log(np.maximum(sizes, 1e-6) /
                                         np.maximum(current_size[None], 1e-6))), axis=1)
    cost = w_center * (distances / center_gates[slot-1]) + w_size * (size_error / size_scale)
    eligible = np.flatnonzero(allowed)
    # Tie breaks: center distance, then historical region ID.
    idx = int(eligible[np.lexsort((np.array([histories[i]['id'] for i in eligible]),
                                   distances[eligible], cost[eligible]))[0]])
    return histories[idx]


def summarize(rows):
    """Recall denominator: current GT-labeled regions with same-ID history region."""
    c = Counter(dict(current_labeled=0, has_true_history=0, y_missed_true=0,
                     center_missed_true=0, predicted=0, unscorable_prediction=0,
                     scorable_prediction=0, correct=0, wrong=0))
    for row in rows:
        c['current_labeled'] += 1
        c['has_true_history'] += int(row['has_true_history'])
        c['y_missed_true'] += int(row['y_missed_true'])
        c['center_missed_true'] += int(row['center_missed_true'])
        if row['predicted']:
            c['predicted'] += 1
            if row['predicted_id'] is None:
                c['unscorable_prediction'] += 1
            else:
                c['scorable_prediction'] += 1
                c['correct'] += int(row['correct'])
                c['wrong'] += int(not row['correct'])
    return dict(**c, precision=safe_ratio(c['correct'], c['scorable_prediction']),
                recall=safe_ratio(c['correct'], c['has_true_history']),
                false_match_rate=safe_ratio(c['wrong'], c['scorable_prediction']),
                prediction_rate=safe_ratio(c['predicted'], c['current_labeled']),
                gt_y_rejection_rate=safe_ratio(c['y_missed_true'], c['has_true_history']),
                gt_center_rejection_rate=safe_ratio(c['center_missed_true'], c['has_true_history']))


class FrameData:
    def __init__(self, args):
        self.args = args
        self.geometry = va.RadarBinGeometry.from_file(args.bins)
        self.extrinsic = (np.load(args.extrinsic, allow_pickle=False) if args.extrinsic
                          else va.geometry_module.radar_to_lidar_default())
        va.geometry_module.validate_transform(self.extrinsic, 'radar_to_lidar')
        self.config = RegionConfig(seed_percentile=args.seed_percentile, seed_mad=args.seed_mad,
                                   box_size=tuple(args.box_size_m),
                                   max_box_size=tuple(args.max_box_size_m)).validate()
        infos = va.read_infos(args.ann_file)
        self.scenes = {}
        for info in infos:
            scene = str(info.get('scene_id', info.get('scene_token')))
            self.scenes.setdefault(scene, []).append(info)
        for scene, frames in self.scenes.items():
            frames.sort(key=lambda x: int(x['order_in_scene']))
            if len(set(int(f['order_in_scene']) for f in frames)) != len(frames):
                raise ValueError(f'Duplicate order_in_scene in scene {scene}')

    def radar_path(self, info):
        path = Path(info.get('sparse_radar_path') or info['radar_path'])
        return path if path.is_absolute() else ROOT / path

    @lru_cache(maxsize=12)
    def frame(self, scene, order):
        info = self.info_index[(scene, order)]
        if not info.get('ego_pose_valid', False):
            raise ValueError(f'Missing valid ego pose for scene={scene}, order={order}')
        pose = va.geometry_module.validate_transform(np.asarray(info['ego_pose']), 'ego_pose')
        path = self.radar_path(info)
        with np.load(path, allow_pickle=False) as src:
            count = 175 * 250
            power = np.asarray(src['power_val'][2, :count], dtype=np.float64)
            az = np.asarray(src['azimuth_ind'][:count], dtype=int)
            el = np.asarray(src['elevation_ind'][:count], dtype=int)
        if len(power) != count or len(az) != count or len(el) != count:
            raise ValueError(f'Invalid P1 data: {path}')
        coords = np.stack([np.repeat(np.arange(175), 250), az, el], axis=-1)
        xyz = self.geometry.to_xyz(coords)
        proposals = extract_regions(xyz, power, self.config)['regions']
        bbox_dir = self.args.bbox_root / scene / 'bbox'
        bbox_index = int(info['occ_frame_idx'])
        gt_boxes = np.load(bbox_dir / f'bbox{bbox_index}.npy', allow_pickle=False).reshape(-1, 7)
        gt_ids = np.load(bbox_dir / f'bbox_token{bbox_index}.npy', allow_pickle=False).reshape(-1).astype(str)
        if len(gt_boxes) != len(gt_ids):
            raise ValueError(f'GT bbox/token count mismatch: {scene}/{order}')
        point_labels = memberships(transform(xyz, self.extrinsic), gt_boxes)
        labels = label_regions(proposals, point_labels, gt_ids, self.args.gt_purity,
                               self.args.min_gt_points)
        accepted = [r for r in proposals if r['accepted']]
        return dict(accepted=accepted, labels=labels, pose=pose, count=len(accepted),
                    assigned=len(labels), order=order)

    def index(self):
        self.info_index = {(scene, int(info['order_in_scene'])): info
                           for scene, frames in self.scenes.items() for info in frames}


def history_regions(current, previous, extrinsic):
    # Radar -> world = (LiDAR -> world) @ (radar -> LiDAR).
    # History radar -> current radar:
    relative = np.linalg.inv(current['pose'] @ extrinsic) @ (previous['pose'] @ extrinsic)
    result = []
    for box in previous['accepted']:
        center, size = aligned_box(box['center'], box['box_size'], relative)
        result.append(dict(id=int(box['id']), aligned_center=center, aligned_size=size))
    return result


def evaluate(args):
    if not (0 < args.gt_purity <= 1 and args.min_gt_points >= 1 and args.size_scale > 0):
        raise ValueError('Invalid GT purity, min_gt_points or size scale')
    if any(x <= 0 for x in (*args.y_gates, *args.center_gates)):
        raise ValueError('All matching gates must be positive')
    data = FrameData(args)
    data.index()
    scenes = args.scenes or sorted(data.scenes)
    for scene in scenes:
        if scene not in data.scenes:
            raise ValueError(f'Scene {scene} not found; available: {sorted(data.scenes)}')
    all_rows = defaultdict(list)
    detailed = []
    start = time.perf_counter()
    frame_count = 0
    for scene in scenes:
        frames = data.scenes[scene]
        chosen = [i for i, info in enumerate(frames)
                  if args.start_order <= int(info['order_in_scene']) <= args.end_order]
        chosen = chosen[::args.stride][:args.max_frames_per_scene or None]
        for pos in chosen:
            info = frames[pos]
            order = int(info['order_in_scene'])
            current = data.frame(scene, order)
            frame_count += 1
            for slot in (1, 2, 3):
                if pos < slot:
                    continue
                history_order = int(frames[pos-slot]['order_in_scene'])
                previous = data.frame(scene, history_order)
                transformed = history_regions(current, previous, data.extrinsic)
                hist_labels = previous['labels']
                for r in current['accepted']:
                    actual = current['labels'].get(int(r['id']))
                    if actual is None:
                        continue
                    truth = [h for h in transformed if hist_labels.get(h['id']) == actual]
                    y_ok = [h for h in truth if abs(h['aligned_center'][1] - r['center'][1]) <=
                            args.y_gates[slot-1] + 1e-10]
                    center_ok = [h for h in y_ok if np.linalg.norm(h['aligned_center'] - r['center']) <=
                                 args.center_gates[slot-1] + 1e-10]
                    for wd, ws in WEIGHTS:
                        matched = select_match(r, transformed, slot, wd, ws,
                                               args.y_gates, args.center_gates, args.size_scale)
                        prediction = hist_labels.get(matched['id']) if matched else None
                        row = dict(scene=scene, order=order, history_order=history_order, slot=slot,
                                   current_region=int(r['id']), current_id=actual,
                                   matched_region=matched['id'] if matched else None,
                                   predicted_id=prediction, predicted=matched is not None,
                                   correct=prediction == actual and matched is not None,
                                   has_true_history=bool(truth),
                                   y_missed_true=bool(truth) and not bool(y_ok),
                                   center_missed_true=bool(y_ok) and not bool(center_ok))
                        all_rows[(wd, ws, slot)].append(row)
                        detailed.append(dict(weight_center=wd, weight_size=ws, **row))
            if frame_count % 10 == 0:
                print(f'Processed {frame_count} frames; scene={scene} order={order}', flush=True)
    result = []
    for wd, ws in WEIGHTS:
        for slot in (1, 2, 3, 0):
            rows = (sum((all_rows[(wd, ws, s)] for s in (1, 2, 3)), []) if slot == 0
                    else all_rows[(wd, ws, slot)])
            result.append(dict(weight_center=wd, weight_size=ws,
                               slot='all' if slot == 0 else slot, **summarize(rows)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    with (args.output_dir/'matches.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=(list(detailed[0]) if detailed else
                                ['weight_center', 'weight_size', 'scene', 'order', 'slot']))
        writer.writeheader()
        writer.writerows(detailed)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    (args.output_dir/'report.json').write_text(json.dumps(dict(config=config, metrics=result,
        frames_evaluated=frame_count, seconds=time.perf_counter()-start,
        notes=['GT bbox IDs used for offline scoring ONLY; NEVER in matching.',
               'Only accepted radar regions with unambiguous GT majority are scored.',
               'Precision excludes predictions with unknown history GT label; count them separately.',
               'Recall denominator is current labeled regions with a same-ID labeled history region.',
               'Unmatched current regions outside accepted proposals are not part of this benchmark.']),
        indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print('Processed frames:', frame_count, '| elapsed seconds:', round(time.perf_counter()-start, 1))
    for row in result:
        if row['slot'] == 'all':
            print(f"weights={row['weight_center']:.2f}/{row['weight_size']:.2f}: "
                  f"precision={row['precision']} recall={row['recall']} "
                  f"GT_Y_rejection={row['gt_y_rejection_rate']} "
                  f"unscorable={row.get('unscorable_prediction', 0)}")
    print('Outputs:', args.output_dir.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file', type=Path, required=True, help='Full temporal split annotation PKL')
    p.add_argument('--bbox-root', type=Path, default=Path('data/K-RadarOcc/train'))
    p.add_argument('--bins', type=Path, default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--extrinsic', type=Path, help='Optional radar-to-LiDAR .npy; otherwise default project calibration')
    p.add_argument('--scenes', nargs='+', help='Scene IDs; default all scenes in the selected annotation')
    p.add_argument('--start-order', type=int, default=0)
    p.add_argument('--end-order', type=int, default=10**9)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames-per-scene', type=int, default=0, help='0 = no limit')
    p.add_argument('--seed-percentile', type=float, default=97.)
    p.add_argument('--seed-mad', type=float, default=3.)
    p.add_argument('--box-size-m', type=float, nargs=3, default=(6., 3., 3.))
    p.add_argument('--max-box-size-m', type=float, nargs=3, default=(12., 4., 4.))
    p.add_argument('--y-gates', type=float, nargs=3, default=(.5, 1., 1.5),
                   metavar=('T1', 'T2', 'T3'))
    p.add_argument('--center-gates', type=float, nargs=3, default=(4., 6., 8.),
                   metavar=('T1', 'T2', 'T3'))
    p.add_argument('--size-scale', type=float, default=.5)
    p.add_argument('--gt-purity', type=float, default=.6)
    p.add_argument('--min-gt-points', type=int, default=3)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/region_matching_benchmark'))
    args = p.parse_args()
    if args.stride <= 0 or args.max_frames_per_scene < 0 or args.end_order < args.start_order:
        p.error('Invalid stride, max frames, or order range')
    evaluate(args)


if __name__ == '__main__':
    main()
