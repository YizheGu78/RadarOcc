#!/usr/bin/env python3
"""GT-only evaluation of GT-free P1 reflection region matching (no learned KNN).

The matcher sees region geometry and ego poses only. GT bbox tokens are read
AFTER matching to score output; they never influence proposals or matching.
"""
import argparse
import csv
import itertools
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
MATCH_FIELDS = ['weight_center', 'weight_size', 'scene', 'order', 'history_order', 'slot',
    'current_region', 'current_id', 'current_scorable', 'matched_region', 'predicted_id',
    'predicted', 'correct', 'has_true_history', 'y_missed_true', 'center_missed_true',
    'current_x', 'current_y', 'current_z', 'history_accepted', 'history_labeled',
    'candidates_after_y', 'candidates_after_center']


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def gt_memberships(xyz, boxes_lidar, radar_to_lidar, space='xyz'):
    """Assign radar returns to GT cuboids (XYZ) or their radar XY footprints.

    XY uses the convex hull of all eight transformed cuboid corners, preserving
    calibration and GT yaw. Token Z is never used in the XY inclusion test.
    -1 means no GT overlap; -2 means overlapping GT footprints/cuboids.
    """
    if space not in ('xy', 'xyz'):
        raise ValueError('GT space must be xy or xyz')
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    boxes = np.asarray(boxes_lidar, dtype=float).reshape(-1, 7)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError('Invalid GT box geometry')
    if space == 'xyz':
        return memberships(transform(xyz, radar_to_lidar), boxes)
    from scipy.spatial import ConvexHull
    inverse = np.linalg.inv(radar_to_lidar)
    bits = np.array(list(itertools.product((0, 1), repeat=3)))
    labels = np.full(len(xyz), -1, np.int32)
    hits = np.zeros(len(xyz), np.int32)
    for index, (x, y, z, dx, dy, dz, yaw) in enumerate(boxes):
        local = bits * [dx, dy, dz] - [dx / 2, dy / 2, 0]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        corners = transform(local @ rotation.T + [x, y, z], inverse)
        hull = ConvexHull(corners[:, :2])
        inside = np.all(xyz[:, :2] @ hull.equations[:, :2].T
                        + hull.equations[:, 2] <= 1e-8, axis=1)
        labels[inside] = index
        hits[inside] += 1
    labels[hits > 1] = -2
    return labels


def label_regions(regions, point_labels, object_ids, purity=0.6, min_gt_points=3,
                  return_diagnostics=False, reject_ties=False):
    """Assign an unambiguous GT ID to a proposal using its SUPPORT points.

    Denominator includes unlabeled/ambiguous support points, avoiding inflated
    purity. Duplicate GT object IDs are always ambiguous.
    """
    object_ids = np.asarray(object_ids).astype(str)
    _, inverse, counts = np.unique(object_ids, return_inverse=True, return_counts=True)
    is_unique_id = counts[inverse] == 1
    result = {}
    diagnostics = []
    for region in regions:
        if not region['accepted']:
            continue
        indices = np.asarray(region['support_point_ids'], dtype=np.int64)
        labels = np.asarray(point_labels)[indices]
        labels = labels[(labels >= 0) & (labels < len(object_ids))]
        overlap = len(labels)
        duplicate_points = int((~is_unique_id[labels]).sum())
        labels = labels[is_unique_id[labels]]
        winner, n, tied = None, 0, False
        if len(labels):
            values, frequencies = np.unique(labels, return_counts=True)
            winner = int(values[np.argmax(frequencies)])
            n = int(frequencies.max())
            tied = int((frequencies == n).sum()) > 1
        fraction = n / len(indices) if len(indices) else 0.0
        if not len(indices):
            reason = 'no_support'
        elif not overlap:
            reason = 'no_gt_overlap'
        elif not len(labels):
            reason = 'duplicate_id_only'
        elif reject_ties and tied:
            reason = 'ambiguous_dominant'
        elif n < min_gt_points:
            reason = 'insufficient_gt_points'
        elif fraction < purity:
            reason = 'low_purity'
        else:
            reason = 'labeled'
            result[int(region['id'])] = str(object_ids[winner])
        diagnostics.append(dict(region=int(region['id']), support_points=len(indices),
            gt_overlap_points=overlap, duplicate_id_points=duplicate_points,
            dominant_id=str(object_ids[winner]) if winner is not None else None,
            dominant_points=n, dominant_fraction=fraction, dominant_tied=tied, reason=reason))
    return (result, diagnostics) if return_diagnostics else result


def aligned_box(center, dimensions, matrix):
    """History radar AABB -> conservative current radar AABB under rotation."""
    center = np.asarray(center, dtype=np.float64)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    rot = np.asarray(matrix[:3, :3], dtype=np.float64)
    result_center = rot @ center + matrix[:3, 3]
    result_extent = np.abs(rot) @ dimensions
    return result_center, result_extent


def select_match(current, histories, slot, w_center, w_size, y_gates, center_gates,
                 size_scale=0.5, match_space='xyz'):
    """Geometry-only cost after Y and XY/XYZ gates; no GT or tracking ID."""
    if slot not in (1, 2, 3):
        raise ValueError('History slot must be 1, 2 or 3')
    if match_space not in ('xy', 'xyz'):
        raise ValueError('Match space must be xy or xyz')
    if not histories:
        return None
    centers = np.asarray([h['aligned_center'] for h in histories], dtype=np.float64)
    sizes = np.asarray([h['aligned_size'] for h in histories], dtype=np.float64)
    current_center = np.asarray(current['center'], dtype=np.float64)
    current_size = np.asarray(current['box_size'], dtype=np.float64)
    y_diff = np.abs(centers[:, 1] - current_center[1])
    dims = 2 if match_space == 'xy' else 3
    distances = np.linalg.norm(centers[:, :dims] - current_center[:dims], axis=1)
    allowed = (y_diff <= y_gates[slot-1] + 1e-10) & (distances <= center_gates[slot-1] + 1e-10)
    if not allowed.any():
        return None
    size_error = np.mean(np.abs(np.log(np.maximum(sizes[:, :dims], 1e-6) /
                                         np.maximum(current_size[None, :dims], 1e-6))), axis=1)
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
                     scorable_prediction=0, correct=0, wrong=0,
                     all_queries=0, all_predictions=0, current_unlabeled=0))
    for row in rows:
        c['all_queries'] += 1
        c['all_predictions'] += int(row['predicted'])
        if not row.get('current_scorable', True):
            c['current_unlabeled'] += 1
            continue
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
                all_prediction_rate=safe_ratio(c['all_predictions'], c['all_queries']),
                gt_y_rejection_rate=safe_ratio(c['y_missed_true'], c['has_true_history']),
                gt_center_rejection_rate=safe_ratio(c['center_missed_true'], c['has_true_history']))


class FrameData:
    def __init__(self, args):
        self.args = args
        self.diagnostics = {}
        self.region_diagnostics = {}
        self.geometry = va.geometry_module.geometry_from_args(args)
        self.extrinsic = (np.load(args.extrinsic, allow_pickle=False) if args.extrinsic
                          else va.geometry_module.radar_to_lidar_default())
        va.geometry_module.validate_transform(self.extrinsic, 'radar_to_lidar')
        self.config = RegionConfig(seed_percentile=args.seed_percentile, seed_mad=args.seed_mad,
                                   box_size=tuple(args.box_size_m),
                                   merge_space=args.merge_space, xy_box_gap=tuple(args.xy_box_gap_m),
                                   xy_size_limit=not args.no_xy_size_limit,
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
        point_labels = gt_memberships(xyz, gt_boxes, self.extrinsic, self.args.gt_space)
        labels, region_diagnostics = label_regions(proposals, point_labels, gt_ids,
            self.args.gt_purity, self.args.min_gt_points, return_diagnostics=True,
            reject_ties=self.args.gt_space == 'xy')
        accepted = [r for r in proposals if r['accepted']]
        reasons = Counter(row['reason'] for row in region_diagnostics)
        _, id_counts = np.unique(gt_ids, return_counts=True)
        record = dict(scene=scene, order=order, occ_frame_idx=bbox_index,
            radar_path=str(path), bbox_path=str(bbox_dir / f'bbox{bbox_index}.npy'),
            gt_space=self.args.gt_space,
            gt_boxes=len(gt_boxes), duplicate_id_boxes=int(id_counts[id_counts > 1].sum()),
            p1_points=len(xyz), p1_unambiguous_gt_points=int((point_labels >= 0).sum()),
            proposals=len(proposals), accepted=len(accepted), labeled=len(labels),
            proposal_status_counts=json.dumps(dict(Counter(r['status'] for r in proposals)), sort_keys=True))
        for reason in ('no_support', 'no_gt_overlap', 'duplicate_id_only',
                       'insufficient_gt_points', 'low_purity', 'ambiguous_dominant'):
            record[reason] = reasons[reason]
        self.diagnostics[(scene, order)] = record
        self.region_diagnostics[(scene, order)] = [dict(scene=scene, order=order, **row)
                                                   for row in region_diagnostics]
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
    if not (0 <= args.gt_purity <= 1 and args.min_gt_points >= 1 and args.size_scale > 0):
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
    dims = 2 if args.match_space == 'xy' else 3
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
                    scorable = actual is not None
                    truth = [h for h in transformed if scorable and hist_labels.get(h['id']) == actual]
                    y_ok = [h for h in truth if abs(h['aligned_center'][1] - r['center'][1]) <=
                            args.y_gates[slot-1] + 1e-10]
                    center_ok = [h for h in y_ok if np.linalg.norm(h['aligned_center'][:dims] - r['center'][:dims]) <=
                                 args.center_gates[slot-1] + 1e-10]
                    candidates_y = [h for h in transformed if abs(h['aligned_center'][1] - r['center'][1]) <=
                                    args.y_gates[slot-1] + 1e-10]
                    candidates_center = [h for h in candidates_y if np.linalg.norm(h['aligned_center'][:dims] - r['center'][:dims]) <=
                                         args.center_gates[slot-1] + 1e-10]
                    for wd, ws in WEIGHTS:
                        matched = select_match(r, transformed, slot, wd, ws,
                                               args.y_gates, args.center_gates, args.size_scale, args.match_space)
                        prediction = hist_labels.get(matched['id']) if matched else None
                        row = dict(scene=scene, order=order, history_order=history_order, slot=slot,
                                   current_region=int(r['id']), current_id=actual, current_scorable=scorable,
                                   matched_region=matched['id'] if matched else None,
                                   predicted_id=prediction, predicted=matched is not None,
                                   correct=(prediction == actual and matched is not None)
                                       if scorable and (prediction is not None or matched is None) else None,
                                   has_true_history=bool(truth) if scorable else None,
                                   y_missed_true=(bool(truth) and not bool(y_ok)) if scorable else None,
                                   center_missed_true=(bool(y_ok) and not bool(center_ok)) if scorable else None,
                                   current_x=float(r['center'][0]), current_y=float(r['center'][1]),
                                   current_z=float(r['center'][2]), history_accepted=len(transformed),
                                   history_labeled=len(hist_labels), candidates_after_y=len(candidates_y),
                                   candidates_after_center=len(candidates_center))
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
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        writer.writerows(detailed)
    frame_diagnostics = [data.diagnostics[key] for key in sorted(data.diagnostics)]
    region_diagnostics = [row for key in sorted(data.region_diagnostics)
                          for row in data.region_diagnostics[key]]
    for name, rows, empty_fields in (
            ('frame_diagnostics.csv', frame_diagnostics, ['scene', 'order', 'accepted', 'labeled']),
            ('region_gt_diagnostics.csv', region_diagnostics, ['scene', 'order', 'region', 'reason'])):
        with (args.output_dir/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else empty_fields)
            writer.writeheader()
            writer.writerows(rows)
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    (args.output_dir/'report.json').write_text(json.dumps(dict(schema_version=3, config=config, metrics=result,
        frames_evaluated=frame_count, seconds=time.perf_counter()-start,
        frame_diagnostics=frame_diagnostics,
        notes=['GT bbox IDs used for offline scoring ONLY; NEVER in matching.',
               'All accepted current regions are matched and exported, including unknown GT IDs.',
               f'GT membership uses {args.gt_space}; temporal center/size costs use {args.match_space}.',
               'XY GT labels reject tied dominant IDs; purity/point thresholds are explicit in config.',
               'Loose XY membership measures projected association, not 3D point purity or occupancy quality.',
               'Only current_scorable rows contribute to GT precision/recall; unknown verdicts are blank.',
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
                  f"queries={row['all_queries']} labeled={row['current_labeled']} "
                  f"unscorable={row.get('unscorable_prediction', 0)}")
    if not detailed:
        print('WARNING: no region/history queries; inspect frame_diagnostics.csv for accepted counts and frame selection.')
    elif not any(row['current_scorable'] for row in detailed):
        print('WARNING: zero GT-labeled current regions. Matches are exported, but GT metrics are undefined. '
              'Inspect frame_diagnostics.csv and region_gt_diagnostics.csv; widening matching gates cannot fix labeling.')
    print('Outputs:', args.output_dir.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    va.geometry_module.add_coordinate_argument(p)
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
    p.add_argument('--merge-space', choices=('xyz', 'xy'), default='xyz', help='Proposal grouping mode, independent of GT/matching space')
    p.add_argument('--gt-space', choices=('xyz', 'xy'), default='xyz', help='GT support-point inclusion: cuboid or XY footprint; scoring only')
    p.add_argument('--match-space', choices=('xyz', 'xy'), default='xyz', help='Dimensions for temporal center gates and size costs')
    p.add_argument('--xy-box-gap-m', type=float, nargs=2, default=(.5, .5))
    p.add_argument('--no-xy-size-limit', action='store_true', help='XY proposal experiment without size caps; independent shape and GT scoring rules remain')
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
    if args.no_xy_size_limit and args.merge_space != 'xy':
        p.error('--no-xy-size-limit requires --merge-space xy')
    if args.stride <= 0 or args.max_frames_per_scene < 0 or args.end_order < args.start_order:
        p.error('Invalid stride, max frames, or order range')
    evaluate(args)


if __name__ == '__main__':
    main()

