"""Compare original linear power inside GT/proposal boxes, independently of scores."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.temporal_vis.benchmark_region_matching import gt_memberships


def power_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Expected finite nonnegative original linear power')
    return dict(n_points=int(values.size), power_sum=float(values.sum()),
                min_power=float(values.min()) if values.size else None,
                max_power=float(values.max()) if values.size else None,
                mean_power=float(values.mean()) if values.size else None,
                median_power=float(np.median(values)) if values.size else None,
                p05_power=float(np.percentile(values, 5)) if values.size else None,
                p95_power=float(np.percentile(values, 95)) if values.size else None)


def frame_power_statistics(results, xyz, raw_power, gt=None, space='xy'):
    """Return per-box and per-frame union statistics in the same analysis ROI.

    All methods use ORIGINAL channel-2 linear power, never post-RA activation.
    XY includes each box's XY footprint at any height within the analysis ROI;
    XYZ includes its actual cuboid. GT overlaps are included in each individual
    box, and deduplicated in frame totals. Proposal overlaps are treated likewise.
    Each GT box is kept separately, even if its ID recurs in another frame or
    is duplicated in the same frame. Each accepted proposal is kept separately;
    all contained, support, inside-GT and outside-GT returns are separate sets.
    """
    if space not in ('xy', 'xyz'):
        raise ValueError('Power inclusion space must be xy or xyz')
    xyz = np.asarray(xyz, dtype=np.float64)
    raw_power = np.asarray(raw_power, dtype=np.float64)
    if xyz.shape != (len(raw_power), 3):
        raise ValueError('Expected matching XYZ [N,3] and raw power [N]')
    power_statistics(raw_power)  # Validate once, including zero power.
    roi = np.asarray(next(iter(results.values()))['analysis_mask'], dtype=bool)
    if roi.shape != raw_power.shape:
        raise ValueError('Analysis mask does not match raw power')
    for result in results.values():
        if not np.array_equal(result['analysis_mask'], roi):
            raise ValueError('Methods must use the same analysis ROI for power comparison')
    boxes, frames = [], []
    gt_union = np.zeros(len(raw_power), dtype=bool)
    gt_masks = []

    def add(target, method, point_set, mask, **extra):
        target.append(dict(method=method, point_set=point_set, space=space,
                           **extra, **power_statistics(raw_power[mask])))

    if gt is not None:
        lidar_to_radar = np.linalg.inv(gt['radar_to_lidar'])
        for index, box in enumerate(gt['boxes']):
            mask = roi & (gt_memberships(xyz, np.asarray(box)[None],
                                         gt['radar_to_lidar'], space) >= 0)
            gt_union |= mask
            gt_masks.append(mask)
            # GT boxes store LiDAR bottom-center coordinates. Report a common
            # radar-frame geometric center so power can be inspected by range.
            center = np.asarray(box[:3], dtype=float) + [0., 0., float(box[5])/2]
            center = center @ lidar_to_radar[:3, :3].T + lidar_to_radar[:3, 3]
            add(boxes, 'GT', 'box', mask, box_kind='gt', box_index=index,
                box_id=str(gt['ids'][index]), gt_id=str(gt['ids'][index]),
                gt_relation='gt', center_x_m=float(center[0]), center_y_m=float(center[1]),
                center_z_m=float(center[2]), center_range_m=float(np.linalg.norm(center)))
        add(frames, 'GT', 'box', gt_union, n_boxes=len(gt['boxes']))

    dims = 2 if space == 'xy' else 3
    for method, result in results.items():
        accepted = [r for r in result['regions'] if r['accepted']]
        union = np.zeros(len(raw_power), dtype=bool)
        support_union = np.zeros(len(raw_power), dtype=bool)
        for region in accepted:
            low, high = np.asarray(region['box_min']), np.asarray(region['box_max'])
            mask = roi & (xyz[:, :dims] >= low[:dims] - 1e-8).all(axis=1)
            mask &= (xyz[:, :dims] <= high[:dims] + 1e-8).all(axis=1)
            support = np.zeros(len(raw_power), dtype=bool)
            support[np.asarray(region['support_point_ids'], dtype=int)] = True
            support &= roi
            union |= mask
            support_union |= support
            center = (low + high)/2
            gt_points = int((mask & gt_union).sum()) if gt is not None else None
            relation = ('not_evaluated' if gt is None else 'no_returns' if not mask.any()
                        else 'outside_gt' if gt_points == 0 else 'overlaps_gt')
            identity = dict(box_kind='proposal', box_index=int(region['id']),
                            box_id=f'R{region["id"]}', gt_id=None, gt_relation=relation,
                            gt_overlap_fraction=gt_points/int(mask.sum()) if gt is not None and mask.any() else None,
                            overlapping_gt_ids=';'.join(str(gt['ids'][j]) for j, gt_mask in enumerate(gt_masks)
                                                        if (mask & gt_mask).any()) if gt is not None else None,
                            center_x_m=float(center[0]), center_y_m=float(center[1]),
                            center_z_m=float(center[2]), center_range_m=float(np.linalg.norm(center)))
            add(boxes, method, 'box', mask, **identity)
            add(boxes, method, 'support', support, **identity)
            if gt is not None:
                add(boxes, method, 'box_inside_gt', mask & gt_union, **identity)
                add(boxes, method, 'box_outside_gt', mask & ~gt_union, **identity)
        add(frames, method, 'box', union, n_boxes=len(accepted))
        add(frames, method, 'support', support_union, n_boxes=len(accepted))
        if gt is not None:
            add(frames, method, 'box_inside_gt', union & gt_union, n_boxes=len(accepted))
            add(frames, method, 'box_outside_gt', union & ~gt_union, n_boxes=len(accepted))
    return boxes, frames


def all_frames_box_table(box_rows):
    """One row per (scene, frame, method, box index), not one row per ID.

    Bare statistics describe all returns in that individual box. The support_,
    inside_gt_ and outside_gt_ columns describe its separate subsets. No box
    statistics are averaged with another box or another frame in this table.
    """
    metrics = ('n_points', 'power_sum', 'min_power', 'max_power', 'mean_power',
               'median_power', 'p05_power', 'p95_power')
    groups = {}
    prefixes = dict(support='support_', box_inside_gt='inside_gt_', box_outside_gt='outside_gt_')
    for row in box_rows:
        key = (row.get('scene'), row.get('order'), row['method'], row['box_index'], row['space'])
        if row['point_set'] == 'box':
            if key in groups:
                raise ValueError(f'Duplicate box row: {key}')
            groups[key] = {k: v for k, v in row.items() if k != 'point_set'}
            groups[key]['box_key'] = '/'.join(str(v) for v in key)
    for row in box_rows:
        prefix = prefixes.get(row['point_set'])
        if prefix is None:
            continue
        key = (row.get('scene'), row.get('order'), row['method'], row['box_index'], row['space'])
        if key not in groups:
            raise ValueError(f'Subset without corresponding box: {key}')
        groups[key].update({prefix + name: row[name] for name in metrics})
    return list(groups.values())


def write_power_statistics(output_dir, box_rows, frame_rows):
    """Save the primary per-box table plus optional frame/pool reference tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        if not rows:
            return
        with (output_dir / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)

    groups = defaultdict(list)
    for row in frame_rows:
        groups[(row['method'], row['point_set'], row['space'])].append(row)
    summary = []
    for (method, point_set, space), rows in groups.items():
        count = sum(r['n_points'] for r in rows)
        total = sum(r['power_sum'] for r in rows)
        box_means = [r['mean_power'] for r in box_rows
                     if (r['method'], r['point_set'], r['space']) == (method, point_set, space)
                     and r['mean_power'] is not None]
        summary.append(dict(method=method, point_set=point_set, space=space,
                            frames=len(rows), frames_with_points=sum(r['n_points'] > 0 for r in rows),
                            n_points=count, power_sum=total,
                            min_power=min((r['min_power'] for r in rows if r['n_points']), default=None),
                            max_power=max((r['max_power'] for r in rows if r['n_points']), default=None),
                            mean_power=total/count if count else None,
                            macro_box_mean_power=float(np.mean(box_means)) if box_means else None))
    write('all_frames_box_power.csv', all_frames_box_table(box_rows))
    write('box_power.csv', box_rows)
    write('frame_power.csv', frame_rows)
    write('power_summary.csv', summary)
    (output_dir / 'POWER_TABLE_README.md').write_text(
        '# Per-frame, per-box original power statistics\n\n'
        'Primary table: `all_frames_box_power.csv`. One row is one GT or accepted proposal '
        'box in one frame and one method. GT IDs recurring in different frames are never merged; '
        'duplicate IDs in a frame remain separate by box_index.\n\n'
        'All statistics use original linear NPZ power_val[2], not post-RA activations or dB. '
        'Bare n_points/mean_power/min_power/max_power columns use ALL ROI returns inside that '
        'individual box. min/max are observed extrema, not clipping thresholds or percentiles. '
        'p05_power/p95_power are the 5th/95th percentiles. Empty boxes remain in the table with '
        'n_points=0 and blank power statistics, not fabricated zero-power samples.\n\n'
        'space=xy uses XY footprint inclusion with height restricted by the analysis ROI; '
        'space=xyz uses the full cuboid. GT boxes retain their actual rotation and calibration. '
        'center_x_m/center_y_m/center_z_m are radar-frame geometric centers.\n\n'
        'support_* columns use the proposal support points. inside_gt_* / outside_gt_* columns '
        'use the same proposal box split by the union of ALL GT boxes. gt_relation=outside_gt '
        'means no contained return overlaps any GT box, not that the box is definitely clutter. '
        'overlaps_gt includes partial overlap; its outside_gt_* fields still report the remaining points. '
        'GT ID overlap is diagnostic and does not change proposal extraction.\n\n'
        'Overlapping boxes each retain their own contained returns in the primary table. '
        'box_power.csv is a long-format companion with one row per box and subset. '
        'frame_power.csv and power_summary.csv contain optional within-frame union and pooled '
        'reference statistics; these do not replace the primary per-box table.\n',
        encoding='utf-8')
    return summary
