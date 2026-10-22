"""Offline, paired, equal-volume box ablation; no model or attention imports."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tools.temporal_vis.audit_dynamic_objects import ratio, transform, write_csv

METHODS = ('symmetric', 'correct_direction', 'wrong_direction')
COUNT_KEYS = ('n_queries', 'n_history_same', 'n_eligible', 'n_no_history',
              'n_covered', 'n_outside_box', 'n_topk_miss', 'n_success',
              'n_selected', 'n_selected_same', 'n_candidates', 'n_same_candidates')


def box_limits(method, direction, extension, reverse, y_half, z_half):
    """Bounds relative to the *original* query, in current radar XYZ axes."""
    if method not in METHODS or direction not in (-1, 1):
        raise ValueError('Unknown method or non-binary history direction')
    width = extension + reverse
    if method == 'symmetric':
        low, high = -width / 2, width / 2
    else:
        sign = direction if method == 'correct_direction' else -direction
        low, high = (-reverse, extension) if sign > 0 else (-extension, reverse)
    return np.array([low, -y_half, -z_half]), np.array([high, y_half, z_half])


def geometric_center(box):
    center = np.array(box[:3], dtype=float, copy=True)
    center[2] += box[5] / 2
    return center


def oracle_delta(current_box, historical_box, current_pose, historical_pose, extrinsic):
    """History minus current centre after ego alignment, in current radar frame.

    Ego motion is removed before sign(dx) is taken. Heading/category labels are
    never used as direction. Only the sign, not the displacement magnitude, is
    supplied to search-region construction.
    """
    current = transform(geometric_center(current_box)[None], np.linalg.inv(extrinsic))[0]
    h_to_cur_radar = np.linalg.inv(current_pose @ extrinsic) @ historical_pose
    historical = transform(geometric_center(historical_box)[None], h_to_cur_radar)[0]
    return historical - current, current


class BoxSearch:
    def __init__(self, history, labels, extension, reverse, y_half, z_half):
        self.history = np.asarray(history)
        self.labels = np.asarray(labels)
        self.half = np.array([(extension + reverse) / 2, y_half, z_half])
        self.tree = cKDTree(self.history / self.half)
        self.extension, self.reverse = extension, reverse
        self.y_half, self.z_half = y_half, z_half

    def counts(self, queries, same_index, method, direction, k):
        counts = dict.fromkeys(COUNT_KEYS, 0)
        counts.update(n_queries=len(queries), n_history_same=int((self.labels == same_index).sum()))
        if counts['n_history_same'] == 0:
            counts['n_no_history'] = len(queries)
            return counts
        counts['n_eligible'] = len(queries)
        if not len(queries):
            return counts
        low, high = box_limits(method, direction, self.extension, self.reverse, self.y_half, self.z_half)
        shift = (low + high) / 2
        # A normalized L-infinity query is exactly an axis-aligned box. The
        # widened prefilter guards roundoff; exact inclusive bounds follow.
        pools = self.tree.query_ball_point((queries + shift) / self.half, 1 + 1e-9, p=np.inf)
        for query, pool in zip(queries, pools):
            ids = np.asarray(pool, dtype=int)
            relative = self.history[ids] - query
            keep = np.all((relative >= low - 1e-9) & (relative <= high + 1e-9), axis=1)
            ids, relative = ids[keep], relative[keep]
            same = self.labels[ids] == same_index
            counts['n_candidates'] += len(ids)
            counts['n_same_candidates'] += int(same.sum())
            covered = bool(same.any())
            counts['n_covered'] += int(covered)
            # Rank by Euclidean distance to the original query, NOT box centre.
            # Original token index only breaks exact distance ties reproducibly.
            ranked = np.lexsort((ids, np.einsum('ij,ij->i', relative, relative)))[:k]
            picked = same[ranked]
            success = bool(picked.any())
            counts['n_selected'] += len(picked)
            counts['n_selected_same'] += int(picked.sum())
            counts['n_success'] += int(success)
            counts['n_outside_box'] += int(not covered)
            counts['n_topk_miss'] += int(covered and not success)
        assert counts['n_outside_box'] + counts['n_topk_miss'] + counts['n_success'] == counts['n_eligible']
        assert counts['n_covered'] == counts['n_success'] + counts['n_topk_miss']
        return counts


def metrics(counts):
    return dict(candidate_coverage=ratio(counts['n_covered'], counts['n_eligible']),
                same_object_recall_at_k=ratio(counts['n_success'], counts['n_eligible']),
                selected_same_precision=ratio(counts['n_selected_same'], counts['n_selected']),
                mean_candidates=ratio(counts['n_candidates'], counts['n_eligible']),
                mean_same_candidates=ratio(counts['n_same_candidates'], counts['n_eligible']))


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for values, group in sorted(groups.items()):
        counts = {key: sum(r[key] for r in group) for key in COUNT_KEYS}
        output.append(dict(zip(keys, values), object_history_pairs=len(group),
                           delta_t_min=min(r['delta_t_sec'] for r in group),
                           delta_t_max=max(r['delta_t_sec'] for r in group),
                           **counts, **metrics(counts)))
    return output


def plot_summary(path, summary, k):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    slots = sorted({r['history_slot'] for r in summary})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    colors = ['#777777', '#16855b', '#c55442']
    for ax, metric, title in zip(axes,
            ['candidate_coverage', 'same_object_recall_at_k', 'selected_same_precision'],
            ['Candidate coverage', f'Same-object Recall@{k}', 'Selected same precision']):
        for m, (method, color) in enumerate(zip(METHODS, colors)):
            values = [next(r[metric] for r in summary if r['method'] == method and r['history_slot'] == slot)
                      for slot in slots]
            ax.bar(np.arange(len(slots)) + (m-1)*.25,
                   [100*v if v is not None else np.nan for v in values], .24, color=color, label=method)
        ax.set(title=title, ylim=(0,100), ylabel='Percent')
        ax.set_xticks(np.arange(len(slots)), [f't-{slot}' for slot in slots])
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run(args, data):
    rows, excluded = [], []
    parameters = dict(zip(args.gaps, args.box_lengths))
    selected = data.selected()
    for order in selected:
        current, labels = data.p1(order)
        boxes, ids, cats = data.annotation(order)
        current_pose = data.pose(order)
        for gap, hist in data.histories(order):
            slot = data.positions[order] - data.positions[hist] if args.history_mode == 'dataset' else gap
            extension = parameters[slot]
            dt = data.time(order) - data.time(hist)
            if dt <= 0:
                raise ValueError('Nonpositive history time interval')
            history, hist_labels = data.p1(hist)
            hboxes, hids, _ = data.annotation(hist)
            historical_pose = data.pose(hist)
            alignment = np.linalg.inv(current_pose @ data.ext) @ (historical_pose @ data.ext)
            history = transform(history, alignment)
            search = BoxSearch(history, hist_labels, extension, args.box_reverse, args.box_y_half, args.box_z_half)
            for i, token in enumerate(ids):
                queries = current[labels == i]
                base = dict(scene=args.scene, order=order, object_id=token, category=int(cats[i]),
                            history_order=hist, history_slot=slot, frame_gap=gap, delta_t_sec=dt)
                idx = np.flatnonzero(hids == token)
                reason = None
                if not len(idx):
                    reason = 'no_same_id_historical_box'
                    delta, center = np.full(3, np.nan), np.full(3, np.nan)
                else:
                    delta, center = oracle_delta(boxes[i], hboxes[idx[0]], current_pose, historical_pose, data.ext)
                    if center[0] <= 0:
                        reason = 'current_center_not_in_front'
                    elif abs(delta[0]) <= args.min_direction_dx:
                        reason = 'ambiguous_near_zero_dx'
                    elif abs(delta[0]) / max(np.linalg.norm(delta[:2]), 1e-12) < args.min_longitudinal_cos:
                        reason = 'not_longitudinal'
                base.update(oracle_history_dx_m=float(delta[0]), oracle_history_dy_m=float(delta[1]),
                            oracle_history_dz_m=float(delta[2]), current_center_radar_x_m=float(center[0]))
                if reason:
                    excluded.append(dict(**base, reason=reason, n_queries=len(queries)))
                    continue
                direction = 1 if delta[0] > 0 else -1
                pair_rows = []
                for method in METHODS:
                    counts = search.counts(queries, int(idx[0]), method, direction, args.k)
                    low, high = box_limits(method, direction, extension, args.box_reverse, args.box_y_half, args.box_z_half)
                    row = dict(**base, method=method, history_direction=direction, k=args.k,
                               box_x_low_m=float(low[0]), box_x_high_m=float(high[0]),
                               box_y_half_m=args.box_y_half, box_z_half_m=args.box_z_half,
                               box_volume_m3=float(np.prod(high-low)), **counts, **metrics(counts))
                    pair_rows.append(row)
                assert len({(r['n_queries'], r['n_eligible'], r['n_no_history']) for r in pair_rows}) == 1
                assert np.ptp([r['box_volume_m3'] for r in pair_rows]) < 1e-9
                rows.extend(pair_rows)
        print(f'box order={order}: {len(ids)} objects; paired rows={len(rows)}', flush=True)
    # Exclusions apply equally to all arms, never only to an unfavorable arm.
    exclusion_path = args.output_dir / 'box_excluded_pairs.csv'
    if excluded:
        write_csv(exclusion_path, excluded)
    else:
        exclusion_path.write_text('scene,order,object_id,history_order,history_slot,reason,n_queries\n')
    write_csv(args.output_dir / 'box_per_object_history.csv', rows)
    summary = summarize(rows, ['history_slot', 'method'])
    write_csv(args.output_dir / 'box_summary_by_history.csv', summary)
    write_csv(args.output_dir / 'box_summary_by_direction.csv', summarize(rows, ['history_slot', 'history_direction', 'method']))
    write_csv(args.output_dir / 'box_summary_by_object.csv', summarize(rows, ['object_id', 'history_slot', 'method']))
    if args.max_plots:
        plot_summary(args.output_dir / 'box_comparison.png', summary, args.k)
    report = ['# Directional box matching (GT direction oracle)', '',
              f'Scene {args.scene}, orders {args.start_order}–{args.end_order}, '
              f'{len(selected)} selected current frames, stride {args.stride}, K={args.k}.', '',
              'All three arms use paired queries and equal box volumes; distance ranking remains relative to the original query.', '',
              '| History | Method | Eligible | Coverage | Recall@K | Precision | Outside box | Top-K miss |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for slot in sorted(parameters):
        for method in METHODS:
            matches = [r for r in summary if r['history_slot']==slot and r['method']==method]
            if not matches:
                continue
            r = matches[0]
            percentages = ['undefined' if r[key] is None else f'{100*r[key]:.2f}%'
                           for key in ['candidate_coverage','same_object_recall_at_k','selected_same_precision']]
            report.append(f"| t-{slot} | {method} | {r['n_eligible']} | {' | '.join(percentages)} | {r['n_outside_box']} | {r['n_topk_miss']} |")
    report += ['', f'Excluded object/history pairs: {len(excluded)}; see box_excluded_pairs.csv.',
               'Membership is an expanded-bbox proxy; GT supplies direction only. This is not a deployable model evaluation.',
               'Near-zero X displacement is excluded equally from all methods. No sphere search or attention is run.',
               'Repeated queries, tracks, and frames are correlated; these descriptive totals do not establish statistical significance.', '']
    (args.output_dir / 'box_results.md').write_text('\n'.join(report))
    protocol = dict(methods=METHODS, selected_orders=selected,
                    n_paired_object_histories=len(rows)//3, n_excluded_object_histories=len(excluded),
                    direction='sign of aligned historical geometric centre minus current centre, in current radar X',
                    symmetric_x='[-(L+reverse)/2, +(L+reverse)/2]',
                    volume='(L+reverse) * (2*y_half) * (2*z_half)',
                    ranking='Euclidean distance to original query; original P1 index breaks exact ties',
                    coverage='queries with >=1 same-object candidate in box / eligible queries',
                    oracle_use='direction sign only; magnitude does not change box dimensions',
                    counts='micro totals; history token counts repeated per object/history, not unique scene tokens',
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [Path(__file__), Path(__file__).with_name('audit_dynamic_objects.py')]})
    (args.output_dir / 'box_protocol.json').write_text(json.dumps(protocol, indent=2))
