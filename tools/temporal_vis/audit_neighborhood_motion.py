#!/usr/bin/env python3
"""Stage B: GT-free Doppler direction vs paired Stage A geometric baselines."""
import argparse
import csv
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.temporal_vis import audit_dynamic_objects as audit
from tools.temporal_vis import directional_box_matching as box
from tools.temporal_vis.neighborhood_motion import (
    VotingConfig, ego_velocity, load_doppler_table, predict_directions,
)

METHODS = ('symmetric', 'oracle_direction', 'single_token', 'neighborhood')
DIRECTION_KEYS = ('n_direction_queries', 'n_direction_correct', 'n_direction_reversed', 'n_direction_uncertain')


class MotionDataset(audit.Dataset):
    @lru_cache(maxsize=4)
    def motion_p1(self, order):
        """Current-frame features/geometry only; never reads annotation()."""
        info = self.by_order[order]
        path = info.get('sparse_radar_path') or info['radar_path']
        coords, _ = audit.va.load_p1(path)
        xyz = self.geometry.to_xyz(coords)
        keep = np.isfinite(xyz).all(1) & (np.hypot(xyz[:, 0], xyz[:, 1]) > self.args.ego_mask_radius)
        with np.load(path, allow_pickle=False) as source:
            power = np.asarray(source['power_val'])
            if power.shape[0] != 8:
                raise ValueError(f'{path}: Stage B requires raw 8-channel power_val')
            features = power[:, :len(coords)].T.astype(np.float64)
        return xyz[keep], features[keep], np.flatnonzero(keep)

    def causal_velocity(self, order):
        position = self.positions[order]
        if position == 0:
            return None, None, None
        previous = self.orders[position-1]
        dt = self.time(order) - self.time(previous)
        return ego_velocity(self.pose(order), self.pose(previous), self.ext, dt), previous, dt


def adaptive_counts(search, queries, same_index, directions, k):
    """Per-query directions use the unchanged Stage A BoxSearch/ranking."""
    directions = np.asarray(directions)
    if directions.shape != (len(queries),) or not np.isin(directions, [-1,0,1]).all():
        raise ValueError('One -1/0/+1 direction per query required')
    totals = dict.fromkeys(box.COUNT_KEYS, 0)
    for sign in [-1,0,1]:
        counts = search.counts(queries[directions == sign], same_index,
                               'symmetric' if sign == 0 else 'correct_direction', sign or 1, k)
        for key in box.COUNT_KEYS:
            if key != 'n_history_same':
                totals[key] += counts[key]
    totals['n_history_same'] = int((search.labels == same_index).sum())
    assert totals['n_success'] + totals['n_topk_miss'] + totals['n_outside_box'] == totals['n_eligible']
    return totals


def direction_counts(prediction, truth):
    prediction = np.asarray(prediction)
    return dict(n_direction_queries=len(prediction),
                n_direction_correct=int((prediction == truth).sum()),
                n_direction_reversed=int((prediction == -truth).sum()),
                n_direction_uncertain=int((prediction == 0).sum()))


def direction_metrics(counts):
    decided = counts['n_direction_correct'] + counts['n_direction_reversed']
    return dict(direction_accuracy_decided=audit.ratio(counts['n_direction_correct'], decided),
                direction_correct_fraction=audit.ratio(counts['n_direction_correct'], counts['n_direction_queries']),
                direction_reversal_fraction=audit.ratio(counts['n_direction_reversed'], counts['n_direction_queries']),
                direction_uncertain_fraction=audit.ratio(counts['n_direction_uncertain'], counts['n_direction_queries']))


def summarize(rows):
    summary = []
    for sign, slot, method in sorted({(r['doppler_sign'],r['history_slot'],r['method']) for r in rows}):
        group = [r for r in rows if (r['doppler_sign'],r['history_slot'],r['method']) == (sign,slot,method)]
        counts = {key: sum(r[key] for r in group) for key in (*box.COUNT_KEYS, *DIRECTION_KEYS)}
        summary.append(dict(doppler_sign=sign, history_slot=slot, method=method,
                            object_history_pairs=len(group), **counts, **box.metrics(counts), **direction_metrics(counts)))
    return summary


def write_rows(path, rows, empty_header):
    if rows:
        audit.write_csv(path, rows)
    else:
        path.write_text(','.join(empty_header)+'\n')


def run(args, data, table, config):
    rows, exclusions, frames = [], [], []
    lengths = dict(zip(args.gaps, args.box_lengths))
    signs = [1,-1] if args.doppler_sign == 'both' else [int(args.doppler_sign)]
    detail_stream = (args.output_dir/'motion_token_details.csv').open('w', newline='') if args.save_token_details else None
    writer = None
    try:
        for order in data.selected():
            xyz, features, original_ids = data.motion_p1(order)
            velocity, previous, velocity_dt = data.causal_velocity(order)
            predictions = {}
            # Compute directions for EVERY current token before reading GT.
            for sign in signs:
                prediction = predict_directions(xyz, features, table, np.zeros(3) if velocity is None else velocity, sign, config)
                if velocity is None:
                    prediction['single'][:] = 0
                    prediction['neighborhood'][:] = 0
                    prediction['fallback_reason'][:] = 'no_causal_pose'
                predictions[sign] = prediction
                projection = prediction['ego_projection_mps']
                # Can a stationary target's required relative Doppler even fit
                # inside the table? This flags alias exposure, not alias truth.
                static_table_velocity = -projection/sign
                frames.append(dict(order=order, doppler_sign=sign, n_tokens=len(xyz),
                    velocity_previous_order=previous, velocity_dt_sec=velocity_dt,
                    ego_velocity_available=velocity is not None,
                    ego_vx_mps=None if velocity is None else float(velocity[0]),
                    ego_vy_mps=None if velocity is None else float(velocity[1]),
                    ego_vz_mps=None if velocity is None else float(velocity[2]),
                    n_unreliable=int((~prediction['reliable']).sum()),
                    n_single_directional=int(np.count_nonzero(prediction['single'])),
                    n_neighborhood_directional=int(np.count_nonzero(prediction['neighborhood'])),
                    n_static_prediction_outside_doppler_table=(None if velocity is None else
                        int(((static_table_velocity < table.min()) | (static_table_velocity > table.max())).sum())),
                    fallback_reasons=json.dumps(dict(zip(*np.unique(prediction['fallback_reason'], return_counts=True))), default=int)))
                if detail_stream:
                    for i in range(len(xyz)):
                        detail = dict(order=order, doppler_sign=sign, original_p1_index=int(original_ids[i]),
                                      x_m=xyz[i,0], y_m=xyz[i,1], z_m=xyz[i,2])
                        detail.update({key: prediction[key][i].item() for key in prediction})
                        if writer is None:
                            writer = csv.DictWriter(detail_stream, fieldnames=list(detail)); writer.writeheader()
                        writer.writerow(detail)
            current, labels = data.p1(order)
            if not np.array_equal(current, xyz):
                raise ValueError('Stage A and motion P1 token order/membership disagree')
            boxes, ids, categories = data.annotation(order)
            current_pose = data.pose(order)
            for gap, hist in data.histories(order):
                slot = data.positions[order]-data.positions[hist] if args.history_mode == 'dataset' else gap
                dt = data.time(order)-data.time(hist)
                if dt <= 0:
                    raise ValueError('Nonpositive history time interval')
                history, hist_labels = data.p1(hist)
                hboxes, hids, _ = data.annotation(hist)
                historical_pose = data.pose(hist)
                alignment = np.linalg.inv(current_pose @ data.ext) @ (historical_pose @ data.ext)
                search = box.BoxSearch(audit.transform(history, alignment), hist_labels,
                                       lengths[slot], args.box_reverse, args.box_y_half, args.box_z_half)
                for i, token in enumerate(ids):
                    query_mask = labels == i
                    queries = xyz[query_mask]
                    base = dict(scene=args.scene, order=order, object_id=token, category=int(categories[i]),
                                history_order=hist, history_slot=slot, frame_gap=gap, delta_t_sec=dt)
                    idx = np.flatnonzero(hids == token)
                    reason = None
                    delta, center = np.full(3, np.nan), np.full(3, np.nan)
                    if not len(idx):
                        reason = 'no_same_id_historical_box'
                    else:
                        delta, center = box.oracle_delta(boxes[i], hboxes[idx[0]], current_pose, historical_pose, data.ext)
                        if center[0] <= 0:
                            reason = 'current_center_not_in_front'
                        elif abs(delta[0]) <= args.min_direction_dx:
                            reason = 'ambiguous_near_zero_dx'
                        elif abs(delta[0])/max(np.linalg.norm(delta[:2]),1e-12) < args.min_longitudinal_cos:
                            reason = 'not_longitudinal'
                    base.update(oracle_history_dx_m=float(delta[0]), oracle_history_dy_m=float(delta[1]),
                                oracle_history_dz_m=float(delta[2]))
                    if reason:
                        exclusions.append(dict(**base, reason=reason, n_queries=len(queries)))
                        continue
                    truth = 1 if delta[0] > 0 else -1
                    # Baselines are computed once, then repeated separately for
                    # each sign so signs are never pooled into an apparent gain.
                    baseline = {
                        'symmetric': search.counts(queries,int(idx[0]),'symmetric',truth,args.k),
                        'oracle_direction': search.counts(queries,int(idx[0]),'correct_direction',truth,args.k),
                    }
                    for sign in signs:
                        pred = predictions[sign]
                        choices = dict(symmetric=np.zeros(len(queries),dtype=int),
                            oracle_direction=np.full(len(queries),truth),
                            single_token=pred['single'][query_mask], neighborhood=pred['neighborhood'][query_mask])
                        pair = []
                        for method, directions in choices.items():
                            counts = baseline[method] if method in baseline else adaptive_counts(search,queries,int(idx[0]),directions,args.k)
                            directions_count = direction_counts(directions, truth)
                            pair.append(dict(**base, doppler_sign=sign, method=method, k=args.k,
                                history_direction=truth, box_extension_m=lengths[slot],
                                box_volume_m3=(lengths[slot]+args.box_reverse)*2*args.box_y_half*2*args.box_z_half,
                                **counts, **box.metrics(counts), **directions_count, **direction_metrics(directions_count)))
                        assert len({(r['n_queries'],r['n_eligible'],r['n_no_history']) for r in pair}) == 1
                        rows.extend(pair)
            print(f'motion order={order}: {len(xyz)} current tokens; paired rows={len(rows)}',flush=True)
    finally:
        if detail_stream:
            detail_stream.close()
    summary = summarize(rows)
    write_rows(args.output_dir/'motion_per_object_history.csv',rows,['scene','order','object_id','history_slot','doppler_sign','method'])
    write_rows(args.output_dir/'motion_summary_by_history.csv',summary,['doppler_sign','history_slot','method'])
    write_rows(args.output_dir/'motion_excluded_pairs.csv',exclusions,['scene','order','object_id','history_slot','reason','n_queries'])
    write_rows(args.output_dir/'motion_frame_diagnostics.csv',frames,['order','doppler_sign','n_tokens'])
    def percent(value):
        return 'undefined' if value is None else f'{100*value:.2f}%'
    report = ['# Stage B Doppler neighborhood direction audit','',
        f'K={args.k}; {len(data.selected())} selected frames. Sign hypotheses are reported separately.', '',
        '| Sign | History | Method | Eligible | Coverage | Recall@K | Precision | Direction accuracy (decided) | Reversed/all | Uncertain/all |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f"| {r['doppler_sign']:+d} | t-{r['history_slot']} | {r['method']} | {r['n_eligible']} | "
            + ' | '.join(percent(r[key]) for key in ['candidate_coverage','same_object_recall_at_k',
                'selected_same_precision','direction_accuracy_decided','direction_reversal_fraction','direction_uncertain_fraction'])+' |')
    report += ['', f'Common Stage A cohort exclusions: {len(exclusions)} object/history pairs.',
        'GT boxes/IDs supply evaluation membership, paired cohort and oracle reference only; predicted directions are computed for all current P1 tokens first.',
        'Direction metrics include queries with a matched historical box, even if it has no historical P1 returns; Recall eligibility follows Stage A.',
        'Direct compensation: sign * table Doppler + radar ego velocity projected onto each line of sight.',
        'No velocity unwrapping. A coherent aliased peak can still give the wrong sign; inspect motion_frame_diagnostics.csv.',
        'The LOS guard limits off-axis queries; it cannot identify true transverse motion from one radial velocity.',
        'Uncertain directions use the equal-volume symmetric box. No attention training or occupancy model evaluation is performed.',
        'Direction accuracy is conditional on making a decision: always read it together with uncertain/all and reversed/all.',
        'Zero denominators are undefined. Repeated tokens/objects/frames are correlated.', '']
    (args.output_dir/'motion_results.md').write_text('\n'.join(report))
    metadata = {key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    metadata.update(selected_orders=data.selected(), doppler_table_mps=table.tolist(),
        doppler_sign_meaning='+1: table positive away; -1: table positive approaching. Both is sensitivity, not GT selection.',
        gt_free_prediction=True, compensation='direct, no modulo, no alias correction',
        ego_velocity='causal interval-average derivative of radar origin in current radar axes',
        cohort='identical filters and expanded-bbox proxy as Stage A; no GT inference inputs',
        direction_truth='sign(aligned historical center X - current center X)',
        ranking='Stage A Euclidean distance to original query; original P1 index tie-break',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('neighborhood_motion.py'),Path(box.__file__),Path(audit.__file__)]},
        doppler_table_sha256=hashlib.sha256(args.doppler_bins.read_bytes()).hexdigest())
    (args.output_dir/'motion_protocol.json').write_text(json.dumps(metadata,indent=2))
    print(f'Results: {args.output_dir.resolve()}',flush=True)
    if not rows:
        print('No paired evaluable directions; inspect motion_excluded_pairs.csv.',flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-file',type=Path,default=Path('data/annotations/kradar_dict_test_official_temporal_doppler8.pkl'))
    p.add_argument('--bbox-root',type=Path,default=Path('data/K-RadarOcc/train'))
    p.add_argument('--bins',type=Path,default=Path('data/K-Radar-official-meta/resources/info_arr.mat'))
    p.add_argument('--doppler-bins',type=Path,default=Path('data/K-Radar-official-meta/resources/arr_doppler.mat'))
    p.add_argument('--doppler-sign',choices=['1','-1','both'],default='both')
    p.add_argument('--extrinsic',type=Path)
    p.add_argument('--scene',default='3')
    for name,default in [('start-order',80),('end-order',200),('stride',5),('max-frames',25),('k',8)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--gaps',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--history-mode',choices=['dataset','scene-order'],default='dataset')
    p.add_argument('--box-lengths',type=float,nargs='+',default=[4.,6.,8.])
    for name,default in [('box-reverse',.5),('box-y-half',1.5),('box-z-half',1.),
            ('min-direction-dx',1e-6),('min-longitudinal-cos',0.),('ego-mask-radius',0.),
            ('vote-radius',2.),('speed-threshold',.5),('min-los-cos',.7),('min-support',.7),
            ('max-mad',1.),('velocity-sigma',1.),('power-sigma',2.),('max-peak-std',.5)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--vote-max-neighbors',type=int,default=64)
    p.add_argument('--vote-min-neighbors',type=int,default=3)
    p.add_argument('--save-token-details',action='store_true')
    p.add_argument('--output-dir',type=Path,required=True)
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    values = [*args.box_lengths,args.box_reverse,args.box_y_half,args.box_z_half,
              args.min_direction_dx,args.min_longitudinal_cos,args.ego_mask_radius]
    if (not np.isfinite(values).all() or min(args.box_lengths) <= 0
            or min(args.box_y_half,args.box_z_half) <= 0
            or min(args.box_reverse,args.min_direction_dx,args.ego_mask_radius) < 0
            or not 0 <= args.min_longitudinal_cos <= 1 or min(args.k,args.stride,args.max_frames) < 1
            or args.end_order < args.start_order or min(args.gaps) < 1
            or len(set(args.gaps)) != len(args.gaps) or len(args.gaps) != len(args.box_lengths)):
        p.error('Invalid selection, box dimensions or gaps/lengths')
    try:
        config = VotingConfig(args.vote_radius,args.vote_max_neighbors,args.vote_min_neighbors,
            args.speed_threshold,args.min_los_cos,args.min_support,args.max_mad,
            args.velocity_sigma,args.power_sigma,args.max_peak_std)
    except ValueError as error:
        p.error(str(error))
    table = load_doppler_table(args.doppler_bins)
    data = MotionDataset(args)
    if not data.selected():
        p.error('No selected current frames')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    run(args,data,table,config)


if __name__ == '__main__':
    main()
