"""Deterministic GT-free temporal query budgets; full single-frame tokens survive."""
from pathlib import Path

import numpy as np

from .doppler_query_pruning import (circular_peaks, static_consistency,
    validate_doppler_table, UNCERTAIN, STATIC_CONSISTENT, NONSTATIC_EVIDENCE)
from .ego_alignment import radar_to_lidar_default, validate_transform


def load_table(path):
    path = Path(path)
    if path.suffix == '.mat':
        from scipy.io import loadmat
        data = loadmat(path)
    elif path.suffix == '.npy':
        return validate_doppler_table(np.load(path, allow_pickle=False))
    else:
        with np.load(path, allow_pickle=False) as source:
            data = dict(source)
    for key in ('arr_doppler', 'arrDoppler', 'doppler_mps'):
        if key in data:
            return validate_doppler_table(data[key])
    raise ValueError(f'No Doppler bin table in {path}')


def causal_ego_velocity(poses, timestamps, valid, extrinsic=None, max_dt=1., max_speed=70.):
    """Latest valid PAST input pose; interval-average radar-origin velocity."""
    poses, times, valid = np.asarray(poses), np.asarray(timestamps), np.asarray(valid, bool)
    if poses.shape != (len(valid), 4, 4) or times.shape != valid.shape:
        raise ValueError('Matching temporal poses/timestamps/frame validity required')
    if not valid[-1] or not np.isfinite(times[-1]):
        return None, 'invalid_current_time'
    history = np.flatnonzero(valid[:-1] & np.isfinite(times[:-1]) & (times[:-1] < times[-1]))
    if not len(history):
        return None, 'no_past_pose'
    previous = history[np.argmax(times[history])]
    dt = float(times[-1]-times[previous])
    if not 0 < dt <= max_dt:
        return None, 'time_gap_guard'
    try:
        e = validate_transform(radar_to_lidar_default() if extrinsic is None else extrinsic)
        current = validate_transform(poses[-1]) @ e
        past = validate_transform(poses[previous]) @ e
    except ValueError:
        return None, 'invalid_pose'
    velocity = current[:3, :3].T @ ((current[:3, 3]-past[:3, 3])/dt)
    if np.linalg.norm(velocity) > max_speed:
        return None, 'speed_guard'
    return velocity, 'ok'


def spread_by_power(coords, features, ids, count, angular_cells=(8, 4)):
    """Round-robin occupied angular cells, highest power within each cell.

    Cell bounds depend only on this range's coordinates, never GT. Coordinate
    tie breaks keep results invariant to model power sorting (except identical
    duplicate tokens, which carry the same observation).
    """
    if not count:
        return np.empty(0, np.int64)
    angles = coords[ids, 1:3]
    low, high = coords[:, 1:3].min(0), coords[:, 1:3].max(0)
    cells = np.minimum(((angles-low)/np.maximum(high-low+1, 1)*angular_cells).astype(int),
                       np.asarray(angular_cells)-1)
    cell = cells[:, 0]*angular_cells[1]+cells[:, 1]
    power = np.max(features[ids, :3], axis=1)
    order = np.lexsort((ids, angles[:, 1], angles[:, 0], -power, cell))
    sorted_cell = cell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cell))+1]
    depth = np.arange(len(ids))-np.repeat(starts, np.diff(np.r_[starts, len(ids)]))
    # Every occupied cell contributes once before any contributes twice.
    order2 = np.lexsort((sorted_cell, depth))
    layer = depth[order2]
    last_depth = layer[count-1]
    complete = order[order2[layer < last_depth]]
    last = order[order2[layer == last_depth]]
    remaining = count-len(complete)
    # A partial round must span the occupied cells, not prefer low azimuth IDs.
    positions = np.floor((np.arange(remaining)+.5)*len(last)/remaining).astype(int)
    return ids[np.r_[complete, last[positions]]]


class TemporalQueryBudget:
    """Per-range budget with motion/uncertain/static reserves 1/2,1/4,1/4.

    Unfilled class quotas go to motion, then uncertainty, then static. Unknown
    ego/sign evidence cannot bypass the fixed budget: it receives spatial/power
    sampling. 'both_consensus' abstains whenever sign hypotheses disagree.
    """
    def __init__(self, per_range=64,
                 doppler_bins='data/K-Radar-official-meta/resources/arr_doppler.mat',
                 doppler_sign='both_consensus', threshold_mps=.2,
                 max_peak_std=.2, min_concentration=.8,
                 class_fractions=(.5, .25, .25), angular_cells=(8, 4),
                 max_ego_dt=1., max_ego_speed=70.):
        if int(per_range) != per_range or per_range <= 0:
            raise ValueError('per_range must be a positive integer')
        if doppler_sign not in ('both_consensus', 1, -1):
            raise ValueError('doppler_sign must be both_consensus, 1 or -1')
        fractions = np.asarray(class_fractions, float)
        if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions < 0).any() or not np.isclose(fractions.sum(), 1):
            raise ValueError('Motion/uncertain/static fractions must sum to 1')
        if len(angular_cells) != 2 or any(int(x) != x or x < 1 for x in angular_cells):
            raise ValueError('Positive integer angular cell counts required')
        if not np.isfinite([max_ego_dt, max_ego_speed]).all() or min(max_ego_dt, max_ego_speed) <= 0:
            raise ValueError('Positive finite ego guards required')
        self.per_range, self.table = int(per_range), load_table(doppler_bins)
        from .doppler_query_pruning import fft_period
        if not 0 < threshold_mps < fft_period(self.table)/2:
            raise ValueError('Threshold must lie in (0, Doppler period/2)')
        self.sign, self.threshold = doppler_sign, threshold_mps
        self.max_std, self.concentration = max_peak_std, min_concentration
        self.fractions, self.cells = fractions, tuple(angular_cells)
        self.max_ego_dt, self.max_ego_speed = max_ego_dt, max_ego_speed

    def select_frame(self, features, coords, xyz, ego_velocity=None, eligible=None):
        features, coords, xyz = np.asarray(features), np.asarray(coords), np.asarray(xyz)
        n = len(features)
        if features.shape != (n, 8) or coords.shape != (n, 3) or xyz.shape != (n, 3):
            raise ValueError('Expected matching descriptors [N,8], coords/XYZ [N,3]')
        if not np.isfinite(features).all() or not np.isfinite(coords).all() or not np.array_equal(coords, coords.astype(int)):
            raise ValueError('Finite features and integer R/A/E coordinates required')
        eligible = np.ones(n, bool) if eligible is None else np.asarray(eligible, bool)
        if eligible.shape != (n,):
            raise ValueError('eligible must be [N]')
        peaks = circular_peaks(features, self.table, self.max_std, self.concentration)
        signs = (1, -1) if self.sign == 'both_consensus' else (self.sign,)
        consistency = [static_consistency(xyz, peaks, ego_velocity, sign) for sign in signs]
        reliable = np.logical_and.reduce([c['reliable'] for c in consistency])
        static = reliable & np.logical_and.reduce([np.abs(c['residual_mps']) <= self.threshold for c in consistency])
        moving = reliable & np.logical_and.reduce([np.abs(c['residual_mps']) > self.threshold for c in consistency])
        state = np.full(n, UNCERTAIN, np.uint8)
        state[static], state[moving] = STATIC_CONSISTENT, NONSTATIC_EVIDENCE
        keep = np.zeros(n, bool)
        classes = (NONSTATIC_EVIDENCE, UNCERTAIN, STATIC_CONSISTENT)
        for r in np.unique(coords[:, 0]):
            ids = np.flatnonzero((coords[:, 0] == r) & eligible)
            target = min(self.per_range, len(ids))
            if not target:
                continue
            groups = [ids[state[ids] == cls] for cls in classes]
            exact = self.fractions*target
            quota = np.floor(exact).astype(int)
            for j in np.argsort(-(exact-quota), kind='stable')[:target-int(quota.sum())]:
                quota[j] += 1
            quota = np.minimum(quota, [len(g) for g in groups])
            missing = target-int(quota.sum())
            for j, group in enumerate(groups):
                extra = min(missing, len(group)-quota[j])
                quota[j] += extra
                missing -= extra
            # Pass local range coordinates for consistent cell boundaries.
            local_coords = coords[ids]
            for group, count in zip(groups, quota):
                local = np.searchsorted(ids, group)
                chosen = spread_by_power(local_coords, features[ids], local, int(count), self.cells)
                keep[ids[chosen]] = True
        return keep, dict(state=state, residual_mps=np.stack([c['residual_mps'] for c in consistency]),
                          peak_reliable=peaks['reliable'])

    def select_batch(self, features, coords, raw_xyz, poses, timestamps, valid_mask,
                     extrinsic=None, point_valid_mask=None):
        queries, diagnostics = [], []
        for b in range(len(features)):
            velocity, reason = causal_ego_velocity(poses[b], timestamps[b], valid_mask[b],
                extrinsic, self.max_ego_dt, self.max_ego_speed)
            eligible = None if point_valid_mask is None else point_valid_mask[b, -1]
            keep, detail = self.select_frame(features[b, -1], coords[b, -1], raw_xyz[b, -1], velocity, eligible)
            queries.append(np.flatnonzero(keep))
            diagnostics.append(dict(ego_reason=reason, kept=int(keep.sum()), total=len(keep),
                selected_by_class={str(int(cls)): int((keep & (detail['state'] == cls)).sum())
                                   for cls in (UNCERTAIN, STATIC_CONSISTENT, NONSTATIC_EVIDENCE)}))
        return queries, diagnostics
