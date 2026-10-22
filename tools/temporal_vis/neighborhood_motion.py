"""GT-free current-frame motion cues for the Stage B offline audit.

Directions describe HISTORY search: -1 = negative X, +1 = positive X,
0 = abstain/use a symmetric box. Radial Doppler is not full target velocity.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class VotingConfig:
    radius: float = 2.
    max_neighbors: int = 64
    min_neighbors: int = 3
    speed_threshold: float = .5
    min_los_cos: float = .7
    min_support: float = .7
    max_mad: float = 1.
    velocity_sigma: float = 1.
    power_sigma: float = 2.
    max_peak_std: float = .5

    def __post_init__(self):
        positive = [self.radius, self.velocity_sigma, self.power_sigma]
        nonnegative = [self.speed_threshold, self.max_mad, self.max_peak_std]
        if (not np.isfinite(positive + nonnegative).all() or min(positive) <= 0
                or min(nonnegative) < 0 or not 0 < self.min_los_cos <= 1
                or not .5 < self.min_support <= 1
                or not 1 <= self.min_neighbors <= self.max_neighbors):
            raise ValueError('Invalid neighborhood voting thresholds')


def load_doppler_table(path):
    path = Path(path)
    if path.suffix.lower() == '.mat':
        from scipy.io import loadmat
        data = loadmat(path)
    elif path.suffix.lower() == '.npy':
        return validate_doppler_table(np.load(path, allow_pickle=False))
    else:
        with np.load(path, allow_pickle=False) as source:
            data = dict(source)
    for key in ('arr_doppler', 'arrDoppler', 'doppler_mps'):
        if key in data:
            return validate_doppler_table(data[key])
    raise ValueError(f'{path}: need arr_doppler, arrDoppler or doppler_mps; use the Doppler table, not info_arr.mat')


def validate_doppler_table(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if (len(values) < 2 or not np.isfinite(values).all()
            or not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))):
        raise ValueError('Doppler table must be finite and strictly monotonic in original bin order')
    return values


def peak_statistics(features, table, max_peak_std):
    """Raw [N,8]: powers[0:3], matching zero-based bins[3:6], mean,var.

    Peaks need not be sorted. Scale before summing to support very large
    linear powers. Invalid peaks and separated/wrap-seam peaks abstain.
    No Doppler unwrapping or hidden integer-alias selection is performed.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 8:
        raise ValueError('Expected raw Doppler8 [N,8], not normalized model features')
    table = validate_doppler_table(table)
    powers, bins = features[:, :3], features[:, 3:6]
    valid = (np.isfinite(features).all(1) & (powers >= 0).all(1)
             & np.isfinite(bins).all(1) & (bins == np.floor(bins)).all(1)
             & (bins >= 0).all(1) & (bins < len(table)).all(1))
    safe_bins = np.clip(np.nan_to_num(bins, nan=0, posinf=0, neginf=0), 0, len(table)-1).astype(int)
    safe_power = np.where(np.isfinite(powers) & (powers >= 0), powers, 0)
    scale = safe_power.max(1)
    weights = safe_power / np.maximum(scale[:, None], np.finfo(float).tiny)
    weights /= np.maximum(weights.sum(1, keepdims=True), np.finfo(float).tiny)
    velocities = table[safe_bins]
    radial = np.sum(weights * velocities, axis=1)
    std = np.sqrt(np.sum(weights * (velocities - radial[:, None])**2, axis=1))
    valid &= (scale > 0) & (std <= max_peak_std)
    # log1p(max peak) remains finite and is independent of peak ordering.
    return radial, np.log1p(scale), valid, std


def ego_velocity(current_pose, previous_pose, extrinsic, dt):
    """Causal interval-average radar-origin velocity in CURRENT radar axes.

    Differencing (LiDAR pose @ radar extrinsic) origins includes the rotating
    lever arm. Uses actual timestamps and no future observations.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite pose time interval required')
    from tools.temporal_vis.vis_alignment import geometry_module
    current = geometry_module.validate_transform(current_pose) @ geometry_module.validate_transform(extrinsic)
    previous = geometry_module.validate_transform(previous_pose) @ extrinsic
    return current[:3, :3].T @ ((current[:3, 3] - previous[:3, 3]) / dt)


def compensate_radial(xyz, measured, velocity, sign):
    if sign not in (-1, 1):
        raise ValueError('Doppler sign must be +1 (table positive away) or -1 (positive approaching)')
    xyz = np.asarray(xyz, dtype=float)
    norm = np.linalg.norm(xyz, axis=1)
    los = xyz / np.maximum(norm[:, None], 1e-12)
    projection = los @ np.asarray(velocity, dtype=float)
    # Direct physical formula. Never fold compensation modulo Doppler period.
    return sign * np.asarray(measured) + projection, los[:, 0], projection


def weighted_median(values, weights):
    ids = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[ids])
    return values[ids[np.searchsorted(cumulative, cumulative[-1] / 2)]]


def predict_directions(xyz, features, table, velocity, sign, config=VotingConfig()):
    """All current P1 tokens vote, including unlabeled/background returns.

    No boxes, Object IDs, GT displacements, historical neighbors or labels
    are accepted by this interface. The query itself cannot vote for itself.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite [N,3] current-frame XYZ')
    if len(features) != len(xyz):
        raise ValueError('XYZ/features token counts disagree')
    radial, log_power, reliable, peak_std = peak_statistics(features, table, config.max_peak_std)
    compensated, los_x, projection = compensate_radial(xyz, radial, velocity, sign)
    reliable &= (xyz[:, 0] > 0) & (los_x >= config.min_los_cos) & np.isfinite(compensated)
    moving = reliable & (np.abs(compensated) > config.speed_threshold)
    single = np.zeros(len(xyz), dtype=np.int8)
    single[moving] = -np.sign(compensated[moving]).astype(np.int8)
    voted = np.zeros_like(single)
    support = np.zeros(len(xyz))
    median = np.full(len(xyz), np.nan)
    mad = np.full(len(xyz), np.nan)
    neighbor_count = np.zeros(len(xyz), dtype=int)
    reason = np.full(len(xyz), 'unreliable_query', dtype='<U32')
    tree = cKDTree(xyz)
    # Bounded neighbors and chunked query avoid an N x N distance matrix.
    for start in range(0, len(xyz), 2048):
        stop = min(start + 2048, len(xyz))
        distances, neighbors = tree.query(xyz[start:stop], k=np.arange(1, min(config.max_neighbors+1, len(xyz))+1),
                                           distance_upper_bound=config.radius)
        for local, i in enumerate(range(start, stop)):
            if not reliable[i]:
                continue
            ids, dist = neighbors[local], distances[local]
            ok = (ids < len(xyz)) & (ids != i) & np.isfinite(dist)
            ids, dist = ids[ok], dist[ok]
            ok = reliable[ids]
            ids, dist = ids[ok][:config.max_neighbors], dist[ok][:config.max_neighbors]
            neighbor_count[i] = len(ids)
            if len(ids) < config.min_neighbors:
                reason[i] = 'too_few_neighbors'
                continue
            # The query's Doppler sign is deliberately NOT a filter: an outlier
            # query can be corrected by a consistent current-frame neighborhood.
            base = np.exp(-.5*(dist/(config.radius/2))**2
                          -.5*((log_power[ids]-log_power[i])/config.power_sigma)**2)
            if base.sum() <= np.finfo(float).tiny:
                reason[i] = 'no_weight'
                continue
            med = weighted_median(compensated[ids], base)
            deviation = weighted_median(np.abs(compensated[ids]-med), base)
            median[i], mad[i] = med, deviation
            weight = base * np.exp(-.5*((compensated[ids]-med)/config.velocity_sigma)**2)
            cls = np.where(np.abs(compensated[ids]) <= config.speed_threshold, 0, np.sign(compensated[ids]))
            totals = np.array([weight[cls == c].sum() for c in [-1,0,1]])
            winner = int(np.argmax(totals)) - 1
            support[i] = totals.max() / totals.sum()
            # Check pre-coherence support too, so weighting cannot turn a
            # genuinely mixed neighborhood into artificial unanimity.
            raw_support = base[cls == winner].sum()/base.sum()
            if support[i] < config.min_support or raw_support < config.min_support or deviation > config.max_mad:
                reason[i] = 'inconsistent_neighbors'
            elif winner == 0 or abs(med) <= config.speed_threshold:
                reason[i] = 'near_zero'
            else:
                voted[i] = -winner
                reason[i] = 'directional'
    return dict(single=single, neighborhood=voted, compensated_radial_mps=compensated,
                measured_radial_mps=radial, ego_projection_mps=projection,
                peak_std_mps=peak_std, reliable=reliable, neighbor_count=neighbor_count,
                support=support, median_radial_mps=median, mad_mps=mad, fallback_reason=reason)
