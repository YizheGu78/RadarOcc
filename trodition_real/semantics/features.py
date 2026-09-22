"""42-D occupancy-first descriptors; raw RPC Doppler is never unwrapped.

This schema is deliberately incompatible with the older dual-branch RF.
Geometry describes occupied voxel centres. Signal statistics use associated
RPC returns from the causal window. No motion label is a semantic label.
"""
import numpy as np
from sklearn.cluster import DBSCAN

FEATURE_NAMES = (
    'voxel_count', 'return_count', 'extent_x', 'extent_y', 'extent_z',
    'bbox_volume', 'xy_diagonal', 'z_mean', 'z_std', 'z_min', 'z_max',
    'cov_major', 'cov_middle', 'cov_minor', 'linearity',
    'power_mean', 'power_std', 'power_min', 'power_max', 'power_q50', 'power_q90',
    'doppler_raw_mean', 'doppler_raw_std', 'doppler_raw_min', 'doppler_raw_max',
    'doppler_raw_abs_mean', 'doppler_raw_span',
    'range_mean', 'range_std', 'range_min', 'range_max',
    'elevation_mean', 'elevation_std', 'elevation_span',
    'occupancy_mean', 'occupancy_min', 'occupancy_max',
    'unique_frame_count', 'current_return_fraction', 'return_age_mean',
    'return_age_max', 'returns_per_voxel',
)
assert len(FEATURE_NAMES) == 42


def _stats(values, kind):
    if len(values) == 0:
        return [0.] * {'power': 6, 'doppler': 6, 'range': 4, 'elevation': 3}[kind]
    mean, std = float(np.mean(values)), float(np.std(values))
    if kind == 'power':
        return [mean, std, np.min(values), np.max(values), np.median(values), np.quantile(values, .9)]
    if kind == 'doppler':
        return [mean, std, np.min(values), np.max(values), np.mean(np.abs(values)), np.ptp(values)]
    if kind == 'range':
        return [mean, std, np.min(values), np.max(values)]
    return [mean, std, np.ptp(values)]


def extract_features(xyz, returns, ages, probabilities):
    extent = np.ptp(xyz, axis=0)
    z = xyz[:, 2]
    eigen = np.maximum(np.linalg.eigvalsh(np.cov(xyz, rowvar=False)), 0)[::-1] if len(xyz) > 1 else np.zeros(3)
    features = [len(xyz), len(returns), *extent, np.prod(extent), np.linalg.norm(extent[:2]),
                np.mean(z), np.std(z), np.min(z), np.max(z), *eigen,
                (eigen[0] - eigen[1]) / eigen[0] if eigen[0] > 1e-12 else 0.]
    for col, kind in [(3, 'power'), (4, 'doppler'), (5, 'range'), (7, 'elevation')]:
        features.extend(_stats(returns[:, col], kind))
    features.extend([np.mean(probabilities), np.min(probabilities), np.max(probabilities),
                     len(np.unique(ages)), np.mean(ages == 0) if len(ages) else 0.,
                     np.mean(ages) if len(ages) else 0., np.max(ages) if len(ages) else 0.,
                     len(returns) / len(xyz)])
    result = np.asarray(features, np.float64)
    if result.shape != (42,) or not np.all(np.isfinite(result)):
        raise ValueError('Invalid feature vector')
    return result


def proposals(probability, occupied, points, returns, ages, config):
    from trodition_real.adapters.layers import voxel_indices
    voxels = np.argwhere(occupied)
    if not len(voxels):
        return []
    xyz = np.asarray(config.min_xyz) + (voxels + .5) * config.resolution
    scale = np.array([config.dbscan_eps_xy, config.dbscan_eps_xy, config.dbscan_eps_z])
    groups = DBSCAN(eps=1., min_samples=config.dbscan_min_samples, algorithm='kd_tree').fit_predict(xyz / scale)
    # Noise remains occupied; give each isolated cell its own RF proposal.
    noise = np.flatnonzero(groups < 0)
    groups[noise] = np.arange(len(noise)) + groups.max() + 1
    lookup = np.full(config.shape_xyz, -1, np.int32)
    lookup[tuple(voxels.T)] = groups
    indices, valid = voxel_indices(points, config)
    return_group = np.full(len(points), -1, np.int32)
    return_group[valid] = lookup[tuple(indices[valid].T)]
    result = []
    for group in np.unique(groups):
        select = groups == group
        members = voxels[select]
        hits = return_group == group
        feature = extract_features(xyz[select], returns[hits], ages[hits], probability[tuple(members.T)])
        result.append({'voxels': members, 'features': feature})
    return result
