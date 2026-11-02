"""GT-free P1 reflection proposals. Geometry only; no detector or learned features.

Point IDs always index the original NPZ's first 175*250 returns, NOT a GPU's
power-sorted P1 tensor. BEV is a temporary raster, not a network feature map.
"""
from dataclasses import asdict, dataclass
import heapq
import itertools

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RegionConfig:
    xy_resolution: float = 0.5
    z_resolution: float = 1.0
    x_limits: tuple = (0.0, 85.0)
    y_limits: tuple = (-60.0, 60.0)
    z_limits: tuple = (-5.0, 8.0)
    range_band_m: float = 8.0
    seed_percentile: float = 95.0
    support_percentile: float = 80.0
    seed_mad: float = 3.0
    support_mad: float = 0.5
    peak_separation_m: float = 1.5
    max_seed_count: int = 128
    grow_half_size: tuple = (4.0, 2.0, 2.0)
    box_size: tuple = (6.0, 3.0, 3.0)
    min_points: int = 6
    min_cells: int = 3
    max_aspect: float = 8.0

    def validate(self):
        positive = (self.xy_resolution, self.z_resolution, self.range_band_m,
                    self.peak_separation_m, self.max_aspect,
                    *self.grow_half_size, *self.box_size)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError('Grid, band, peak, size and aspect settings must be finite and positive')
        for limits in (self.x_limits, self.y_limits, self.z_limits):
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Each analysis limit must be a finite increasing pair')
        if len(self.box_size) != 3 or len(self.grow_half_size) != 3:
            raise ValueError('box_size and grow_half_size must contain XYZ values')
        if not 0 < self.support_percentile < self.seed_percentile < 100:
            raise ValueError('Require 0 < support_percentile < seed_percentile < 100')
        if not np.isfinite([self.seed_mad, self.support_mad]).all() or not 0 <= self.support_mad <= self.seed_mad:
            raise ValueError('Require finite 0 <= support_mad <= seed_mad')
        if min(self.max_seed_count, self.min_points, self.min_cells) < 1:
            raise ValueError('Seed, point and cell counts must be positive')
        if np.prod(self.grid_shape) > 8_000_000:
            raise ValueError('Analysis grid exceeds 8 million cells; increase resolution or reduce limits')
        return self

    @property
    def lower(self):
        return np.array([self.x_limits[0], self.y_limits[0], self.z_limits[0]])

    @property
    def upper(self):
        return np.array([self.x_limits[1], self.y_limits[1], self.z_limits[1]])

    @property
    def spacing(self):
        return np.array([self.xy_resolution, self.xy_resolution, self.z_resolution])

    @property
    def grid_shape(self):
        return tuple(np.ceil((self.upper - self.lower) / self.spacing).astype(int))


def power_masks(xyz, power, eligible, config):
    """Per-slant-range-band quantiles AND robust contrast; constant bands seed nothing.

    A percentile alone forces proposals even in noise-only bands. The extra MAD
    contrast gate reduces that behavior, without claiming a CFAR probability.
    """
    support = np.zeros(len(power), bool)
    seeds = np.zeros(len(power), bool)
    contrast = np.zeros(len(power), np.float64)
    thresholds = []
    radius = np.linalg.norm(xyz[eligible], axis=1)
    bands = np.floor(radius / config.range_band_m).astype(int)
    ids = np.flatnonzero(eligible)
    for band in np.unique(bands):
        selected = ids[bands == band]
        values = power[selected]
        median = float(np.median(values))
        sigma = float(1.4826 * np.median(np.abs(values - median)))
        low, high = np.percentile(values, [config.support_percentile, config.seed_percentile])
        low = float(max(low, median + config.support_mad * sigma))
        high = float(max(high, median + config.seed_mad * sigma))
        epsilon = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values)))) * 16
        nonconstant = float(np.ptp(values)) > epsilon
        if nonconstant:
            above_background = values > median + epsilon
            support[selected] = (values >= low) & above_background
            seeds[selected] = (values >= high) & above_background
            scale = max(sigma, high - median, epsilon)
            contrast[selected] = np.clip((values - median) / scale, 0, 100)
        thresholds.append(dict(range_min_m=float(band * config.range_band_m),
            range_max_m=float((band + 1) * config.range_band_m), count=len(selected),
            median=median, robust_sigma=sigma, support_threshold=low,
            seed_threshold=high, constant=not nonconstant))
    return support, seeds, contrast, thresholds


def seed_markers(strength, seed_grid, config):
    """Separate local maxima even when their low-strength support is connected."""
    window = tuple(2 * max(1, int(np.ceil(config.peak_separation_m / s))) + 1
                   for s in config.spacing)
    local = ndimage.maximum_filter(strength, size=window, mode='constant', cval=-np.inf)
    peaks = seed_grid & (strength >= local)
    labels, count = ndimage.label(peaks, structure=np.ones((3, 3, 3), bool))
    slices = ndimage.find_objects(labels)
    candidates = []
    for label, section in enumerate(slices, 1):
        local_ids = np.argwhere(labels[section] == label)
        cells = local_ids + np.array([s.start for s in section])
        values = strength[tuple(cells.T)]
        best = cells[values == values.max()]
        # On a plateau use the central supported cell, not a bounding-box corner.
        peak = best[np.argmin(np.sum((best - best.mean(axis=0)) ** 2, axis=1))]
        candidates.append((float(strength[tuple(peak)]), tuple(int(v) for v in peak)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    kept = candidates[:config.max_seed_count]
    markers = np.zeros(strength.shape, np.int32)
    for label, (_, cell) in enumerate(kept, 1):
        markers[cell] = label
    return markers, [cell for _, cell in kept], dict(local_maxima=count,
        seeds_used=len(kept), seeds_dropped_by_cap=max(0, len(candidates) - len(kept)))


def grow_regions(support_grid, seed_cells, config):
    """Multi-source geodesic Voronoi growth on supported cells only.

    No path may cross an empty cell. Growth is bounded relative to each seed;
    weak bridges do not merge independent seeds into a single region.
    """
    shape = support_grid.shape
    owner = np.zeros(shape, np.int32)
    distance = np.full(shape, np.inf, np.float64)
    queue = []
    for label, cell in enumerate(seed_cells, 1):
        if not support_grid[cell]:
            raise ValueError('Every seed must be a supported cell')
        owner[cell] = label
        distance[cell] = 0.0
        heapq.heappush(queue, (0.0, label, *cell))
    neighbors = [(*o, float(np.linalg.norm(np.asarray(o) * config.spacing)))
                 for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
    # Scalar index arithmetic avoids allocating NumPy arrays for every edge.
    # Growth visits supported cells once, rather than searching per P1 query.
    nx, ny, nz = shape
    sx, sy, sz = config.spacing
    hx, hy, hz = config.grow_half_size
    while queue:
        cost, label, x, y, z = heapq.heappop(queue)
        here = (x, y, z)
        if owner[here] != label or cost != distance[here]:
            continue
        seed_x, seed_y, seed_z = seed_cells[label - 1]
        for dx, dy, dz, step in neighbors:
            xx, yy, zz = x + dx, y + dy, z + dz
            if not (0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz):
                continue
            cell = (xx, yy, zz)
            if (not support_grid[cell] or abs(xx - seed_x) * sx > hx
                    or abs(yy - seed_y) * sy > hy or abs(zz - seed_z) * sz > hz):
                continue
            next_cost = cost + step
            if next_cost < distance[cell] - 1e-12:
                distance[cell] = next_cost
                owner[cell] = label
                heapq.heappush(queue, (next_cost, label, *cell))
    return owner


def extract_regions(xyz, power, config=None):
    config = (config or RegionConfig()).validate()
    xyz = np.asarray(xyz, np.float64)
    power = np.asarray(power, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or power.shape != (len(xyz),):
        raise ValueError('Expected XYZ [N,3] and stored power [N]')
    eligible = (np.isfinite(xyz).all(axis=1) & np.isfinite(power)
                & (xyz >= config.lower).all(axis=1) & (xyz < config.upper).all(axis=1))
    support, seed_points, contrast, thresholds = power_masks(xyz, power, eligible, config)
    ids = np.flatnonzero(eligible)
    cells = np.floor((xyz[ids] - config.lower) / config.spacing).astype(int)
    shape = config.grid_shape
    strength = np.full(shape, -np.inf, np.float64)
    raw_power = np.full(shape, -np.inf, np.float64)
    counts = np.zeros(shape, np.int32)
    support_grid = np.zeros(shape, bool)
    seed_grid = np.zeros(shape, bool)
    if len(ids):
        location = tuple(cells.T)
        np.maximum.at(strength, location, contrast[ids])
        np.maximum.at(raw_power, location, power[ids])
        np.add.at(counts, location, 1)
        np.logical_or.at(support_grid, location, support[ids])
        np.logical_or.at(seed_grid, location, seed_points[ids])
    markers, seeds, seed_stats = seed_markers(strength, seed_grid, config)
    owner = grow_regions(support_grid, seeds, config)
    point_region = np.full(len(xyz), -1, np.int32)
    grid_owner = owner[tuple(cells.T)] if len(ids) else np.zeros(0, int)
    assigned = support[ids] & (grid_owner > 0)
    point_region[ids[assigned]] = grid_owner[assigned] - 1
    regions = []
    cell_counts = np.bincount(owner.ravel(), minlength=len(seeds) + 1)
    for region_id, seed in enumerate(seeds):
        members = np.flatnonzero(point_region == region_id)
        points = xyz[members]
        low, high = points.min(axis=0), points.max(axis=0)
        robust_low, robust_high = np.percentile(points, [10, 90], axis=0)
        center = (robust_low + robust_high) / 2
        span = high - low
        cell_count = int(cell_counts[region_id + 1])
        reasons = []
        if len(members) < config.min_points or cell_count < config.min_cells:
            reasons.append('small_support')
        xy_span = np.maximum(span[:2], config.xy_resolution)
        if xy_span.max() / xy_span.min() > config.max_aspect:
            reasons.append('elongated_support')
        box_min = center - np.asarray(config.box_size) / 2
        box_max = center + np.asarray(config.box_size) / 2
        # Include weaker ORIGINAL points inside the fusion box, not just seed/support points.
        box_ids = np.flatnonzero(eligible & (xyz >= box_min).all(axis=1) & (xyz <= box_max).all(axis=1))
        regions.append(dict(id=region_id, accepted=not reasons, status='accepted' if not reasons else '+'.join(reasons),
            center=center, observed_min=low, observed_max=high, box_min=box_min, box_max=box_max,
            seed_xyz=config.lower + (np.asarray(seed) + 0.5) * config.spacing,
            support_count=len(members), cell_count=cell_count, box_count=len(box_ids),
            support_point_ids=members, box_point_ids=box_ids,
            peak_power=float(power[members].max()), mean_power=float(power[members].mean())))
    box_region = np.full(len(xyz), -1, np.int32)
    best_distance = np.full(len(xyz), np.inf)
    for region in regions:
        if not region['accepted']:
            continue
        inside = region['box_point_ids']
        distance = np.sum(((xyz[inside] - region['center']) / np.asarray(config.box_size)) ** 2, axis=1)
        better = distance < best_distance[inside]
        box_region[inside[better]] = region['id']
        best_distance[inside[better]] = distance[better]
    stats = dict(n_input=len(xyz), n_analysis=int(eligible.sum()), n_excluded=int((~eligible).sum()),
        n_support=int(support.sum()), n_seed_points=int(seed_points.sum()),
        n_assigned_support=int((point_region >= 0).sum()), n_regions=len(regions),
        n_accepted=sum(r['accepted'] for r in regions), n_box_points=int((box_region >= 0).sum()),
        box_fraction_all_p1=float(np.mean(box_region >= 0)) if len(xyz) else None,
        **seed_stats)
    return dict(config=asdict(config), regions=regions, stats=stats, thresholds=thresholds,
        analysis_mask=eligible, support_mask=support, seed_point_mask=seed_points,
        support_region_ids=point_region, box_region_ids=box_region,
        raw_bev=raw_power.max(axis=2), count_bev=counts.sum(axis=2),
        support_bev=support_grid.any(axis=2), seed_bev=seed_grid.any(axis=2),
        peak_cells=np.asarray(seeds, int).reshape(-1, 3))


def npz_payload(result, xyz, power, coords):
    """Array-only output, loadable with allow_pickle=False; ragged lists use CSR offsets."""
    regions = result['regions']
    payload = dict(xyz=np.asarray(xyz), power_channel2=np.asarray(power), coords_rae=np.asarray(coords),
        p1_npz_indices=np.arange(len(xyz), dtype=np.int64),
        analysis_mask=result['analysis_mask'], support_mask=result['support_mask'],
        seed_point_mask=result['seed_point_mask'], support_region_ids=result['support_region_ids'],
        box_region_ids=result['box_region_ids'],
        region_ids=np.array([r['id'] for r in regions], np.int32),
        accepted=np.array([r['accepted'] for r in regions], bool),
        status=np.array([r['status'] for r in regions], dtype='U64'))
    for key in ('center', 'observed_min', 'observed_max', 'box_min', 'box_max', 'seed_xyz'):
        payload[key] = np.asarray([r[key] for r in regions], float).reshape(-1, 3)
    for kind in ('support', 'box'):
        lists = [r[f'{kind}_point_ids'] for r in regions]
        payload[f'{kind}_point_offsets'] = np.r_[0, np.cumsum([len(v) for v in lists])].astype(np.int64)
        payload[f'{kind}_point_indices'] = np.concatenate(lists) if lists else np.zeros(0, np.int64)
    return payload
