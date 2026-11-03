"""GT-free P1 reflection proposals. Geometry only; no detector or learned features.

Point IDs always index the original NPZ's first 175*250 returns, NOT a GPU's
power-sorted P1 tensor. BEV is a temporary raster, not a network feature map.
"""
from dataclasses import asdict, dataclass
import heapq
import itertools

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


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
    merge_regions: bool = True
    merge_space: str = 'xyz'
    xy_box_gap: tuple = (0.5, 0.5)
    merge_gap: tuple = (1.5, 1.0, 1.0)
    merge_min_contact_cells: int = 2
    max_box_size: tuple = (12.0, 4.0, 4.0)
    box_padding: tuple = (0.5, 0.25, 0.5)
    min_points: int = 6
    min_cells: int = 3
    max_aspect: float = 8.0

    def validate(self):
        if self.merge_space not in ('xyz', 'xy'):
            raise ValueError('merge_space must be xyz or xy')
        if len(self.xy_box_gap) != 2 or not np.isfinite(self.xy_box_gap).all() or min(self.xy_box_gap) < 0:
            raise ValueError('xy_box_gap must contain two finite nonnegative XY values')
        positive = (self.xy_resolution, self.z_resolution, self.range_band_m,
                    self.peak_separation_m, self.max_aspect,
                    *self.grow_half_size, *self.box_size, *self.merge_gap, *self.max_box_size)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError('Grid, band, peak, size and aspect settings must be finite and positive')
        for limits in (self.x_limits, self.y_limits, self.z_limits):
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Each analysis limit must be a finite increasing pair')
        if any(len(v) != 3 for v in (self.box_size, self.grow_half_size, self.merge_gap,
                                     self.max_box_size, self.box_padding)):
            raise ValueError('Size, gap and padding settings must contain XYZ values')
        if not np.isfinite(self.box_padding).all() or min(self.box_padding) < 0:
            raise ValueError('Box padding must be finite and nonnegative')
        if self.merge_regions and (np.any(np.asarray(self.max_box_size) < self.box_size)
                or np.any(np.asarray(self.max_box_size) <= 2 * np.asarray(self.box_padding))):
            raise ValueError('Maximum box size must cover minimum size and leave room for padding')
        if not 0 < self.support_percentile < self.seed_percentile < 100:
            raise ValueError('Require 0 < support_percentile < seed_percentile < 100')
        if not np.isfinite([self.seed_mad, self.support_mad]).all() or not 0 <= self.support_mad <= self.seed_mad:
            raise ValueError('Require finite 0 <= support_mad <= seed_mad')
        if min(self.max_seed_count, self.min_points, self.min_cells, self.merge_min_contact_cells) < 1:
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


def merge_support_regions(xyz, support_ids, support_cells, support_grid, owner, seeds, config):
    """Join fine basins and nearby unseeded support fragments, with bounded unions.

    Search is over occupied support CELLS once per frame, never over every P1
    query. Empty cells are not synthesized. A link needs contact on both sides
    near the closest boundary, so a thin weak bridge does not join two bodies.
    Bounds are rechecked for the entire union, preventing unlimited chaining.
    This is a geometry heuristic, not proof that a group is one physical object.
    """
    seed_count = len(seeds)
    stats = dict(n_orphan_components=0, merge_candidate_links=0,
                 merge_links_rejected_contact=0, merge_links_rejected_size=0)
    if not seed_count:
        return np.full(len(xyz), -1, np.int32), [], [], stats
    orphan_labels, orphan_count = ndimage.label(support_grid & (owner == 0),
                                               structure=np.ones((3, 3, 3), bool))
    stats['n_orphan_components'] = orphan_count
    fragments = owner.copy()
    missing = orphan_labels > 0
    fragments[missing] = seed_count + orphan_labels[missing]
    active = np.argwhere(support_grid)
    labels = fragments[tuple(active.T)] - 1
    point_labels = fragments[tuple(support_cells.T)] - 1
    count = seed_count + orphan_count
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, point_labels, xyz[support_ids])
    np.maximum.at(high, point_labels, xyz[support_ids])

    scaled = active * config.spacing / np.asarray(config.merge_gap)
    pairs = cKDTree(scaled).query_pairs(1.0 + 1e-10, p=np.inf, output_type='ndarray')
    if len(pairs):
        pairs = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
    links = []
    if len(pairs):
        a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
        codes = np.minimum(a, b).astype(np.int64) * count + np.maximum(a, b)
        unique, inverse = np.unique(codes, return_inverse=True)
        costs = np.linalg.norm(scaled[pairs[:, 0]] - scaled[pairs[:, 1]], axis=1)
        minimum = np.full(len(unique), np.inf)
        np.minimum.at(minimum, inverse, costs)
        # Count distinct boundary cells only at the nearest contact distance;
        # counting every cell in the radius would inflate a one-cell bridge.
        closest = costs <= minimum[inverse] + 1e-8
        contact_pairs, contact_inverse = pairs[closest], inverse[closest]
        left = np.where(labels[contact_pairs[:, 0]] < labels[contact_pairs[:, 1]],
                        contact_pairs[:, 0], contact_pairs[:, 1])
        right = np.where(labels[contact_pairs[:, 0]] < labels[contact_pairs[:, 1]],
                         contact_pairs[:, 1], contact_pairs[:, 0])
        contacts = []
        for side in (left, right):
            distinct = np.unique(contact_inverse.astype(np.int64) * len(active) + side)
            contacts.append(np.bincount(distinct // len(active), minlength=len(unique)))
        links = sorted((float(minimum[i]), int(code // count), int(code % count),
                        int(min(contacts[0][i], contacts[1][i]))) for i, code in enumerate(unique))
    stats['merge_candidate_links'] = len(links)

    def root(parent, node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    # Inspect full connected chains before bounded splitting. Otherwise a long
    # thin barrier could look like many independent accepted vehicle-sized boxes.
    connected = np.arange(count)
    for _, a, b, _ in links:
        ra, rb = root(connected, a), root(connected, b)
        connected[max(ra, rb)] = min(ra, rb)
    connected_roots = np.array([root(connected, i) for i in range(count)])
    chain_low, chain_high = np.full_like(low, np.inf), np.full_like(high, -np.inf)
    np.minimum.at(chain_low, connected_roots, low)
    np.maximum.at(chain_high, connected_roots, high)
    chain_span = chain_high[connected_roots] - chain_low[connected_roots]
    horizontal = np.maximum(chain_span[:, :2], config.xy_resolution)
    long_chain = ((horizontal.max(axis=1) / horizontal.min(axis=1) > config.max_aspect)
                  & (chain_span[:, :2] > np.asarray(config.max_box_size)[:2]).any(axis=1))

    parent = np.arange(count)
    merged_low, merged_high = low.copy(), high.copy()
    max_span = np.asarray(config.max_box_size) - 2 * np.asarray(config.box_padding)
    for _, a, b, contact in links:
        ra, rb = root(parent, a), root(parent, b)
        if ra == rb:
            continue
        if contact < config.merge_min_contact_cells:
            stats['merge_links_rejected_contact'] += 1
            continue
        union_low = np.minimum(merged_low[ra], merged_low[rb])
        union_high = np.maximum(merged_high[ra], merged_high[rb])
        if np.any(union_high - union_low > max_span + 1e-8):
            stats['merge_links_rejected_size'] += 1
            continue
        keep, discard = min(ra, rb), max(ra, rb)
        parent[discard] = keep
        merged_low[keep], merged_high[keep] = union_low, union_high
    roots = np.array([root(parent, i) for i in range(count)])
    groups = {}
    for seed_id in range(seed_count):
        groups.setdefault(int(roots[seed_id]), []).append(seed_id)
    sources = sorted(groups.values(), key=lambda values: values[0])
    root_to_region = np.full(count, -1, np.int32)
    for region_id, source in enumerate(sources):
        root_to_region[roots[source[0]]] = region_id
    point_region = np.full(len(xyz), -1, np.int32)
    point_region[support_ids] = root_to_region[roots[point_labels]]
    flags = [bool(long_chain[source[0]]) for source in sources]
    return point_region, sources, flags, stats


def merge_xy_boxes(xyz, fine_regions, fine_labels, config):
    """Merge overlapping/nearby XY box footprints, ignoring height entirely.

    Start from ALL fine basins, before acceptance filtering. Link by the edge
    gap of original fixed-box XY footprints, not support-cell contact or Z.
    Recheck the whole union's support XY span plus padding after each merge.
    Original point IDs and Z values are untouched; no support is synthesized.
    """
    count = len(fine_regions)
    stats = dict(n_orphan_components=0, merge_candidate_links=0,
                 merge_links_rejected_contact=0, merge_links_rejected_size=0)
    if not count:
        return fine_labels.copy(), [], [], stats
    lows = np.array([r['observed_min'][:2] for r in fine_regions])
    highs = np.array([r['observed_max'][:2] for r in fine_regions])
    links = []
    for a in range(count):
        ra = fine_regions[a]
        for b in range(a + 1, count):
            rb = fine_regions[b]
            gap = np.maximum(np.maximum(ra['box_min'][:2] - rb['box_max'][:2],
                                        rb['box_min'][:2] - ra['box_max'][:2]), 0)
            if np.all(gap <= np.asarray(config.xy_box_gap) + 1e-8):
                center_distance = np.linalg.norm(ra['center'][:2] - rb['center'][:2])
                links.append((float(np.linalg.norm(gap)), float(center_distance), a, b))
    links.sort()
    stats['merge_candidate_links'] = len(links)

    def root(parent, node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    connected = np.arange(count)
    for _, _, a, b in links:
        a, b = root(connected, a), root(connected, b)
        connected[max(a, b)] = min(a, b)
    connected_roots = np.array([root(connected, i) for i in range(count)])
    chain_low, chain_high = np.full_like(lows, np.inf), np.full_like(highs, -np.inf)
    np.minimum.at(chain_low, connected_roots, lows)
    np.maximum.at(chain_high, connected_roots, highs)
    chain_span = chain_high[connected_roots] - chain_low[connected_roots]
    extent = np.maximum(chain_span, config.xy_resolution)
    long_chain = ((extent.max(axis=1) / extent.min(axis=1) > config.max_aspect)
                  & (chain_span > np.asarray(config.max_box_size)[:2]).any(axis=1))

    parent = np.arange(count)
    merged_low, merged_high = lows.copy(), highs.copy()
    max_span = np.asarray(config.max_box_size)[:2] - 2 * np.asarray(config.box_padding)[:2]
    for _, _, a, b in links:
        a, b = root(parent, a), root(parent, b)
        if a == b:
            continue
        low, high = np.minimum(merged_low[a], merged_low[b]), np.maximum(merged_high[a], merged_high[b])
        if np.any(high - low > max_span + 1e-8):
            stats['merge_links_rejected_size'] += 1
            continue
        keep, discard = min(a, b), max(a, b)
        parent[discard] = keep
        merged_low[keep], merged_high[keep] = low, high
    roots = np.array([root(parent, i) for i in range(count)])
    sources = [np.flatnonzero(roots == r).tolist() for r in np.unique(roots)]
    mapping = np.full(count, -1, np.int32)
    for region_id, source in enumerate(sources):
        mapping[source] = region_id
    point_region = np.full(len(xyz), -1, np.int32)
    assigned = fine_labels >= 0
    point_region[assigned] = mapping[fine_labels[assigned]]
    return point_region, sources, [bool(long_chain[s[0]]) for s in sources], stats


def make_regions(xyz, power, eligible, point_region, seeds, sources, config,
                 adaptive=False, long_flags=None):
    regions = []
    for region_id, source in enumerate(sources):
        members = np.flatnonzero(point_region == region_id)
        points = xyz[members]
        low, high = points.min(axis=0), points.max(axis=0)
        robust_low, robust_high = np.percentile(points, [10, 90], axis=0)
        center = (robust_low + robust_high) / 2
        span = high - low
        occupied = np.floor((points - config.lower) / config.spacing).astype(int)
        cell_count = len(np.unique(occupied, axis=0))
        reasons = []
        if len(members) < config.min_points or cell_count < config.min_cells:
            reasons.append('small_support')
        xy_span = np.maximum(span[:2], config.xy_resolution)
        if xy_span.max() / xy_span.min() > config.max_aspect:
            reasons.append('elongated_support')
        size = np.asarray(config.box_size)
        if adaptive:
            padding = np.asarray(config.box_padding)
            size = np.maximum(size, span + 2 * padding)
            # Keep the robust center where possible, but enclose ALL support.
            center = np.clip(center, high + padding - size / 2, low - padding + size / 2)
            # XY grouping must not be rejected again solely because Z is large.
            dims = 2 if config.merge_space == 'xy' else 3
            if np.any(size[:dims] > np.asarray(config.max_box_size)[:dims] + 1e-8):
                reasons.append('oversized_support')
            if long_flags[region_id]:
                reasons.append('long_structure_candidate')
        box_min, box_max = center - size / 2, center + size / 2
        tolerance = 1e-8 if adaptive else 0.0
        box_ids = np.flatnonzero(eligible & (xyz >= box_min - tolerance).all(axis=1)
                                & (xyz <= box_max + tolerance).all(axis=1))
        outside = ~((points >= box_min - 1e-8).all(axis=1) & (points <= box_max + 1e-8).all(axis=1))
        regions.append(dict(id=region_id, accepted=not reasons, status='accepted' if not reasons else '+'.join(reasons),
            center=center, observed_min=low, observed_max=high, box_min=box_min, box_max=box_max,
            box_size=size, source_seed_ids=source, source_seed_count=len(source),
            seed_xyz=config.lower + (np.asarray(seeds[source[0]]) + .5) * config.spacing,
            support_count=len(members), cell_count=cell_count, box_count=len(box_ids),
            support_outside_box_count=int(outside.sum()),
            support_point_ids=members, box_point_ids=box_ids,
            peak_power=float(power[members].max()), mean_power=float(power[members].mean())))
    return regions


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
    _, seeds, seed_stats = seed_markers(strength, seed_grid, config)
    owner = grow_regions(support_grid, seeds, config)
    point_region = np.full(len(xyz), -1, np.int32)
    grid_owner = owner[tuple(cells.T)] if len(ids) else np.zeros(0, int)
    assigned = support[ids] & (grid_owner > 0)
    point_region[ids[assigned]] = grid_owner[assigned] - 1
    fine_labels = point_region.copy()
    sources = [[i] for i in range(len(seeds))]
    fine_regions = make_regions(xyz, power, eligible, fine_labels, seeds, sources, config)
    merge_stats = dict(n_orphan_components=0, merge_candidate_links=0,
                       merge_links_rejected_contact=0, merge_links_rejected_size=0)
    if config.merge_regions:
        if config.merge_space == 'xy':
            point_region, sources, long_flags, merge_stats = merge_xy_boxes(
                xyz, fine_regions, fine_labels, config)
        else:
            support_ids = ids[support[ids]]
            point_region, sources, long_flags, merge_stats = merge_support_regions(
                xyz, support_ids, cells[support[ids]], support_grid, owner, seeds, config)
        regions = make_regions(xyz, power, eligible, point_region, seeds, sources, config,
                               adaptive=True, long_flags=long_flags)
    else:
        regions = fine_regions
    box_region = np.full(len(xyz), -1, np.int32)
    best_distance = np.full(len(xyz), np.inf)
    for region in regions:
        if not region['accepted']:
            continue
        inside = region['box_point_ids']
        distance = np.sum(((xyz[inside] - region['center']) / region['box_size']) ** 2, axis=1)
        better = distance < best_distance[inside]
        box_region[inside[better]] = region['id']
        best_distance[inside[better]] = distance[better]
    stats = dict(n_input=len(xyz), n_analysis=int(eligible.sum()), n_excluded=int((~eligible).sum()),
        n_support=int(support.sum()), n_seed_points=int(seed_points.sum()),
        n_assigned_support=int((point_region >= 0).sum()), n_regions=len(regions),
        n_fine_regions=len(fine_regions), n_fine_accepted=sum(r['accepted'] for r in fine_regions),
        n_seed_regions_merged=len(fine_regions) - len(regions),
        n_recovered_support=int(((point_region >= 0) & (fine_labels < 0)).sum()),
        n_unassigned_support=int((support & (point_region < 0)).sum()),
        n_long_structure_candidates=sum('long_structure_candidate' in r['status'] for r in regions),
        n_accepted=sum(r['accepted'] for r in regions), n_box_points=int((box_region >= 0).sum()),
        box_fraction_all_p1=float(np.mean(box_region >= 0)) if len(xyz) else None,
        **seed_stats, **merge_stats)
    return dict(config=asdict(config), regions=regions, fine_regions=fine_regions,
        fine_support_region_ids=fine_labels, stats=stats, thresholds=thresholds,
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
        fine_support_region_ids=result['fine_support_region_ids'],
        box_region_ids=result['box_region_ids'],
        region_ids=np.array([r['id'] for r in regions], np.int32),
        accepted=np.array([r['accepted'] for r in regions], bool),
        status=np.array([r['status'] for r in regions], dtype='U128'))
    for key in ('center', 'observed_min', 'observed_max', 'box_min', 'box_max', 'box_size', 'seed_xyz'):
        payload[key] = np.asarray([r[key] for r in regions], float).reshape(-1, 3)
    for kind in ('support', 'box'):
        lists = [r[f'{kind}_point_ids'] for r in regions]
        payload[f'{kind}_point_offsets'] = np.r_[0, np.cumsum([len(v) for v in lists])].astype(np.int64)
        payload[f'{kind}_point_indices'] = np.concatenate(lists) if lists else np.zeros(0, np.int64)
    sources = [r['source_seed_ids'] for r in regions]
    payload['source_seed_offsets'] = np.r_[0, np.cumsum([len(v) for v in sources])].astype(np.int64)
    payload['source_seed_indices'] = np.array([i for values in sources for i in values], np.int32)
    payload['fine_region_ids'] = np.array([r['id'] for r in result['fine_regions']], np.int32)
    for key in ('center', 'box_min', 'box_max'):
        payload[f'fine_{key}'] = np.asarray([r[key] for r in result['fine_regions']], float).reshape(-1, 3)
    return payload

