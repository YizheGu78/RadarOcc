"""GT-free, vectorized Doppler query-budget diagnostics (not velocity recovery)."""
import numpy as np

def validate_doppler_table(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if (len(values) < 2 or not np.isfinite(values).all()
            or not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0))):
        raise ValueError('Doppler table must be finite and strictly monotonic')
    return values

UNCERTAIN, STATIC_CONSISTENT, NONSTATIC_EVIDENCE = 0, 1, 2


def wrap_velocity(value, period):
    if not np.isfinite(period) or period <= 0:
        raise ValueError('Positive finite Doppler period required')
    return (np.asarray(value) + period/2) % period - period/2


def fft_period(table):
    table = validate_doppler_table(table)
    delta = np.diff(table)
    if not np.allclose(delta, np.median(delta), rtol=1e-6, atol=1e-10):
        raise ValueError('Circular statistics require a uniform Doppler FFT table')
    return float(len(table)*abs(np.median(delta)))


def circular_peaks(features, table, max_std=.2, min_concentration=.8):
    """Use raw powers[0:3] and paired zero-based bins[3:6], in NPZ order.

    Crossing the velocity seam is handled circularly. This does not determine
    the unknown integer Doppler alias. A separated/multimodal spectrum abstains.
    """
    f = np.asarray(features, dtype=np.float64)
    table = validate_doppler_table(table)
    period = fft_period(table)
    if f.ndim != 2 or f.shape[1] != 8:
        raise ValueError('Expected raw Doppler8 [N,8]')
    if not 0 < max_std < period/2 or not 0 < min_concentration <= 1:
        raise ValueError('Require 0 < max_std < period/2 and 0 < concentration <= 1')
    power, bins = f[:, :3], f[:, 3:6]
    valid = (np.isfinite(power).all(1) & (power >= 0).all(1)
             & np.isfinite(bins).all(1) & (bins == np.floor(bins)).all(1)
             & (bins >= 0).all(1) & (bins < len(table)).all(1))
    safe_power = np.where(np.isfinite(power) & (power >= 0), power, 0.)
    scale = safe_power.max(axis=1)
    weight = safe_power/np.maximum(scale[:, None], np.finfo(float).tiny)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), np.finfo(float).tiny)
    ix = np.clip(np.nan_to_num(bins, nan=0, posinf=0, neginf=0), 0, len(table)-1).astype(int)
    velocity = table[ix]
    phase = velocity*(2*np.pi/period)
    c = (weight*np.cos(phase)).sum(1)
    s = (weight*np.sin(phase)).sum(1)
    concentration = np.hypot(c, s)
    folded = wrap_velocity(np.arctan2(s, c)*period/(2*np.pi), period)
    deviation = wrap_velocity(velocity-folded[:, None], period)
    std = np.sqrt((weight*deviation**2).sum(1))
    reliable = valid & (scale > 0) & (concentration >= min_concentration) & (std <= max_std)
    return dict(folded_mps=folded, circular_std_mps=std, concentration=concentration,
                reliable=reliable, valid_input=valid & (scale > 0), period_mps=period,
                seam_crossing=(np.ptp(velocity, axis=1) > period/2) & valid,
                linear_mean_mps=(weight*velocity).sum(1))


def static_consistency(xyz, peaks, ego_velocity, sign, max_target_radial_mps=30.):
    """Residual to the stationary hypothesis modulo the Doppler period.

    A small residual is compatible with stationary AND aliased moving targets.
    Alias candidate counts use an explicit assumed world radial-speed bound;
    they are diagnostics, not ground truth or a selected alias integer.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if sign not in (-1, 1) or xyz.shape != (len(peaks['reliable']), 3):
        raise ValueError('Matching XYZ and sign +1/-1 required')
    if not np.isfinite(max_target_radial_mps) or max_target_radial_mps <= 0:
        raise ValueError('Positive finite target radial-speed bound required')
    n = len(xyz)
    geometry_ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1e-8)
    reliable = peaks['reliable'] & geometry_ok
    if ego_velocity is None:
        return dict(residual_mps=np.full(n, np.nan), ego_projection_mps=np.full(n, np.nan),
                    static_folded_mps=np.full(n, np.nan), reliable=np.zeros(n, bool),
                    alias_candidate_count=np.zeros(n, np.int32), compensated_base_mps=np.full(n, np.nan),
                    period_mps=peaks['period_mps'])
    velocity = np.asarray(ego_velocity, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError('Ego velocity must be a finite radar-frame vector')
    los = xyz/np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-8)
    projection = los @ velocity
    period = peaks['period_mps']
    base = sign*peaks['folded_mps'] + projection
    residual = wrap_velocity(base, period)
    lower = np.ceil((-max_target_radial_mps-base)/period - 1e-12)
    upper = np.floor((max_target_radial_mps-base)/period + 1e-12)
    count = np.maximum(upper-lower+1, 0)
    count = np.where(geometry_ok, count, 0).astype(np.int32)
    reliable &= count > 0
    return dict(residual_mps=residual, ego_projection_mps=projection,
                static_folded_mps=wrap_velocity(-projection, period), reliable=reliable,
                alias_candidate_count=count, compensated_base_mps=base, period_mps=period)


def range_order(coords, ids):
    order = np.lexsort((ids, coords[ids, 2], coords[ids, 1], coords[ids, 0]))
    ids = ids[order]
    ranges, starts, sizes = np.unique(coords[ids, 0], return_index=True, return_counts=True)
    return ids, ranges, starts, sizes


def spatial_budget(coords, count, eligible=None):
    """Exact budget, proportionally allocated by range, then spread over angles.

    This is deterministic stratified sampling, not GT-aware or power-ranked.
    Coordinates/returned mask retain the original NPZ order.
    """
    coords = np.asarray(coords)
    eligible = np.ones(len(coords), bool) if eligible is None else np.asarray(eligible, bool)
    if coords.shape != (len(eligible), 3) or count < 0 or count > int(eligible.sum()):
        raise ValueError('Matching R/A/E coordinates and a feasible integer budget required')
    keep = np.zeros(len(coords), bool)
    ids = np.flatnonzero(eligible)
    if not len(ids) or count == 0:
        return keep
    ids, ranges, starts, sizes = range_order(coords, ids)
    exact = sizes*(count/len(ids))
    quotas = np.floor(exact).astype(int)
    remainder = count-int(quotas.sum())
    order = np.lexsort((ranges, -(exact-quotas)))
    quotas[order[:remainder]] += 1
    for start, size, quota in zip(starts, sizes, quotas):
        if not quota:
            continue
        positions = np.floor((np.arange(quota)+.5)*size/quota).astype(int)
        keep[ids[start+positions]] = True
    return keep


def query_masks(coords, consistency, threshold, static_keep_ratio=.25):
    """Compare an explicit heuristic with a guarded, speed-bound-dependent policy.

    Unreliable peaks/ego always keep their queries. Heuristic pruning operates
    on stationary-consistent tokens despite unresolved aliases. The guarded
    variant only prunes if every plausible radial alias is within threshold.
    """
    if not np.isfinite(threshold) or not 0 < threshold < consistency['period_mps']/2 or not 0 <= static_keep_ratio <= 1:
        raise ValueError('Require 0 < threshold < period/2 and keep ratio in [0,1]')
    residual, reliable = consistency['residual_mps'], consistency['reliable']
    static = reliable & (np.abs(residual) <= threshold)
    state = np.full(len(static), UNCERTAIN, np.uint8)
    state[static] = STATIC_CONSISTENT
    state[reliable & ~static] = NONSTATIC_EVIDENCE

    def keep_with_reserve(removable):
        # Per-range ceil ensures a nonzero reserve in every static range when ratio > 0.
        keep = ~removable
        ids, _, starts, sizes = range_order(coords, np.flatnonzero(removable))
        for start, size in zip(starts, sizes):
            reserve = int(np.ceil(size*static_keep_ratio))
            if reserve:
                positions = np.floor((np.arange(reserve)+.5)*size/reserve).astype(int)
                keep[ids[start+positions]] = True
        return keep

    heuristic = keep_with_reserve(static)
    # threshold < period/2 guarantees at most one near-zero alias (validated by CLI).
    ambiguous = static & (consistency['alias_candidate_count'] > 1)
    guarded = keep_with_reserve(static & ~ambiguous)
    return dict(state=state, static_consistent=static, alias_ambiguous_static=ambiguous,
                doppler_heuristic=heuristic, alias_guarded=guarded,
                spatial_same_budget=spatial_budget(coords, int(heuristic.sum())))
