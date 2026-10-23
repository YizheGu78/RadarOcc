"""GT-free cuboid candidates shared by the offline trainer and RadarOcc."""
import numpy as np
from scipy.spatial import cKDTree


def box_candidates(queries, history, half_widths, k=64, query_chunk=256):
    """Nearest K in a closed cuboid, with original history index tie breaks.

    Return local IDs and a validity mask [Q,K]. Invalid IDs are safe zeros;
    they are placeholders, never copies of real candidates. No dense QxH
    distance matrix or GT-dependent filtering is used.
    """
    queries, history = np.asarray(queries), np.asarray(history)
    half = np.asarray(half_widths, dtype=np.float64)
    if (queries.ndim != 2 or queries.shape[1] != 3 or history.ndim != 2
            or history.shape[1] != 3 or half.shape != (3,)
            or not np.isfinite(half).all() or np.any(half <= 0)
            or not np.isfinite(queries).all() or not np.isfinite(history).all()
            or k <= 0 or query_chunk <= 0):
        raise ValueError('Finite XYZ, positive cuboid half widths, K and chunk required')
    ids = np.zeros((len(queries), k), dtype=np.int64)
    valid = np.zeros_like(ids, dtype=bool)
    if not len(history) or not len(queries):
        return ids, valid
    tree = cKDTree(history / half)
    for start in range(0, len(queries), query_chunk):
        pools = tree.query_ball_point(queries[start:start+query_chunk] / half,
                                     1 + 1e-9, p=np.inf, workers=1)
        for offset, pool in enumerate(pools):
            ix = np.asarray(pool, dtype=np.int64)
            relative = history[ix] - queries[start+offset]
            keep = np.all(np.abs(relative) <= half + 1e-9, axis=1)
            ix, relative = ix[keep], relative[keep]
            order = np.lexsort((ix, np.einsum('ij,ij->i', relative, relative)))[:k]
            ix = ix[order]
            ids[start+offset, :len(ix)] = ix
            valid[start+offset, :len(ix)] = True
    return ids, valid


def temporal_box_candidates(xyz, valid_mask, x_half_by_history=(4., 6., 8.),
                            y_half=1.5, z_half=1., k=64, point_valid_mask=None):
    """Return [B,N,(T-1)*K] flattened history IDs, oldest frame first."""
    xyz = np.asarray(xyz)
    b, t, n, c = xyz.shape
    frames = np.asarray(valid_mask, dtype=bool)
    if c != 3 or frames.shape != (b, t) or t-1 > len(x_half_by_history):
        raise ValueError('XYZ/frame mask dimensions or number of history widths disagree')
    points = (np.ones((b, t, n), dtype=bool) if point_valid_mask is None
              else np.asarray(point_valid_mask, dtype=bool))
    if points.shape != (b, t, n):
        raise ValueError('point_valid_mask must be [B,T,N]')
    ids = np.zeros((b, n, (t-1)*k), dtype=np.int64)
    good = np.zeros_like(ids, dtype=bool)
    for batch in range(b):
        query_ids = np.flatnonzero(points[batch, -1] & frames[batch, -1])
        for frame in range(t-1):
            if not frames[batch, frame]:
                continue
            candidate_ids = np.flatnonzero(points[batch, frame])
            if not len(candidate_ids):
                continue
            slot = t-1-frame
            local, valid = box_candidates(xyz[batch, -1, query_ids],
                xyz[batch, frame, candidate_ids],
                (x_half_by_history[slot-1], y_half, z_half), k)
            target = slice(frame*k, (frame+1)*k)
            ids[batch, query_ids, target] = np.where(valid, candidate_ids[local]+frame*n, 0)
            good[batch, query_ids, target] = valid
    return ids, good
