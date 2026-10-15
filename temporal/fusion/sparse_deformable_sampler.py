"""Bounded CPU candidate search and differentiable torch sparse interpolation."""
import numpy as np
from scipy.spatial import cKDTree
import torch

from ..core.range_anisotropic_geometry import normalized_displacement, sampling_anchors


def _select_chunk(xyz, queries, neighborhoods, basis, scales, anchors,
                  candidate_pool_size, offset_bound, interpolation_cutoff):
    """Batch local geometry and anchor distances over ragged neighborhoods.

    Only the small nearest-support fill loops over queries. No [Q,max_neighbors,
    K,3] padding and no per-query coordinate transform/anchor-distance allocation.
    """
    counts = np.fromiter((len(values) for values in neighborhoods), dtype=np.int64)
    if not counts.sum():
        return None
    rows = np.repeat(np.arange(len(queries)), counts)
    original = np.concatenate(neighborhoods).astype(np.int64, copy=False)
    local = normalized_displacement(xyz[original], queries[rows], basis[rows], scales[rows])
    distance_to_cube = np.maximum(np.abs(local) - offset_bound, 0)
    keep = (distance_to_cube ** 2).sum(-1) <= interpolation_cutoff ** 2
    rows, original, local = rows[keep], original[keep], local[keep]
    if not len(original):
        return None
    counts = np.bincount(rows, minlength=len(queries))
    nonempty = np.flatnonzero(counts)
    starts = np.r_[0, counts.cumsum()][nonempty]
    groups = np.searchsorted(nonempty, rows)
    norm2 = (local ** 2).sum(-1)
    selected = np.zeros(len(original), dtype=bool)
    all_rows = np.arange(len(original))[:, None]
    # Four anchors at a time bounds CPU temporary storage even for dense bins.
    for start in range(0, len(anchors), 4):
        anchor = anchors[start:start + 4]
        distance = norm2[:, None] + (anchor ** 2).sum(-1)[None] - 2 * local @ anchor.T
        minimum = np.minimum.reduceat(distance, starts, axis=0)
        winners = np.minimum.reduceat(np.where(distance == minimum[groups],
                                               all_rows, len(original)), starts, axis=0)
        selected[winners.ravel()] = True
    for start, count in zip(starts, counts[nonempty]):
        stop = start + count
        chosen = int(selected[start:stop].sum())
        fill = min(candidate_pool_size, count) - chosen
        if fill:
            remaining = np.flatnonzero(~selected[start:stop]) + start
            closest = np.argpartition(norm2[remaining], fill - 1)[:fill]
            selected[remaining[closest]] = True
    return rows[selected], original[selected], local[selected]


def anisotropic_candidates(xyz, valid_mask, geometry, candidate_pool_size=24,
                           offset_bound=0.5, interpolation_cutoff=2.0,
                           search_chunk_size=256, point_valid_mask=None):
    """Return IDs/masks [B,N,T-1,K] and normalized local positions [...,3].

    Search the entire dynamic coarse ball, filter by distance to the allowed
    offset cube, then choose points close to spatially spread anchors. A fixed
    nearest-K Euclidean preselection would prematurely discard far-range support.
    Ragged coarse neighborhoods exist only for one CPU query chunk at a time.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 4 or xyz.shape[-1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('aligned_xyz must be finite [B,T,N,3]')
    b, t, n, _ = xyz.shape
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != (b, t) or not valid_mask[:, -1].all():
        raise ValueError('valid_mask must be [B,T] with valid current slots')
    point_mask = (np.ones((b, t, n), dtype=bool) if point_valid_mask is None
                  else np.asarray(point_valid_mask, dtype=bool))
    if point_mask.shape != (b, t, n):
        raise ValueError('point_valid_mask must be [B,T,N]')
    shape = (b, n, t - 1, candidate_pool_size)
    ids, good = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool)
    local = np.zeros(shape + (3,), dtype=np.float32)
    anchors = sampling_anchors(candidate_pool_size, offset_bound)
    basis, scales = geometry.query_geometry(xyz[:, -1])
    radii = (offset_bound * np.linalg.norm(scales, axis=-1)
             + interpolation_cutoff * scales.max(axis=-1))
    for batch in range(b):
        queries = np.flatnonzero(point_mask[batch, -1])
        for frame in range(t - 1):
            candidates = np.flatnonzero(point_mask[batch, frame])
            if not valid_mask[batch, frame] or not candidates.size:
                continue
            tree = cKDTree(xyz[batch, frame, candidates])
            for start in range(0, len(queries), search_chunk_size):
                chunk = queries[start:start + search_chunk_size]
                neighborhoods = tree.query_ball_point(xyz[batch, -1, chunk],
                                                       radii[batch, chunk], workers=1)
                selection = _select_chunk(xyz[batch, frame, candidates],
                    xyz[batch, -1, chunk], neighborhoods, basis[batch, chunk],
                    scales[batch, chunk], anchors, candidate_pool_size,
                    offset_bound, interpolation_cutoff)
                if selection is None:
                    continue
                rows, original, coords = selection
                counts = np.bincount(rows, minlength=len(chunk))
                starts = np.r_[0, counts.cumsum()][:-1]
                slots = np.arange(len(rows)) - starts[rows]
                target = chunk[rows]
                ids[batch, target, frame, slots] = candidates[original] + frame * n
                good[batch, target, frame, slots] = True
                local[batch, target, frame, slots] = coords
    return ids, good, local


def masked_softmax(logits, mask, dim):
    """Finite all-masked rows, with exactly zero mass on invalid slots."""
    weights = logits.masked_fill(~mask, -1e9).softmax(dim=dim) * mask
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)


def sparse_sample(keys, values, candidate_local, offsets, candidate_mask,
                  temperature=0.5, interpolation_cutoff=2.0):
    """Gaussian interpolation in normalized radar coordinates, entirely torch.

    keys/values [Q,F,K,H,D], local [Q,F,K,3], offsets [Q,F,H,M,3].
    Returns sampled keys/values [Q,F,H,M,D] and validity [Q,F,H,M].
    Gradients flow through interpolation weights into continuous offsets.
    """
    distance2 = (candidate_local.square().sum(-1)[:, :, None, None, :]
                 + offsets.square().sum(-1)[..., None]
                 - 2 * torch.einsum('qfkx,qfhmx->qfhmk', candidate_local, offsets))
    distance2 = distance2.clamp_min(0)
    mask = candidate_mask[:, :, None, None, :] & (distance2 <= interpolation_cutoff ** 2)
    weights = masked_softmax(-distance2 / (2 * temperature), mask, dim=-1)
    sampled_key = torch.einsum('qfhmk,qfkhd->qfhmd', weights, keys)
    sampled_value = torch.einsum('qfhmk,qfkhd->qfhmd', weights, values)
    return sampled_key, sampled_value, mask.any(dim=-1)
