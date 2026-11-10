"""Run the existing frozen ranker on compact query rows and full history."""
import numpy as np
import torch

from .box_candidates import box_candidates
from .learned_topk import masked_topk


@torch.no_grad()
def select_query_subset(selector, features, xyz, valid_mask, timestamps, point_valid_mask, query_ids):
    b, t, n, _ = features.shape
    if len(query_ids) != b:
        raise ValueError('One query index array per batch item required')
    if selector.cache is not None:
        # Reuse existing full-query caches without changing their input/key or
        # source namespace. Fail on miss rather than silently rank all queries.
        if selector.cache.mode != 'read':
            raise ValueError('Sparse query fusion supports existing Top-8 caches in read mode only')
        ids, good = selector(features, xyz, valid_mask, timestamps, point_valid_mask)
        return ([ids[i, q] for i, q in enumerate(query_ids)],
                [good[i, q] for i, q in enumerate(query_ids)])
    frames = np.asarray(valid_mask, bool)
    points = np.ones((b, t, n), bool) if point_valid_mask is None else np.asarray(point_valid_mask, bool)
    if frames.shape != (b, t) or points.shape != (b, t, n) or np.shape(xyz) != (b, t, n, 3):
        raise ValueError('Temporal geometry/mask shapes disagree')
    times = None if timestamps is None else np.asarray(timestamps, np.float64)
    if selector.method == 'learned' and (times is None or times.shape != (b, t) or not np.isfinite(times[frames]).all()):
        raise ValueError('Learned selection requires finite real timestamps for valid frames')
    if t-1 > len(selector.policy['x_half_by_history']):
        raise ValueError('Not enough history box widths')
    device, policy = features.device, selector.policy
    raw = features.detach().float()
    xyz_t = torch.as_tensor(xyz, dtype=torch.float64, device=device)
    indices, masks = [], []
    with torch.autocast(device_type=device.type, enabled=False):
        for batch, queries in enumerate(query_ids):
            chosen = np.zeros((len(queries), (t-1)*8), np.int64)
            chosen_valid = np.zeros_like(chosen, bool)
            q_t = torch.as_tensor(queries, device=device)
            history = raw[batch, :-1].reshape(-1, 8)
            history_xyz = xyz_t[batch, :-1].reshape(-1, 3)
            for frame in range(t-1):
                candidates = np.flatnonzero(points[batch, frame])
                if not frames[batch, frame] or not frames[batch, -1] or not len(candidates) or not len(queries):
                    continue
                pool, good = box_candidates(xyz[batch, -1, queries], xyz[batch, frame, candidates],
                    (policy['x_half_by_history'][t-2-frame], policy['y_half'], policy['z_half']), 64)
                pool = np.where(good, candidates[pool]+frame*n, 0)
                slots = slice(frame*8, (frame+1)*8)
                if selector.method == 'distance':
                    chosen[:, slots], chosen_valid[:, slots] = pool[:, :8], good[:, :8]
                    continue
                delta_t = float(times[batch, frame]-times[batch, -1])
                if delta_t >= 0:
                    raise ValueError('History must precede the current timestamp')
                for start in range(0, len(queries), selector.query_chunk_size):
                    end = min(start+selector.query_chunk_size, len(queries))
                    ix = torch.as_tensor(pool[start:end], device=device)
                    valid = torch.as_tensor(good[start:end], device=device)
                    if not good[start:end].any():
                        continue
                    relative = history_xyz[ix]-xyz_t[batch, -1, q_t[start:end], None]
                    score = selector.ranker(raw[batch, -1, q_t[start:end]], history[ix], relative,
                                           raw.new_full((end-start,), delta_t))
                    order, valid = masked_topk(score, valid, 8)
                    chosen[start:end, slots] = ix.gather(-1, order).cpu().numpy()
                    chosen_valid[start:end, slots] = valid.cpu().numpy()
            indices.append(chosen)
            masks.append(chosen_valid)
    return indices, masks
