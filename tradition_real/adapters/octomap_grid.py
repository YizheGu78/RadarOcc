"""RadarOcc grid adapter; coordinate translation/export, never an OctoMap rule."""
import numpy as np
from ..octomap import OcTree


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def voxel_indices(xyz, config):
    # OctoMap receives point3d (float32) after moving the grid origin. Match
    # coordToKey's DOUBLE resolution factor applied to those float coordinates.
    shifted = (np.asarray(xyz) - np.asarray(config.min_xyz)).astype(np.float32)
    index = np.floor(shifted.astype(np.float64) * (1.0 / config.resolution)).astype(np.int64)
    valid = np.all((index >= 0) & (index < np.asarray(config.shape_xyz)), axis=1)
    return index, valid


class OctomapGrid:
    def __init__(self, config):
        self.config = config
        self.tree = OcTree(config.resolution)
        self.tree.setProbHit(config.p_hit)
        self.tree.setProbMiss(config.p_miss)
        self.tree.setOccupancyThres(config.occupancy_threshold)
        self.tree.setClampingThresMin(config.clamping_min)
        self.tree.setClampingThresMax(config.clamping_max)

    def insert(self, points, origin):
        cfg = self.config
        # A rigid change of coordinates aligns all RadarOcc voxel boundaries
        # (including min_z=-2.6) with OctoMap's zero-anchored 0.4 m lattice.
        shifted = (np.asarray(points) - np.asarray(cfg.min_xyz)).astype(np.float32)
        shifted_origin = (np.asarray(origin) - np.asarray(cfg.min_xyz)).astype(np.float32)
        if not np.isfinite(shifted).all() or not np.isfinite(shifted_origin).all():
            raise ValueError('OctoMap input must be finite float32 XYZ')
        # No ROI/height prefilter: out-of-ROI endpoints can clear in-ROI space.
        # Reject unrepresentable data instead of silently losing observations.
        if self.tree.coordToKeyChecked(shifted_origin) is None or any(self.tree.coordToKeyChecked(p) is None for p in shifted):
            raise ValueError('OctoMap input outside depth-16 key range')
        self.tree.insertPointCloud(shifted, shifted_origin, cfg.max_range,
                                   cfg.lazy_eval, cfg.discretize)

    def export(self):
        cfg, tree = self.config, self.tree
        if cfg.lazy_eval:
            tree.updateInnerOccupancy()
            # Upstream lazy insertion does not prune. Keep that behavior.
        probability = np.full(cfg.shape_xyz, .5, np.float64)
        observed = np.zeros(cfg.shape_xyz, bool)
        occupied = np.zeros(cfg.shape_xyz, bool)
        # Expand each pruned leaf's extent into the fixed-resolution output.
        # This is equivalent to full-depth search at every grid cell, without
        # 229376 separate Python root-to-leaf traversals per frame.
        for key_lower, span, node in tree.iter_leaves():
            start = np.asarray(key_lower, np.int64) - tree.tree_max_val
            lower = np.maximum(start, 0)
            upper = np.minimum(start + span, cfg.shape_xyz)
            if np.any(lower >= upper):
                continue
            region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            probability[region] = node.getOccupancy()
            observed[region] = True
            occupied[region] = tree.isNodeOccupied(node)
        free = observed & ~occupied
        # Diagnostic projection only, NOT a second 2D sensor model/fusion.
        # Unknown columns stay .5; otherwise use maximum observed occupancy.
        bev = np.max(np.where(observed, probability, -np.inf), axis=2)
        bev[~np.any(observed, axis=2)] = .5
        return probability, observed, occupied, free, bev
