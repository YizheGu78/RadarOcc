"""Explicit 2D-to-height-layer adaptation, NOT an Autoware algorithm.

Each endpoint contributes to its own height layer. Rays are horizontal in
that layer, not physical 3D rays. No full-height occupied extrusion is used.
Coordinates passed to the unmodified 2D port are measured in grid cells.
"""
import numpy as np
from trodition_real.autoware.costmap import OccupancyGridMap, NO_INFORMATION


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def voxel_indices(xyz, config):
    # Match float32 PointCloud2 coordinate precision used by the 2D port.
    relative = ((np.asarray(xyz) - np.asarray(config.min_xyz)) / config.resolution).astype(np.float32)
    index = np.floor(relative).astype(np.int64)
    valid = np.all((index >= 0) & (index < np.asarray(config.shape_xyz)), axis=1)
    return index, valid


def evidence_maps(xyz, sensor_origin, config):
    relative = ((np.asarray(xyz) - config.min_xyz) / config.resolution).astype(np.float32)
    origin = (np.asarray(sensor_origin) - config.min_xyz) / config.resolution
    # Pad the temporary map so historic sensor origins remain in bounds.
    # Cropping happens outside the official core, after ray tracing.
    lower = np.minimum(np.floor(origin[:2]).astype(int), 0)
    upper = np.maximum(np.floor(origin[:2]).astype(int) + 1, config.shape_xyz[:2])
    grid = OccupancyGridMap(*(upper - lower), 1.)
    grid.updateOrigin(*lower)
    sx, sy, sz = config.shape_xyz
    def run(points):
        grid.resetMaps()
        grid.raytrace2D(points, origin[:2])
        return grid.costmap_[-lower[1]:sy - lower[1], -lower[0]:sx - lower[0]].T.copy()
    layers = np.full(config.shape_xyz, NO_INFORMATION, np.uint8)
    iz = np.floor(relative[:, 2]).astype(int)
    for z in np.unique(iz):
        if 0 <= z < sz:
            layers[:, :, z] = run(relative[iz == z, :2])
    bev = run(relative[(iz >= 0) & (iz < sz), :2])
    return layers, bev
