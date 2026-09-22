# Copyright 2021, 2024 Tier IV, Inc.
# Copyright (c) 2008, 2013, Willow Garage, Inc. All rights reserved.
# Apache-2.0 AND BSD-3-Clause; see ../vendor/LICENSE-Autoware and ../NOTICE.md.
# Python translation of the pinned C++ sources in ../SOURCE_MANIFEST.json.
"""Autoware CPU OccupancyGridMap subset, with original method names.

The map is [Y,X] uint8, exactly matching C++ index = y * size_x + x.
Inputs correspond to finite float32 PointCloud2 XY coordinates in map frame.
No radar-specific behavior belongs in this module.
"""
import math
import numpy as np

FREE_SPACE = 0
NO_INFORMATION = 128
LETHAL_OBSTACLE = 255
OCCUPIED_THRESHOLD = 180
FREE_THRESHOLD = 50


class OccupancyGridMap:
    def __init__(self, cells_size_x, cells_size_y, resolution):
        if min(cells_size_x, cells_size_y, resolution) <= 0:
            raise ValueError("Map dimensions and resolution must be positive")
        self.size_x_ = int(cells_size_x)
        self.size_y_ = int(cells_size_y)
        # The upstream constructor takes float, Costmap2D stores double.
        self.resolution_ = float(np.float32(resolution))
        self.origin_x_ = self.origin_y_ = 0.0
        self.first_iteration_ = True
        self.costmap_ = np.full((self.size_y_, self.size_x_), NO_INFORMATION, np.uint8)

    def resetMaps(self):
        self.costmap_.fill(NO_INFORMATION)

    def worldToMap(self, wx, wy):
        if wx < self.origin_x_ or wy < self.origin_y_:
            return None
        mx = math.floor((wx - self.origin_x_) / self.resolution_)
        my = math.floor((wy - self.origin_y_) / self.resolution_)
        return (mx, my) if mx < self.size_x_ and my < self.size_y_ else None

    def updateOrigin(self, new_origin_x, new_origin_y):
        cell_ox = math.floor((new_origin_x - self.origin_x_) / self.resolution_)
        cell_oy = math.floor((new_origin_y - self.origin_y_) / self.resolution_)
        new_grid_ox = self.origin_x_ + cell_ox * self.resolution_
        new_grid_oy = self.origin_y_ + cell_oy * self.resolution_
        lx = min(max(cell_ox, 0), self.size_x_)
        ly = min(max(cell_oy, 0), self.size_y_)
        ux = min(max(cell_ox + self.size_x_, 0), self.size_x_)
        uy = min(max(cell_oy + self.size_y_, 0), self.size_y_)
        saved = self.costmap_[ly:uy, lx:ux].copy()
        self.resetMaps()
        self.origin_x_, self.origin_y_ = new_grid_ox, new_grid_oy
        if self.first_iteration_:
            self.first_iteration_ = False
            return
        sx, sy = lx - cell_ox, ly - cell_oy
        if saved.size:
            self.costmap_[sy:sy + uy - ly, sx:sx + ux - lx] = saved

    def raytraceLine(self, value, x0, y0, x1, y1, max_length=10000, min_length=0):
        # nav2_costmap_2d::Costmap2D::raytraceLine / bresenham2D, humble.
        dx_full, dy_full = x1 - x0, y1 - y0
        dist = math.hypot(dx_full, dy_full)
        if dist < min_length:
            return
        mx = int(x0 + dx_full / dist * min_length) if dist > 0 else x0
        my = int(y0 + dy_full / dist * min_length) if dist > 0 else y0
        offset = my * self.size_x_ + mx
        dx, dy = x1 - mx, y1 - my
        ax, ay = abs(dx), abs(dy)
        ox = 1 if dx > 0 else -1
        oy = (1 if dy > 0 else -1) * self.size_x_
        scale = min(1., max_length / dist) if dist else 1.
        da, db, oa, ob = (ax, ay, ox, oy) if ax >= ay else (ay, ax, oy, ox)
        error = da // 2
        flat = self.costmap_.reshape(-1)
        for _ in range(min(int(scale * da), da)):
            flat[offset] = value
            offset += oa
            error += db
            if error >= da:
                offset += ob
                error -= da
        flat[offset] = value

    def raytraceFreespace(self, pointcloud, robot_pose_xy):
        ox, oy = map(float, robot_pose_xy)
        start = self.worldToMap(ox, oy)
        if start is None:
            return
        origin_x, origin_y = self.origin_x_, self.origin_y_
        end_x = origin_x + self.size_x_ * self.resolution_
        end_y = origin_y + self.size_y_ * self.resolution_
        for point in np.asarray(pointcloud, dtype=np.float32):
            wx, wy = float(point[0]), float(point[1])
            a, b = wx - ox, wy - oy
            if wx < origin_x:
                t = (origin_x - ox) / a
                wx, wy = origin_x, oy + b * t
            if wy < origin_y:
                t = (origin_y - oy) / b
                wx, wy = ox + a * t, origin_y
            # Preserve strict '>' and .001 clipping from upstream.
            if wx > end_x:
                t = (end_x - ox) / a
                wx, wy = end_x - .001, oy + b * t
            if wy > end_y:
                t = (end_y - oy) / b
                wx, wy = ox + a * t, end_y - .001
            end = self.worldToMap(wx, wy)
            if end is not None:
                self.raytraceLine(FREE_SPACE, *start, *end, 10000)

    def updateCellsByPointCloud(self, pointcloud, cost):
        for point in np.asarray(pointcloud, dtype=np.float32):
            cell = self.worldToMap(float(point[0]), float(point[1]))
            if cell is not None:
                self.costmap_[cell[1], cell[0]] = cost

    def updateFreespaceCells(self, pointcloud):
        self.updateCellsByPointCloud(pointcloud, FREE_SPACE)

    def updateOccupiedCells(self, pointcloud):
        self.updateCellsByPointCloud(pointcloud, LETHAL_OBSTACLE)

    def raytrace2D(self, pointcloud, robot_pose_xy):
        self.raytraceFreespace(pointcloud, robot_pose_xy)
        self.updateOccupiedCells(pointcloud)


def apply_bbf(observation, previous, occupied_to_occupied=.95,
              occupied_to_free=.05, free_to_free=.8,
              free_to_occupied=.2, v_ratio=.1):
    """Vectorized float32 translation of applyBBF; optional reference backend.

    GM2019 and BBF are alternative updaters, never applied consecutively.
    """
    z, old = np.broadcast_arrays(np.asarray(observation), np.asarray(previous))
    if not np.isin(z, [FREE_SPACE, NO_INFORMATION, LETHAL_OBSTACLE]).all():
        raise ValueError("BBF observations must be Autoware ternary costs")
    f = np.float32
    po = old.astype(np.float32) * (f(1) / f(255))
    result = np.zeros(po.shape, np.float32)
    for mask, pz, npz in [
        (z == LETHAL_OBSTACLE, f(occupied_to_occupied), f(occupied_to_free)),
        (z == FREE_SPACE, f(1) - f(free_to_free), f(1) - f(free_to_occupied)),
    ]:
        p = po[mask]
        result[mask] = p * pz / (p * pz + (f(1) - p) * npz)
    inv = f(1) / f(v_ratio)
    mask = z == NO_INFORMATION
    result[mask] = (po[mask] + f(.5) * inv) / (inv + f(1))
    # std::lround: half away from zero, unlike numpy.rint.
    return np.clip(np.floor(result * f(255) + f(.5)), 1, 254).astype(np.uint8)
