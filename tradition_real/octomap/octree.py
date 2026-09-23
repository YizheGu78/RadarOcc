# Copyright (c) 2009-2013, K.M. Wurm and A. Hornung, University of Freiburg.
# BSD-3-Clause: see ../vendor/octomap/LICENSE.txt.
# Translated from the immutable files listed in ../OCTOMAP_SOURCE_MANIFEST.json.
"""OcTree occupancy core, preserving upstream method names and float32 nodes.

Scope: default OcTree insertPointCloud (including maxrange/discretize/BBX and
lazy evaluation), computeRayKeys, search, update, expansion, pruning and inner
occupancy. Python replaces C++ pointers/KeySets with nodes/lists/sets. No radar,
ROI, temporal-window, label or RF policy belongs in this module.
"""
import math
import sys

import numpy as np


def logodds(probability):
    return float(np.float32(math.log(probability / (1.0 - probability))))


def probability(log_odds):
    return 1.0 - 1.0 / (1.0 + math.exp(log_odds))


def _norm(vector):
    # Vector3::norm_sq evaluates products and left-associated sums in float.
    x, y, z = vector
    return math.sqrt(float(np.float32(np.float32(x*x + y*y) + z*z)))


def computeChildIdx(key, depth):
    return ((key[0] >> depth) & 1) + (((key[1] >> depth) & 1) << 1) + (((key[2] >> depth) & 1) << 2)


class OcTreeNode:
    __slots__ = ('value', 'children')

    def __init__(self, value=0.0):
        self.value = float(np.float32(value))
        self.children = None

    def getLogOdds(self):
        return self.value

    def getOccupancy(self):
        return probability(self.value)

    def setLogOdds(self, value):
        self.value = float(np.float32(value))

    def addValue(self, value):
        self.value = float(np.float32(self.value + float(np.float32(value))))

    def getMaxChildLogOdds(self):
        return max((child.value for child in self.children if child is not None),
                   default=-float(np.finfo(np.float32).max)) if self.children else -float(np.finfo(np.float32).max)

    def updateOccupancyChildren(self):
        self.value = self.getMaxChildLogOdds()


class OcTree:
    tree_depth = 16
    tree_max_val = 32768

    def __init__(self, resolution):
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('Resolution must be finite and positive')
        self.resolution = float(resolution)
        self.resolution_factor = 1.0 / self.resolution
        self.root = None
        self.tree_size = 0
        self.use_bbx_limit = False
        self.bbx_min = self.bbx_max = np.zeros(3, np.float32)
        self.bbx_min_key = self.bbx_max_key = (32768, 32768, 32768)
        self.setOccupancyThres(.5)
        self.setProbHit(.7)
        self.setProbMiss(.4)
        self.setClampingThresMin(.1192)
        self.setClampingThresMax(.971)

    def setOccupancyThres(self, prob):
        self.occ_prob_thres_log = logodds(prob)

    def setProbHit(self, prob):
        self.prob_hit_log = logodds(prob)
        assert self.prob_hit_log >= 0

    def setProbMiss(self, prob):
        self.prob_miss_log = logodds(prob)
        assert self.prob_miss_log <= 0

    def setClampingThresMin(self, prob):
        self.clamping_thres_min = logodds(prob)

    def setClampingThresMax(self, prob):
        self.clamping_thres_max = logodds(prob)

    def isNodeOccupied(self, node):
        return node.value >= self.occ_prob_thres_log

    def size(self):
        return self.tree_size

    def clear(self):
        self.root = None
        self.tree_size = 0

    def coordToKeyChecked(self, point):
        key = tuple(math.floor(self.resolution_factor * float(c)) + self.tree_max_val for c in point)
        return key if all(0 <= k < 65536 for k in key) else None

    def coordToKey(self, point):
        return tuple((math.floor(self.resolution_factor * float(c)) + self.tree_max_val) & 65535 for c in point)

    def keyToCoord(self, key):
        # Scalar overload is double; point3d overload is float32.
        if np.isscalar(key):
            return (int(key) - self.tree_max_val + .5) * self.resolution
        return np.array([self.keyToCoord(k) for k in key], np.float32)

    def computeRayKeys(self, origin, end):
        origin, end = np.asarray(origin, np.float32), np.asarray(end, np.float32)
        key_origin, key_end = self.coordToKeyChecked(origin), self.coordToKeyChecked(end)
        if key_origin is None or key_end is None:
            return None  # C++ false; no ray inserted.
        if key_origin == key_end:
            return []
        ray = [key_origin]
        direction = end - origin
        length = np.float32(_norm(direction))
        direction = direction / length
        current_key = list(key_origin)
        step, t_max, t_delta = [], [], []
        for i in range(3):
            d = float(direction[i])
            step.append(1 if d > 0 else (-1 if d < 0 else 0))
            if step[i]:
                border = self.keyToCoord(current_key[i]) + float(np.float32(step[i] * self.resolution * .5))
                t_max.append((border - float(origin[i])) / d)
                t_delta.append(self.resolution / abs(d))
            else:
                t_max.append(sys.float_info.max)
                t_delta.append(sys.float_info.max)
        while True:
            # Keep upstream strict comparisons, especially edge/corner ties.
            if t_max[0] < t_max[1]:
                dim = 0 if t_max[0] < t_max[2] else 2
            else:
                dim = 1 if t_max[1] < t_max[2] else 2
            current_key[dim] = (current_key[dim] + step[dim]) & 65535
            t_max[dim] += t_delta[dim]
            current = tuple(current_key)
            if current == key_end or min(t_max) > float(length):
                break
            ray.append(current)
            assert len(ray) < 100000 - 1  # upstream KeyRay capacity
        return ray

    def useBBXLimit(self, enabled):
        self.use_bbx_limit = bool(enabled)

    def setBBXMin(self, point):
        self.bbx_min = np.asarray(point, np.float32)
        self.bbx_min_key = self.coordToKeyChecked(self.bbx_min)
        if self.bbx_min_key is None:
            raise ValueError('BBX minimum outside key range')

    def setBBXMax(self, point):
        self.bbx_max = np.asarray(point, np.float32)
        self.bbx_max_key = self.coordToKeyChecked(self.bbx_max)
        if self.bbx_max_key is None:
            raise ValueError('BBX maximum outside key range')

    def computeUpdate(self, scan, origin, maxrange=-1.0):
        scan, origin = np.asarray(scan, np.float32), np.asarray(origin, np.float32)
        free_cells, occupied_cells = set(), set()
        for point in scan:
            difference = point - origin
            length = _norm(difference)
            within_range = maxrange < 0 or length <= maxrange
            new_end = point
            if not within_range:
                direction = difference.copy()
                if length > 0:
                    direction /= np.float32(length)
                new_end = origin + direction * np.float32(maxrange)
            if not self.use_bbx_limit:
                ray = self.computeRayKeys(origin, new_end)
                if ray is not None:
                    free_cells.update(ray)
                if within_range:
                    key = self.coordToKeyChecked(point)
                    if key is not None:
                        occupied_cells.add(key)
            else:
                if within_range and np.all(point >= self.bbx_min) and np.all(point <= self.bbx_max):
                    key = self.coordToKeyChecked(point)
                    if key is not None:
                        occupied_cells.add(key)
                ray = self.computeRayKeys(origin, new_end)
                if ray is not None:
                    for key in ray:
                        if all(lo <= k <= hi for k, lo, hi in zip(key, self.bbx_min_key, self.bbx_max_key)):
                            free_cells.add(key)
                        else:
                            break
        free_cells.difference_update(occupied_cells)
        return free_cells, occupied_cells

    def computeDiscreteUpdate(self, scan, origin, maxrange=-1.0):
        endpoints, discrete = set(), []
        for point in np.asarray(scan, np.float32):
            key = self.coordToKey(point)
            if key not in endpoints:
                endpoints.add(key)
                discrete.append(self.keyToCoord(key))
        return self.computeUpdate(discrete, origin, maxrange)

    def insertPointCloud(self, scan, sensor_origin, maxrange=-1.0, lazy_eval=False, discretize=False):
        method = self.computeDiscreteUpdate if discretize else self.computeUpdate
        free_cells, occupied_cells = method(scan, sensor_origin, maxrange)
        for key in free_cells:
            self.updateNode(key, False, lazy_eval)
        for key in occupied_cells:
            self.updateNode(key, True, lazy_eval)

    @staticmethod
    def nodeHasChildren(node):
        return node.children is not None and any(child is not None for child in node.children)

    @staticmethod
    def nodeChildExists(node, index):
        return node.children is not None and node.children[index] is not None

    def createNodeChild(self, node, index):
        if node.children is None:
            node.children = [None] * 8
        assert node.children[index] is None
        child = OcTreeNode()
        node.children[index] = child
        self.tree_size += 1
        return child

    def search(self, key, depth=0):
        assert 0 <= depth <= self.tree_depth
        depth = depth or self.tree_depth
        node = self.root
        if node is None:
            return None
        # adjustKeyAtDepth only changes lower bits, which this traversal ignores.
        for bit in range(self.tree_depth - 1, self.tree_depth - depth - 1, -1):
            position = computeChildIdx(key, bit)
            if self.nodeChildExists(node, position):
                node = node.children[position]
            elif not self.nodeHasChildren(node):
                return node
            else:
                return None
        return node

    def expandNode(self, node):
        assert not self.nodeHasChildren(node)
        for index in range(8):
            self.createNodeChild(node, index).value = node.value

    def isNodeCollapsible(self, node):
        if not self.nodeChildExists(node, 0):
            return False
        first = node.children[0]
        if self.nodeHasChildren(first):
            return False
        return all(child is not None and not self.nodeHasChildren(child) and child.value == first.value
                   for child in node.children[1:])

    def pruneNode(self, node):
        if not self.isNodeCollapsible(node):
            return False
        node.value = node.children[0].value
        node.children = None
        self.tree_size -= 8
        return True

    def updateNodeLogOdds(self, node, update):
        node.addValue(update)
        if node.value < self.clamping_thres_min:
            node.value = self.clamping_thres_min
            return
        if node.value > self.clamping_thres_max:
            node.value = self.clamping_thres_max

    def updateNode(self, key, occupied_or_log_odds, lazy_eval=False):
        update = (self.prob_hit_log if occupied_or_log_odds else self.prob_miss_log) if isinstance(occupied_or_log_odds, (bool, np.bool_)) else float(np.float32(occupied_or_log_odds))
        leaf = self.search(key)
        if leaf is not None and ((update >= 0 and leaf.value >= self.clamping_thres_max)
                                 or (update <= 0 and leaf.value <= self.clamping_thres_min)):
            return leaf
        created_root = self.root is None
        if created_root:
            self.root = OcTreeNode()
            self.tree_size += 1
        return self.updateNodeRecurs(self.root, created_root, key, 0, update, lazy_eval)

    def updateNodeRecurs(self, node, node_just_created, key, depth, update, lazy_eval):
        if depth < self.tree_depth:
            position = computeChildIdx(key, self.tree_depth - 1 - depth)
            created_node = False
            if not self.nodeChildExists(node, position):
                if not self.nodeHasChildren(node) and not node_just_created:
                    self.expandNode(node)
                else:
                    self.createNodeChild(node, position)
                    created_node = True
            result = self.updateNodeRecurs(node.children[position], created_node, key, depth + 1, update, lazy_eval)
            if not lazy_eval:
                if self.pruneNode(node):
                    result = node
                else:
                    node.updateOccupancyChildren()
            return result
        self.updateNodeLogOdds(node, update)
        return node

    def updateInnerOccupancy(self):
        if self.root is not None:
            self.updateInnerOccupancyRecurs(self.root, 0)

    def updateInnerOccupancyRecurs(self, node, depth):
        if self.nodeHasChildren(node):
            if depth < self.tree_depth:
                for child in node.children:
                    if child is not None:
                        self.updateInnerOccupancyRecurs(child, depth + 1)
            node.updateOccupancyChildren()

    def prune(self):
        if self.root is None:
            return
        for depth in range(self.tree_depth - 1, -1, -1):
            if not self._pruneRecurs(self.root, 0, depth):
                break

    def _pruneRecurs(self, node, depth, max_depth):
        if depth < max_depth:
            return sum(self._pruneRecurs(child, depth + 1, max_depth)
                       for child in (node.children or []) if child is not None)
        return int(self.pruneNode(node))

    def iter_leaves(self):
        """Python traversal of upstream leaf topology, yielding key origin/span/node.

        Key origins denote the minimum full-depth key of each (possibly pruned)
        leaf. This adapter-facing iterator does not modify the tree.
        """
        if self.root is None:
            return
        stack = [(self.root, (0, 0, 0), 65536)]
        while stack:
            node, lower, span = stack.pop()
            if not self.nodeHasChildren(node):
                yield lower, span, node
                continue
            half = span // 2
            for index in range(7, -1, -1):
                child = node.children[index]
                if child is not None:
                    start = tuple(lower[axis] + (half if index & (1 << axis) else 0) for axis in range(3))
                    stack.append((child, start, half))
