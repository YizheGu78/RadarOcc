"""The binding must preserve Python-reference results and existing cache keys."""
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tradition_real.config import Config
from tradition_real.frame_fusion import load_frame, save_frame, signature
from tradition_real.octomap import backend
from tradition_real.octomap.octree import OcTree as PythonOcTree
from tradition_real.pipeline import Pipeline


def leaves(tree):
    return sorted([*lower, span, node.value] for lower, span, node in tree.iter_leaves())


def rpc(points):
    result = np.zeros((len(points), 11), np.float64)
    result[:, :3] = points
    result[:, 3] = np.arange(len(points)) + 1
    result[:, 4] = np.linspace(-2, 2, len(points))
    result[:, 5] = np.linalg.norm(points, axis=1)
    return result


class NativeBackendTests(unittest.TestCase):
    def test_no_silent_fallback_and_cache_import_needs_no_binary(self):
        with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': 'cpp'}), \
             patch.object(backend, '_native', None), \
             patch.object(backend.importlib, 'import_module', side_effect=ImportError('missing')):
            # Cache consumers may construct Pipeline without ever constructing a tree.
            Pipeline(Config())
            with self.assertRaisesRegex(RuntimeError, 'build-octomap'):
                backend.OcTree(.4)
        with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': 'python'}), \
             patch.object(backend, 'native_module', side_effect=AssertionError('native loaded')):
            self.assertEqual(backend.OcTree(.4).backend, 'python')
        with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': 'gpu'}):
            with self.assertRaisesRegex(ValueError, 'cpp or python'):
                backend.OcTree(.4)
        with patch.object(backend, '_native', None), \
             patch.object(backend.importlib, 'import_module', return_value=SimpleNamespace(source_fingerprint='stale')):
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                backend.native_module()

    def test_rays_scans_parents_and_leaf_values(self):
        native = backend.native_module()
        rng = np.random.default_rng(902)
        for resolution in (.4, .2):
            a, b = PythonOcTree(resolution), native.OcTree(resolution)
            pairs = [(np.zeros(3), np.ones(3) * 4), (np.zeros(3), np.ones(3) * -4),
                     (np.zeros(3), np.zeros(3)), (np.full(3, .4), np.zeros(3))]
            pairs += [(rng.uniform(-6, 6, 3), rng.uniform(-6, 6, 3)) for _ in range(35)]
            for origin, end in pairs:
                self.assertEqual(a.computeRayKeys(origin, end), b.computeRayKeys(origin, end))
            for lazy, discrete, maximum, bbx in [(False, False, -1., False),
                    (True, False, -1., False), (False, True, 2.5, False),
                    (True, True, 2.5, True), (False, False, 0., False)]:
                for tree in (a, b):
                    tree.clear()
                    tree.setBBXMin([-2, -2, -2]); tree.setBBXMax([2, 2, 2]); tree.useBBXLimit(bbx)
                for frame in range(4):
                    points = rng.uniform(-4, 4, (25, 3))
                    points = np.vstack([points, points[:4], [[.8, 0, 0], [2, 0, 0]]])
                    for tree in (a, b):
                        tree.insertPointCloud(points, [0, 0, 0], maximum, lazy, discrete)
                    self.assertEqual(leaves(a), leaves(b))
                    self.assertEqual(a.size(), b.size())
                    self.assertEqual(None if a.root is None else a.root.value,
                                     None if b.root is None else b.root.value)
                for tree in (a, b):
                    tree.updateInnerOccupancy(); tree.prune()
                self.assertEqual(leaves(a), leaves(b))
                self.assertEqual(a.size(), b.size())
                for key in [(32768, 32768, 32768), (32770, 32770, 32770)]:
                    for depth in (0, 4, 15, 16):
                        na, nb = a.search(key, depth), b.search(key, depth)
                        self.assertEqual(None if na is None else na.value, None if nb is None else nb.value)

    def test_clamping_prune_expand_and_safe_snapshots(self):
        a, b = PythonOcTree(.4), backend.native_module().OcTree(.4)
        cells = [(32768 + x, 32768 + y, 32768 + z)
                 for x in range(2) for y in range(2) for z in range(2)]
        for tree in (a, b):
            for key in cells:
                tree.updateNode(key, True)
        self.assertEqual(leaves(a), leaves(b))
        snapshot = b.search(cells[0])
        before = snapshot.value
        for update in [True] * 12 + [False] * 22 + [True] * 9:
            for tree in (a, b):
                tree.updateNode(cells[0], update)
            self.assertEqual(leaves(a), leaves(b))
        b.clear()
        self.assertEqual(snapshot.value, before)  # no use-after-free
        self.assertGreater(snapshot.getOccupancy(), .5)

    def test_full_mapping_proposals_pose_window_and_cross_backend_cache(self):
        base = Config(min_xyz=(0., -4., -2.6), shape_xyz=(24, 20, 14), dbscan_min_samples=1)
        rng = np.random.default_rng(47)
        clouds = [rpc(rng.uniform([.2, -3.5, -2.4], [11., 3.5, 2.7], (22, 3))) for _ in range(7)]
        clouds[2] = np.empty((0, 11))
        # Include a ray outside ROI, duplicates and voxel-boundary endpoints.
        clouds[0] = rpc(np.array([[12., .4, 1.4], [4., .4, 1.4], [4., .4, 1.4], [0., 0., -2.6]]))
        for cfg in [base, replace(base, lazy_eval=True),
                    replace(base, discretize=True, max_range=4., p_hit=.8, p_miss=.3,
                            occupancy_threshold=.6, clamping_min=.1, clamping_max=.98)]:
            python_pipe, cpp_pipe = Pipeline(cfg), Pipeline(cfg)
            for i, points in enumerate(clouds):
                pose = np.eye(4)
                angle = i * .015
                pose[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                pose[0, 3] = i * .1
                ordinal, scene = (i, '3') if i < 5 else (i + 2, '4')
                mapped = []
                for name, pipe in [('python', python_pipe), ('cpp', cpp_pipe)]:
                    with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': name}):
                        mapped.append(pipe.map_frame(points, pose, np.array([.1, -.2, -.7]), scene, ordinal))
                for field in ('probability', 'observed', 'occupied', 'free', 'bev_probability'):
                    np.testing.assert_array_equal(getattr(mapped[0], field), getattr(mapped[1], field))
                self.assertEqual(len(mapped[0].candidates), len(mapped[1].candidates))
                for a, b in zip(mapped[0].candidates, mapped[1].candidates):
                    for field in ('features', 'voxels'):
                        np.testing.assert_array_equal(a[field], b[field])
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / 'frame.npz'
                    frame = {'scene': scene, 'token': str(i), 'ordinal': ordinal}
                    with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': 'python'}):
                        spec = signature(cfg, -.7)
                        save_frame(path, mapped[0], frame, spec)
                    with patch.dict(os.environ, {'TRADITION_REAL_OCTOMAP_BACKEND': 'cpp'}):
                        self.assertEqual(spec, signature(cfg, -.7))
                        restored = load_frame(path, frame, signature(cfg, -.7))
                    np.testing.assert_array_equal(restored.probability, mapped[1].probability)


if __name__ == '__main__':
    unittest.main()
