"""Rigid-geometry checks independent of MMCV, CUDA and checkpoints."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alignment', ROOT / 'temporal/core/ego_alignment.py')
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


class AlignmentTests(unittest.TestCase):
    def test_static_world_points_translation_rotation_and_lever_arm(self):
        poses = np.tile(np.eye(4), (1, 4, 1, 1))
        for t, angle in enumerate([0, .1, .3, .5]):
            c, s = np.cos(angle), np.sin(angle)
            poses[0, t, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
            poses[0, t, :3, 3] = [t, .2 * t, 0]
        extrinsic = alignment.radar_to_lidar_default()
        world = np.array([[10, 2, 1, 1], [20, -3, 0, 1]])
        local = np.stack([(np.linalg.inv(p @ extrinsic) @ world.T).T[:, :3] for p in poses[0]])[None]
        result, transforms = alignment.align_to_current(local, poses, extrinsic)
        np.testing.assert_allclose(result, np.broadcast_to(local[:, -1:], local.shape), atol=1e-12)
        np.testing.assert_array_equal(result[:, -1], local[:, -1])
        restored = np.einsum('btij,btnj->btni', np.linalg.inv(transforms)[..., :3, :3], result) + np.linalg.inv(transforms)[..., None, :3, 3]
        np.testing.assert_allclose(restored, local, atol=1e-12)
        wrong, _ = alignment.align_to_current(local, poses, np.eye(4))
        self.assertGreater(np.max(np.abs(wrong - result)), .1)

    def test_bin_lookup_and_axis_signs(self):
        geom = alignment.RadarBinGeometry([0, 2, 4], [0, np.pi / 2], [0, np.pi / 2])
        np.testing.assert_allclose(geom.to_xyz([[1, 0, 0], [2, 1, 0], [1, 0, 1]]),
                                   [[2, 0, 0], [0, 4, 0], [0, 0, 2]], atol=1e-12)
        negative = alignment.RadarBinGeometry([2], [np.pi / 2], [0], azimuth_sign=-1)
        np.testing.assert_allclose(negative.to_xyz([[0, 0, 0]]), [[0, -2, 0]], atol=1e-12)
        with self.assertRaises(ValueError): geom.to_xyz([[3, 0, 0]])
        with self.assertRaises(ValueError): geom.to_xyz([[1.5, 0, 0]])
        with self.assertRaises(ValueError): alignment.RadarBinGeometry([1], [53], [0])

    def test_invalid_history_and_invalid_pose(self):
        coords = np.ones((1, 4, 2, 3))
        poses = np.tile(np.eye(4), (1, 4, 1, 1))
        poses[0, 0, 0, 3] = 100
        result, transforms = alignment.align_to_current(coords, poses, valid_mask=[[False, True, True, True]])
        np.testing.assert_array_equal(result[:, 0], coords[:, 0])
        np.testing.assert_array_equal(transforms[0, 0], np.eye(4))
        with self.assertRaises(ValueError):
            alignment.align_to_current(coords, poses, valid_mask=[[True, True, True, False]])
        poses[0, 0, 0, 0] = 2
        with self.assertRaises(ValueError): alignment.align_to_current(coords, poses)


if __name__ == '__main__':
    unittest.main()
