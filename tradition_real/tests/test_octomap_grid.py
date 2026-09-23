"""Project adaptation checks independent of the core C++ parity suite."""
from dataclasses import replace
import itertools
import unittest
import numpy as np
from tradition_real.config import Config
from tradition_real.adapters.octomap_grid import OctomapGrid, voxel_indices


class OctomapGridTests(unittest.TestCase):
    def test_slanted_ray_and_outside_endpoint(self):
        cfg = Config(min_xyz=(0., 0., -2.6), shape_xyz=(8, 8, 14))
        grid = OctomapGrid(cfg)
        endpoint = np.array([[2.9, 2.5, 2.5]])
        grid.insert(endpoint, [.1, .1, -2.5])
        probability, observed, occupied, free, bev = grid.export()
        self.assertGreater(np.count_nonzero(free.any(axis=(0, 1))), 5)
        cell = tuple(voxel_indices(endpoint, cfg)[0][0])
        self.assertTrue(occupied[cell])
        self.assertEqual(occupied.sum(), 1)
        self.assertTrue(np.all(probability[~observed] == .5))
        outside = OctomapGrid(cfg)
        outside.insert([[4.9, .1, -2.5]], [.1, .1, -2.5])
        _, seen, occ, empty, _ = outside.export()
        self.assertFalse(occ.any())
        self.assertTrue(empty[:, 0, 0].all())
        self.assertEqual(seen.sum(), cfg.shape_xyz[0])

    def test_pruned_export_and_known_half_probability(self):
        cfg = Config(min_xyz=(0., 0., -2.6), shape_xyz=(4, 4, 4))
        grid = OctomapGrid(cfg)
        for index in itertools.product(range(2), repeat=3):
            grid.tree.updateNode(tuple(32768 + x for x in index), True)
        leaves = list(grid.tree.iter_leaves())
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0][1], 2)
        # A present node at probability .5 is occupied at the default >= threshold.
        grid.tree.updateNode((32771, 32771, 32771), 0.)
        probability, observed, occupied, free, bev = grid.export()
        self.assertEqual(occupied.sum(), 9)
        self.assertTrue(occupied[:2, :2, :2].all())
        self.assertTrue(occupied[3, 3, 3])
        self.assertEqual(probability[3, 3, 3], .5)
        self.assertFalse(observed[2, 2, 2])
        self.assertEqual(probability[2, 2, 2], .5)
        self.assertFalse(free.any())
        # Check every cell against independent full-depth tree traversal.
        for index in np.ndindex(cfg.shape_xyz):
            node = grid.tree.search(tuple(32768 + x for x in index))
            self.assertEqual(observed[index], node is not None)
            if node is not None:
                self.assertEqual(probability[index], node.getOccupancy())
                self.assertEqual(occupied[index], grid.tree.isNodeOccupied(node))

    def test_lazy_and_eager_dense_export_agree(self):
        cfg = Config(min_xyz=(0., -2., -2.6), shape_xyz=(12, 10, 14))
        eager, lazy = OctomapGrid(cfg), OctomapGrid(replace(cfg, lazy_eval=True))
        rng = np.random.default_rng(9)
        for _ in range(3):
            points = rng.uniform([.1, -1.9, -2.5], [4.7, 1.9, 2.9], (30, 3))
            for grid in (eager, lazy):
                grid.insert(points, [.1, 0., 0.])
        for expected, actual in zip(eager.export(), lazy.export()):
            np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()
