import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mask_audit', ROOT/'tools/temporal_vis/vis_ego_power_mask.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class EgoPowerMaskTest(unittest.TestCase):
    def test_raw_xy_boundary_and_descriptor_preservation(self):
        # Boundary included, elevation does not alter XY masking, >r retained.
        xyz = np.array([[0, 0, 0], [.6, .8, 3], [1.01, 0, 0], [2, 0, 0]])
        power = np.arange(32, dtype=float).reshape(8, 4) + 1
        original = power.copy()
        after, mask = audit.zero_ego_power(xyz, power, 1)
        np.testing.assert_array_equal(mask, [True, True, False, False])
        np.testing.assert_array_equal(after[3:6], original[3:6])
        np.testing.assert_array_equal(after[:, ~mask], original[:, ~mask])
        np.testing.assert_array_equal(after[np.ix_([0,1,2,6,7], [0,1])], 0)
        np.testing.assert_array_equal(power, original)

    def test_every_scene_and_fallback(self):
        infos = [dict(scene_id=scene, order_in_scene=order)
                 for scene in ['1','10','2'] for order in [5, 10, 80]]
        selected = audit.choose_frames(infos)
        self.assertEqual([x[0] for x in selected], ['1','2','10'])
        self.assertTrue(all(x[1]['order_in_scene']==10 for x in selected))
        self.assertTrue(all(x[1]['order_in_scene']==80 for x in audit.choose_frames(infos, 80)))
        self.assertTrue(all(x[3]=='middle_of_split_scene' for x in audit.choose_frames(infos, 99)))

    def test_raster_keeps_max_power_with_shared_location(self):
        xyz = np.array([[.1,.1,0], [.2,.2,0], [1.1,1.1,0]])
        grid = audit.raster(xyz, np.array([9,99,999]), (0,2,0,2), 1)
        np.testing.assert_allclose(grid, [[2,0],[0,3]])


if __name__ == '__main__':
    unittest.main()
