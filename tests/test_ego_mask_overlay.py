import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/temporal_vis'))
try:
    spec = importlib.util.spec_from_file_location('overlay_test', ROOT/'tools/temporal_vis/vis_ego_mask_overlay.py')
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
finally:
    sys.path.pop(0)


class OverlayTest(unittest.TestCase):
    def test_mask_before_transform_removes_displaced_self_echo(self):
        raw = np.array([[[[0.,0,0], [5,0,0]], [[0,0,0], [5,0,0]]]])
        poses = np.tile(np.eye(4), (1,2,1,1))
        poses[0,1,0,3] = 3
        aligned, _ = overlay.alignment.align_to_current(raw, poses)
        power = np.ones((8,2))
        after, mask = overlay.audit.zero_ego_power(raw[0,0], power, .5)
        self.assertEqual(aligned[0,0,0,0], -3)
        np.testing.assert_array_equal(mask, [True,False])
        self.assertEqual(after[2,0], 0)
        # Masking at the current origin would miss this displaced self echo.
        self.assertFalse(np.hypot(*aligned[0,0,0,:2]) <= .5)

    def test_history_start_padding_and_real_offsets(self):
        rows = [dict(order_in_scene=i, ego_pose=np.eye(4)) for i in [10,12,15]]
        hist,valid,offsets=overlay.history_for(rows,rows[1],4)
        np.testing.assert_array_equal(valid,[False,False,True,True])
        np.testing.assert_array_equal(offsets,[-2,-2,-2,0])
        self.assertIs(hist[-1],rows[1])

    def test_weak_self_echo_kept_in_original_selection(self):
        ids=overlay.selected_ids(np.array([1.,100.,90.]),np.array([[0,0,0],[10,0,0],[20,0,0]]),1,.5)
        np.testing.assert_array_equal(ids,[0,1])


if __name__=='__main__':
    unittest.main()
