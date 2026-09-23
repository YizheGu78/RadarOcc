from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tradition_real.config import Config
from tradition_real.frame_fusion import load_frame, save_frame, signature
from tradition_real.pipeline import Pipeline
from tradition_real.tests.test_pipeline import rpc_at


class FrameFusionTests(unittest.TestCase):
    def test_roundtrip_empty_and_causal_window(self):
        config = Config(min_xyz=(0., -4., -2.), shape_xyz=(32, 20, 10), temporal_window=3)
        pipeline = Pipeline(config)
        spec = signature(config, -.7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.npz'
            for i, rpc in enumerate([np.empty((0, 11)), rpc_at(), rpc_at(), rpc_at()]):
                frame = {'scene': '3', 'token': str(i), 'ordinal': i}
                mapped = pipeline.map_frame(rpc, np.eye(4), np.zeros(3), '3', i)
                save_frame(path, mapped, frame, spec)
                restored = load_frame(path, frame, spec)
                for field in ('probability', 'observed', 'occupied', 'free', 'bev_probability'):
                    np.testing.assert_array_equal(getattr(restored, field), getattr(mapped, field))
                self.assertEqual(len(restored.candidates), len(mapped.candidates))
                for original, cached in zip(mapped.candidates, restored.candidates):
                    for field in ('features', 'voxels'):
                        np.testing.assert_array_equal(original[field], cached[field])
                if i == 0:
                    with np.load(path, allow_pickle=False) as data:
                        self.assertEqual(data['proposal_features'].shape, (0, 42))
                        self.assertEqual(data['proposal_voxels'].shape, (0, 3))
                        np.testing.assert_array_equal(data['proposal_offsets'], [0])
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    load_frame(path, {**frame, 'ordinal': i + 1}, spec)

    def test_mapping_parameters_invalidate_but_rf_export_does_not(self):
        config = Config()
        original = signature(config, -.7)
        self.assertEqual(original, signature(replace(config, foreground_threshold=.8, unknown_export='background'), -.7))
        for overrides in ({'temporal_window': 1}, {'p_hit': .8}, {'prior': .48},
                          {'dbscan_eps_xy': 2.}, {'occupied_threshold': .6}):
            self.assertNotEqual(original, signature(replace(config, **overrides), -.7))
        self.assertNotEqual(original, signature(config, -.8))


if __name__ == '__main__':
    unittest.main()
