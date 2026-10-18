import unittest
from pathlib import Path


class UnalignedConfigTest(unittest.TestCase):
    def test_opt_in_config_preserves_backbone_and_training_settings(self):
        try:
            from mmcv import Config
        except ImportError:
            self.skipTest('MMCV is required for config inheritance')
        root = Path(__file__).resolve().parents[1] / 'projects/configs/baselines'
        original = Config.fromfile(str(root / 'RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py'))
        new = Config.fromfile(str(root / 'RadarOcc_Small_5060_true_fp32_temporal_t4_unaligned_memory_p1.py'))
        self.assertEqual(new.model.type, 'RadarOccSmallUnalignedTemporal')
        self.assertIsNone(new.model.temporal_cfg)
        self.assertEqual(new.plugin_dir, 'projects/unaligned_temporal/')
        self.assertEqual(new.model.unaligned_temporal.num_layers, 2)
        self.assertEqual(new.model.unaligned_temporal.memory_tokens, 256)
        for key in ('optimizer', 'runner', 'lr_config', 'evaluation'):
            self.assertEqual(new[key], original[key])
        for key in ('pts_middle_encoder', 'occ_encoder_neck', 'occ_encoder_backbone',
                    'pts_bbox_head', 'self_transformer', 'cross_transformer'):
            self.assertEqual(new.model[key], original.model[key])
        for split in ('train', 'val', 'test'):
            self.assertEqual(new.data[split].type, 'UnalignedTemporalKRadarDataset')
            self.assertFalse(new.data[split].current_only)
            self.assertEqual(new.data[split].ann_file, original.data[split].ann_file)
        small = Config.fromfile(str(root / 'RadarOcc_Small_5060_true_fp32_temporal_t4_unaligned_memory_p1_20pct.py'))
        for split in ('train', 'val'):
            self.assertTrue(small.data[split].ann_file.endswith('_20pct.pkl'))
        self.assertEqual(small.data.test.ann_file, new.data.test.ann_file)


if __name__ == '__main__':
    unittest.main()
