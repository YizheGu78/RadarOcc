"""Exercise real sequence methods with only the heavy dataset base stubbed."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_dataset_class():
    class Registry:
        def register_module(self):
            return lambda cls: cls

    fake_mmdet = types.ModuleType('mmdet.datasets')
    fake_mmdet.DATASETS = Registry()
    base = types.ModuleType('projects.occ_plugin.datasets.nuscenes_occ_dataset')
    base.NuscOCCDataset = object
    parent_name = 'projects.occ_plugin.datasets.temporal_kradar_dataset'
    spec = importlib.util.spec_from_file_location(parent_name,
        ROOT / 'projects/occ_plugin/datasets/temporal_kradar_dataset.py')
    parent = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'mmdet.datasets': fake_mmdet,
                                 base.__name__: base, parent_name: parent}):
        spec.loader.exec_module(parent)
        spec2 = importlib.util.spec_from_file_location('unaligned_dataset_test',
            ROOT / 'projects/unaligned_temporal/dataset.py')
        child = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(child)
    return child.UnalignedTemporalKRadarDataset


class UnalignedDatasetTest(unittest.TestCase):
    def dataset(self):
        cls = load_dataset_class()
        ds = cls.__new__(cls)
        ds.frame_nums = 4
        ds.data_infos = [dict(scene_id=scene, order_in_scene=order,
                              sparse_radar_path='%s/%s.npz' % (scene, order),
                              occ_path='current.npy')
                         for scene, order in [('a', 8), ('b', 0), ('a', 2), ('a', 5)]]
        ds._validate_temporal_metadata()
        ds.sequence_indices, ds.sequence_valid_masks, ds.sequence_frame_offsets = ds._build_sequence_indices()
        return ds

    def test_pose_free_boundaries_gaps_and_missing_history(self):
        ds = self.dataset()
        info = ds.get_temporal_info(0)
        self.assertEqual(info['orders_in_scene'], [2, 2, 5, 8])
        np.testing.assert_array_equal(info['frame_offsets'], [-6, -6, -3, 0])
        np.testing.assert_array_equal(info['valid_mask'], [False, True, True, True])
        for idx in info['sequence_indices']:
            self.assertEqual(ds.data_infos[idx]['scene_id'], 'a')
        info_b = ds.get_temporal_info(1)
        np.testing.assert_array_equal(info_b['valid_mask'], [False, False, False, True])
        self.assertEqual(info_b['radar_paths'], ['b/0.npz'] * 4)
        # Arbitrary poses in annotations cannot affect the sequence/payload.
        for row in ds.data_infos:
            row['ego_pose'] = 'not a pose'
        ds._validate_temporal_metadata()
        np.testing.assert_array_equal(info['ego_poses'], ds.get_temporal_info(0)['ego_poses'])

    def test_duplicate_scene_order_rejected(self):
        ds = self.dataset()
        ds.data_infos.append(ds.data_infos[0].copy())
        with self.assertRaises(ValueError):
            ds._validate_temporal_metadata()


if __name__ == '__main__':
    unittest.main()
