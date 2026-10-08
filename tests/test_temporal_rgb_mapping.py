"""Verify ordinal RGB association without equating radar/camera filenames."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('alignment_vis', ROOT / 'tools/temporal_vis/vis_alignment.py')
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


class RGBMappingTests(unittest.TestCase):
    def test_natural_order_uses_scene_ordinal_not_filename_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('cam_100.png', 'cam_10.png', 'cam_2.png'):
                (root / name).touch()
            infos = [dict(scene_id=3, order_in_scene=i, lidar_token=f'3_{i:05d}', radar_frame_id=i + 31) for i in range(3)]
            path, mapping = vis.resolve_rgb(infos[1], infos, camera_dir=root)
            self.assertEqual(path.name, 'cam_10.png')
            self.assertEqual(mapping['camera_ordinal'], 1)

    def test_subset_requires_full_reference_and_checks_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(4): (root / f'cam_{i}.jpg').touch()
            infos = [dict(scene_id=3, order_in_scene=i, lidar_token=f'3_{i:05d}') for i in range(4)]
            with self.assertRaises(ValueError): vis.resolve_rgb(infos[2], infos[2:], camera_dir=root)
            path, mapping = vis.resolve_rgb(infos[2], infos, camera_dir=root)
            self.assertEqual(path.name, 'cam_2.jpg')
            self.assertEqual(mapping['reference_count'], 4)
            with self.assertRaises(IndexError):
                vis.resolve_rgb(infos[-1], infos, camera_dir=root, camera_offset=1)

    def test_metadata_and_explicit_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'current.png'; path.touch()
            info = dict(cams={'front': {'data_path': str(path)}})
            self.assertEqual(vis.resolve_rgb(info, [info])[0], path)
            info['cams']['rear'] = {'data_path': str(path)}
            with self.assertRaises(ValueError): vis.resolve_rgb(info, [info])
            self.assertEqual(vis.resolve_rgb(info, [info], camera_name='front')[1]['camera'], 'front')
            self.assertEqual(vis.resolve_rgb({}, [], rgb_image=path)[1]['method'], 'explicit_image')


if __name__ == '__main__':
    unittest.main()
