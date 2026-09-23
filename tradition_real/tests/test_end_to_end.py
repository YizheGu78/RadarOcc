"""Synthetic train/test dataset with different cross-sensor frame IDs."""
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import shutil
import joblib
import numpy as np
from tradition_real.cli import main
from tradition_real.config import Config
from tradition_real.adapters.layers import voxel_indices
from tradition_real.semantics.random_forest import RandomForest


class EndToEndTests(unittest.TestCase):
    def test_train_evaluate_predictions_video_and_no_leakage(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV required for video integration test')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = Config()
            for scene in ['1', '3']:
                infos = []
                for directory in [root/'rpc'/scene, root/'poses'/scene/'pose', root/'calib'/scene/'info_calib', root/'camera'/scene]:
                    directory.mkdir(parents=True, exist_ok=True)
                (root/'calib'/scene/'info_calib/calib_radar_lidar.txt').write_text('frame difference,x,y\n30,-2.54,0.3\n')
                for i in range(12):
                    # Two separated clusters; BG long, FG compact, distinct power.
                    xyz = np.array([[8.21 + j*.4, -4.19, .21] for j in range(5)] +
                                   [[6.21+j*.4, 2.21, .61] for j in range(2)])
                    sensor = xyz - np.array([2.54, -.3, -.7])
                    rpc = np.zeros((len(xyz), 11))
                    rpc[:, :3] = sensor
                    rpc[:, 3] = [3]*5 + [30]*2
                    rpc[:, 4] = [0.1]*5 + [1.7]*2
                    rpc[:, 5] = np.linalg.norm(sensor, axis=1)
                    rpc[:, 6] = np.arctan2(sensor[:, 1], sensor[:, 0])
                    rpc[:, 7] = np.arcsin(sensor[:, 2]/rpc[:, 5])
                    rpc[:, 8:11] = [20, 53, 18]
                    rpc_path = root/'rpc'/scene/f'rpc_{31+i*2:05d}.npy'
                    np.save(rpc_path, rpc)
                    np.save(root/'poses'/scene/'pose'/f'lidar_ego_pose{i}.npy', np.eye(4))
                    cell, valid = voxel_indices(xyz, cfg)
                    gt_path = root/f'gt_{scene}_{i}.npy'
                    np.save(gt_path, np.column_stack([cell, [1]*5+[2]*2]))
                    token = f'{scene}_{i:05d}'
                    infos.append({'scene_token': scene, 'lidar_token': token, 'radar_path': f'missing_{i}.npz',
                                  'occ_path': str(gt_path), 'timestamp': i})
                    cv2.imwrite(str(root/'camera'/scene/f'image_{i*7+5:05d}.png'), np.full((60, 100, 3), 120, np.uint8))
                with (root/f'{scene}.pkl').open('wb') as stream:
                    pickle.dump({'infos': infos}, stream)
            common = ['--radar-root', str(root/'rpc'), '--pose-root', str(root/'poses'), '--calib-root', str(root/'calib')]
            main(['train', '--annotation', str(root/'1.pkl'), '--output', str(root/'train'), '--n-estimators', '40', *common])
            model_path = root/'train/random_forest.joblib'
            bundle = joblib.load(model_path)
            self.assertGreater(bundle['metadata']['background_samples'], 0)
            self.assertGreater(bundle['metadata']['foreground_samples'], 0)
            main(['evaluate', '--annotation', str(root/'3.pkl'), '--model', str(model_path), '--output', str(root/'test'),
                  '--save-predictions', '--camera-dir', str(root/'camera/3'), '--video-start', '2', '--video-end', '5', *common])
            run = json.loads((root/'test/run.json').read_text())
            self.assertEqual(run['processed_frames'], 12)
            self.assertEqual(run['video_frames'], 4)
            self.assertEqual(run['status'], 'completed')
            metrics = json.loads((root/'test/metrics.json').read_text())
            self.assertEqual(len(metrics), 15)
            self.assertEqual(metrics['SC_non-empty'], 1.)
            self.assertEqual(len(list((root/'test/predictions/3').glob('*.npz'))), 12)
            saved = np.load(root/'test/predictions/3/3_00000.npz')
            self.assertEqual(saved['labels_xyz'].shape, (128, 128, 14))
            self.assertGreater(saved['unknown_mask'].sum(), 0)
            capture = cv2.VideoCapture(str(root/'test/scene_3.mp4'))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            capture.release()
            # Preprocessing does not read GT or train RF.
            with patch('tradition_real.adapters.dataset.load_gt_sparse_xyz', side_effect=AssertionError('GT read')):
                for scene in ['1', '3']:
                    main(['preprocess', '--annotation', str(root/f'{scene}.pkl'),
                          '--output', str(root/f'cache_{scene}'), *common])
            manifest = json.loads((root/'cache_1/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(len(manifest['frames']), 12)
            self.assertIn('sha256', manifest['calibrations']['1'])
            with np.load(root/'cache_1/1/frame_1_00000.npz', allow_pickle=False) as cached:
                self.assertEqual(cached['proposal_features'].shape[1], 42)
                self.assertNotIn('targets', cached.files)
                self.assertNotIn('gt', cached.files)
            train_cached = ['train', '--annotation', str(root/'1.pkl'),
                            '--frame-fusion-root', str(root/'cache_1'),
                            '--output', str(root/'cached_train'), '--n-estimators', '40']
            # Explicit changed calibration is rejected even though cache-only
            # runs normally need no original calibration files.
            calibration = root/'calib/1/info_calib/calib_radar_lidar.txt'
            calibration.write_text('frame difference,x,y\n30,-1.54,0.3\n')
            with self.assertRaisesRegex(ValueError, 'calibration mismatch'):
                main([*train_cached, '--calib-root', str(root/'calib')])
            for directory in ['rpc', 'poses', 'calib']:
                shutil.rmtree(root/directory)
            with patch('tradition_real.pipeline.Pipeline.map_frame', side_effect=AssertionError('mapping called')), \
                 patch('tradition_real.semantics.features.DBSCAN', side_effect=AssertionError('DBSCAN called')):
                main(train_cached)
                cached_model = root/'cached_train/random_forest.joblib'
                main(['evaluate', '--annotation', str(root/'3.pkl'), '--model', str(cached_model),
                      '--frame-fusion-root', str(root/'cache_3'), '--output', str(root/'cached_test'),
                      '--save-predictions', '--camera-dir', str(root/'camera/3'),
                      '--video-start', '2', '--video-end', '5'])
                # RF hyperparameters and label policy change without preprocessing.
                main([*train_cached, '--output', str(root/'retuned'), '--n-estimators', '50',
                      '--max-depth', '6', '--class-weight', 'none', '--positive-fraction', '0.5',
                      '--negative-fraction', '0.01'])
                tuned = joblib.load(root/'retuned/random_forest.joblib')
                self.assertEqual(tuned['estimator'].max_depth, 6)
                self.assertIsNone(tuned['estimator'].class_weight)
                self.assertEqual(tuned['metadata']['positive_fraction'], .5)
                with self.assertRaisesRegex(ValueError, 'preprocessing config'):
                    main([*train_cached, '--temporal-window', '1'])
                with self.assertRaisesRegex(ValueError, 'preprocessing config'):
                    main([*train_cached, '--radar-z', '-0.8'])
            for name in ['features', 'targets']:
                with np.load(root/'train/training_features.npz') as online, np.load(root/'cached_train/training_features.npz') as cached:
                    np.testing.assert_array_equal(online[name], cached[name])
            self.assertEqual(json.loads((root/'cached_test/metrics.json').read_text()), metrics)
            self.assertEqual(json.loads((root/'cached_test/run.json').read_text())['video_frames'], 4)
            for prediction in (root/'test/predictions/3').glob('*.npz'):
                with np.load(prediction) as online, np.load(root/'cached_test/predictions/3'/prediction.name) as cached:
                    for name in online.files:
                        np.testing.assert_array_equal(online[name], cached[name])
            # Incomplete/missing/corrupt caches must fail, never fall back to RPC.
            cache_file = root/'cache_1/1/frame_1_00000.npz'
            original = cache_file.read_bytes()
            cache_file.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing cached frame'):
                main(train_cached)
            cache_file.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                main(train_cached)
            cache_file.write_bytes(original)
            manifest['status'] = 'running'
            (root/'cache_1/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                main(train_cached)
            manifest['status'] = 'completed'
            (root/'cache_1/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlaps'):

                main(['evaluate', '--annotation', str(root/'1.pkl'), '--model', str(model_path), '--output', str(root/'bad'), '--frame-fusion-root', str(root/'cache_1')])
            old = root/'old.joblib'
            joblib.dump({'estimator': bundle['estimator'], 'feature_names': bundle['feature_names']}, old)
            with self.assertRaisesRegex(ValueError, 'old tradition'):
                RandomForest.load(old, cfg)


if __name__ == '__main__':
    unittest.main()
