"""CPU-only, dependency-free regression tests for 20% temporal PKL subsets."""
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/temporal/create_temporal_subsets.py'
spec = importlib.util.spec_from_file_location('create_temporal_subsets', SCRIPT)
subsets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subsets)


class TemporalSubsetTests(unittest.TestCase):
    def make_source(self, path, scene_sizes):
        infos = []
        for scene, size in scene_sizes.items():
            for order in range(size):
                infos.append({
                    'scene_id': scene,
                    'order_in_scene': order,
                    'ego_pose': [[1, 0, 0, 0]] * 4,
                    'radar_path': f'/fake/{scene}/{order}.npz',
                    'occ_path': f'/fake/{scene}/{order}.npy',
                    'timestamp': 1000000 * (order + 1),
                    'custom_field': (scene, order),
                })
        # Interleave scenes; subset should preserve source info order.
        infos.sort(key=lambda row: (row['order_in_scene'], row['scene_id']))
        data = {'infos': infos, 'metadata': {'version': 'v-test'}, 'extra': [5, 8]}
        with path.open('wb') as f:
            pickle.dump(data, f)
        return data

    def test_per_scene_contiguous_windows_preserve_schema_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = {str(i): 10 + i * 2 for i in range(1, 12)}
            train = self.make_source(root / 'train.pkl', sizes)
            self.make_source(root / 'val.pkl', sizes)
            output_dir = root / 'outputs'
            args = ['--train-pkl', str(root / 'train.pkl'),
                    '--val-pkl', str(root / 'val.pkl'),
                    '--output-dir', str(output_dir), '--seed', '42']
            subsets.main(args)
            first_results = {}
            for stem in ('train', 'val'):
                name = output_dir / (stem + '_20pct.pkl')
                manifest = name.with_suffix('.selection.json')
                result = pickle.loads(name.read_bytes())
                report = json.loads(manifest.read_text())
                source = train if stem == 'train' else pickle.loads((root / 'val.pkl').read_bytes())
                self.assertEqual(source['metadata'], result['metadata'])
                self.assertEqual(source['extra'], result['extra'])
                self.assertGreater(len(result['infos']), 0)
                self.assertEqual(set(report['selected_scenes']), set(sizes))
                windows = report['scene_windows']
                self.assertEqual(set(windows), set(sizes))
                chosen = []
                for scene in sorted(sizes):
                    rows = [x for x in result['infos'] if x['scene_id'] == scene]
                    orders = sorted(x['order_in_scene'] for x in rows)
                    self.assertEqual(
                        orders,
                        list(range(windows[scene]['start_order_in_scene'],
                                   windows[scene]['end_order_in_scene'] + 1))
                    )
                    self.assertEqual(len(rows), windows[scene]['selected_frames'])
                chosen = [
                    x for x in source['infos']
                    if windows[x['scene_id']]['start_order_in_scene']
                    <= x['order_in_scene']
                    <= windows[x['scene_id']]['end_order_in_scene']
                ]
                self.assertEqual(result['infos'], chosen)
                self.assertLess(abs(report['fraction_actual'] - .2), .02)
                first_results[stem] = name.read_bytes()
            subsets.main(args + ['--overwrite'])
            for stem in ('train', 'val'):
                self.assertEqual(first_results[stem],
                                 (output_dir / (stem + '_20pct.pkl')).read_bytes())
            self.assertEqual(train, pickle.loads((root / 'train.pkl').read_bytes()))

    def test_two_scene_val_retains_both_and_near_20_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_source(root / 'val.pkl', {'scene3': 462, 'scene7': 597})
            out_path, _, subset, report = subsets.plan_subset(
                root / 'val.pkl', root, seed=43
            )
            self.assertTrue(out_path.name.endswith('_20pct.pkl'))
            self.assertEqual(report['total_scenes'], 2)
            self.assertEqual(report['selected_scenes'], ['scene3', 'scene7'])
            self.assertLess(abs(report['fraction_actual'] - .2), .005)
            self.assertEqual(
                len(subset['infos']),
                sum(row['selected_frames']
                    for row in report['scene_windows'].values())
            )

    def test_source_gaps_require_actual_consecutive_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root / 'train.pkl', {'scene': 30})
            # No continuous run of >= 6 records exists.
            for i, row in enumerate(source['infos']):
                row['order_in_scene'] = 2 * i
            with (root / 'train.pkl').open('wb') as f:
                pickle.dump(source, f)
            with self.assertRaisesRegex(ValueError, 'cannot find'):
                subsets.plan_subset(root / 'train.pkl', root, seed=42)

    def test_dry_run_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = {str(i): 7 for i in range(10)}
            self.make_source(root / 'train.pkl', sizes)
            self.make_source(root / 'val.pkl', sizes)
            args = ['--train-pkl', str(root/'train.pkl'),
                    '--val-pkl', str(root/'val.pkl'),
                    '--output-dir', str(root/'new')]
            subsets.main(args + ['--dry-run'])
            self.assertFalse((root/'new').exists())
            subsets.main(args)
            with self.assertRaises(SystemExit):
                subsets.main(args)

    def test_reject_duplicate_scene_order(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'train.pkl'
            payload = self.make_source(src, {'3': 7})
            payload['infos'].append(dict(payload['infos'][0]))
            with src.open('wb') as f:
                pickle.dump(payload, f)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                subsets.load_and_validate(src)


if __name__ == '__main__':
    unittest.main()
