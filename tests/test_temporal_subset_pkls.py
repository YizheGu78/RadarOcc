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

    def test_full_scenes_preserve_schema_and_determinism(self):
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
                selected = set(report['selected_scenes'])
                self.assertEqual(result['infos'], [x for x in source['infos']
                                                    if x['scene_id'] in selected])
                self.assertLess(abs(report['fraction_actual'] - .2), .05)
                first_results[stem] = name.read_bytes()
            subsets.main(args + ['--overwrite'])
            for stem in ('train', 'val'):
                self.assertEqual(first_results[stem],
                                 (output_dir / (stem + '_20pct.pkl')).read_bytes())
            self.assertEqual(train, pickle.loads((root / 'train.pkl').read_bytes()))

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
