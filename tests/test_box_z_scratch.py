"""Scratch experiments must jointly train, retain a trained baseline and exact metrics."""
import ast
import json
import math
from numbers import Real
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from mmcv import Config
from tools.temporal.summarize_box_z_ablation import ALL_IOU_KEYS, summarize

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'projects/occ_plugin/core/evaluation/eval_hooks.py'
node = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'save_box_z_metrics')
namespace = dict(json=json, math=math, Path=Path, Real=Real)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)


class ScratchExperimentTest(unittest.TestCase):
    def test_random_initialization_and_joint_training_configuration(self):
        cfg = Config.fromfile(str(ROOT/'projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct_scratch.py'))
        self.assertIsNone(cfg.load_from); self.assertIsNone(cfg.resume_from)
        self.assertFalse(cfg.model.box_z_cfg.freeze_original)
        self.assertEqual(cfg.runner.max_epochs, 15)
        self.assertEqual(cfg.optimizer.lr, 3e-4)
        self.assertIn('_20pct.pkl', cfg.data.train.ann_file)
        self.assertEqual(cfg.model.type, 'RadarOccBoxZ')

    def test_evaluation_hook_preserves_raw_metric_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            score = .21994070978271388
            runner = SimpleNamespace(work_dir=temp, epoch=6,
                model=SimpleNamespace(box_z_fusion=object()),
                log_buffer=SimpleNamespace(output={'SSC_mean': score, 'SC2_non-empty': .6,
                                                   'SSC2_Foreground': float('nan'),
                                                   'eval_iter_num': 211}))
            namespace['save_box_z_metrics'](runner)
            row = json.loads((Path(temp)/'val_metrics.jsonl').read_text())
            self.assertEqual(row['epoch'], 7)
            self.assertEqual(row['SSC_mean'], score)
            self.assertNotIn('eval_iter_num', row)
            self.assertIsNone(row['SSC2_Foreground'])

    def test_trained_single_frame_control_and_all_distance_metrics_are_summarized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # An obsolete initialization-only score must not replace a trained control.
            (root/'single_frame_metrics.json').write_text(json.dumps({'SSC_mean': .99}))
            for method, score in [('Single-frame', .220000123), ('Z1', .230000456)]:
                directory = root/method; directory.mkdir()
                row = dict(mode='val', epoch=2, **{k: .5 for k in ALL_IOU_KEYS})
                row['SSC_mean'] = score
                (directory/'a.log.json').write_text(json.dumps(dict(row, SSC_mean=round(score, 5))))
                (directory/'val_metrics.jsonl').write_text(json.dumps(row)+'\n')
                (directory/'best_SSC_mean_epoch_2.pth').touch()
            summarize(root, 211)
            rows = json.loads((root/'comparison.json').read_text())
            self.assertEqual([r['method'] for r in rows], ['Single-frame', 'Z1'])
            self.assertEqual(rows[0]['epoch'], 2)
            self.assertEqual(rows[0]['SSC_mean'], .220000123)
            self.assertEqual(rows[1]['SSC_mean'], .230000456)
            self.assertEqual(rows[0]['SSC2_mean'], .5)
            self.assertEqual(rows[0]['metric_source'], 'val_metrics_jsonl')


if __name__ == '__main__':
    unittest.main()
