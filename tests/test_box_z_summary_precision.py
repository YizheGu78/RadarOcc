"""Saved best checkpoints and summary metrics must agree despite logger rounding."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.temporal.summarize_box_z_ablation import IOU_KEYS, read_epoch_metrics, select_best, summarize


class SummaryPrecisionTest(unittest.TestCase):
    def test_buffered_console_values_restore_precision_and_best_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root/'Z0'; directory.mkdir()
            exact = [dict(zip(IOU_KEYS, [.269646, .940451, .268648, .171232, score]))
                     for score in [.21993995379685233, .21994070978271388]]
            (directory/'a.log.json').write_text('\n'.join(
                json.dumps(dict(epoch=i, mode='val', **{k: round(v, 5) for k, v in row.items()}))
                for i, row in enumerate(exact, 1)))
            # Evaluation stdout arrives after the logger has emitted its epoch.
            (directory/'console.log').write_text(
                'Epoch(val) [1][211]\nSaving checkpoint at 2 epochs\n'+repr(exact[0])+
                '\nEpoch(val) [2][211]\n'+repr(exact[1])+'\n')
            (directory/'best_SSC_mean_epoch_2.pth').touch()
            summarize(root, 211)
            row = json.loads((root/'comparison.json').read_text())[0]
            self.assertEqual(row['epoch'], 2)
            self.assertEqual(row['SSC_mean'], exact[1]['SSC_mean'])
            self.assertEqual(row['metric_source'], 'console_eval_values')
            self.assertIn('best_SSC_mean_epoch_2.pth', row['checkpoint'])

    def test_incomplete_console_does_not_shift_metrics_and_saved_best_wins_rounded_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            row = dict(zip(IOU_KEYS, [.26965, .94045, .26865, .17123, .21994]))
            (directory/'a.log.json').write_text('\n'.join(
                json.dumps(dict(epoch=i, mode='val', **row)) for i in (1, 2)))
            (directory/'console.log').write_text(
                'Epoch(val) [1][211]\n'+repr(row)+'\nEpoch(val) [2][211]\n')
            (directory/'best_SSC_mean_epoch_2.pth').touch()
            values = read_epoch_metrics(directory, 'Z0')
            self.assertTrue(all(r['metric_source'] == 'log_json' for r in values))
            self.assertEqual(select_best(values, directory)['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
