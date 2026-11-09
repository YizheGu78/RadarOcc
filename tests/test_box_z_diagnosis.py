"""Check exact-count diagnostics without importing the CUDA model stack."""
import ast
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1]/'tools/temporal/diagnose_box_z_ablation.py'
nodes = ast.parse(path.read_text()).body
functions = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in ('metrics_from_counts', 'finite_json')]
namespace = dict(np=np)
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)


class DiagnosticMetricsTest(unittest.TestCase):
    def test_micro_iou_keeps_original_classes_ranges_and_excludes_free_from_mean(self):
        counts = dict(SC_metric=np.array([[90, 4], [6, 30]]),
                      SSC_metric=np.array([[90, 2, 3], [4, 20, 5], [6, 7, 10]]),
                      SSC_metric_1=np.eye(3, dtype=np.int64))
        result = namespace['metrics_from_counts'](counts)
        self.assertAlmostEqual(result['SC_non-empty'], 30/40)
        self.assertAlmostEqual(result['SSC_Background'], 20/38)
        self.assertAlmostEqual(result['SSC_Foreground'], 10/31)
        self.assertAlmostEqual(result['SSC_mean'], (20/38+10/31)/2)
        self.assertEqual(result['SSC1_mean'], 1.)
        self.assertNotEqual(result['SSC_mean'], round(result['SSC_mean'], 3))

    def test_absent_classes_export_as_null_instead_of_invalid_json_nan(self):
        self.assertEqual(namespace['finite_json']({'ious': [float('nan'), .25, float('inf')]}),
                         {'ious': [None, .25, None]})


if __name__ == '__main__':
    unittest.main()
