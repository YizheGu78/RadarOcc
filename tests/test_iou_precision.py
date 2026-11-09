"""Checkpoint comparison must use raw IoUs, even if display values tie."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/'projects/occ_plugin/utils/formating.py'
spec = importlib.util.spec_from_file_location('metric_format', path)
formatting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formatting)


class IoUPrecisionTest(unittest.TestCase):
    def test_semantic_checkpoint_selection_distinguishes_rounded_tie(self):
        _, first = formatting.format_SSC_results([.943, .24011, .26011], return_dic=True)
        _, later = formatting.format_SSC_results([.943, .24022, .26022], return_dic=True)
        self.assertEqual(round(first['mean'], 3), round(later['mean'], 3))
        self.assertGreater(later['mean'], first['mean'])
        self.assertAlmostEqual(later['mean'], .25022, places=12)
        self.assertAlmostEqual(later['Background'], .24022, places=12)

    def test_scene_completion_export_keeps_precision(self):
        _, values = formatting.format_SC_results([.245123456789], return_dic=True)
        self.assertEqual(values['non-empty'], .245123456789)


if __name__ == '__main__':
    unittest.main()
