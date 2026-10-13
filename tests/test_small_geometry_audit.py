import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT/'tools/temporal_vis/check_small_geometry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SmallGeometryAuditTests(unittest.TestCase):
    def setUp(self):
        self.bins = (np.arange(256)*0.462890625, np.arange(-53,54), np.arange(-18,19))

    def test_corrected_zero_angle_centres_and_xyz(self):
        result = module.audit([[108,53,18]], self.bins, formula='corrected')
        # Original indices R=108,A=53,E=18 -> padded [216,255,27];
        # two stride-2 layers nominally map to [54,63.75,6.75].
        target = [54,63.75,6.75]
        np.testing.assert_allclose(result['expected_feature'][0], target, atol=1e-12)
        np.testing.assert_allclose(result['query_feature'][0], target, atol=1e-12)
        np.testing.assert_allclose(result['spherical_error'][0], [0,0,0], atol=1e-12)
        np.testing.assert_allclose(result['query_xyz'][0], result['true_xyz'][0], atol=1e-12)
        self.assertLess(result['euclidean_error'][0], 1e-12)

    def test_corrected_varied_bins_and_nonzero_angles(self):
        # Check every axis at different noncentral indices, excluding E bins
        # outside the sparse encoder / feature-centre spatial bounds.
        indices = [[r,a,e] for r in (1,20,85,174)
                   for a in (0,17,53,86,106) for e in (5,8,18,25,30)]
        result = module.audit(indices, self.bins, formula='corrected')
        self.assertTrue(result['valid'].all())
        np.testing.assert_allclose(result['feature_error'], 0, atol=1e-12)
        np.testing.assert_allclose(result['spherical_error'], 0, atol=1e-12)
        np.testing.assert_allclose(result['euclidean_error'], 0, atol=1e-11)

    def test_legacy_formula_still_matches_historical_offset(self):
        result = module.audit([[108,53,18]], self.bins, formula='legacy')
        np.testing.assert_allclose(result['expected_feature'][0],[54,63.75,6.75])
        np.testing.assert_allclose(result['query_feature'][0,1:],[63.5,6.5])
        np.testing.assert_allclose(result['spherical_error'][0,1:],[-.5,-.5])
        # Independent trigonometric oracle from the historical implementation.
        r = self.bins[0][108]
        rq = (r/.46-1)*.462890625
        angle = np.pi/360
        expected = [rq*np.cos(angle)**2,
                    rq*np.cos(angle)*(-np.sin(angle)),
                    -rq*np.sin(angle)]
        np.testing.assert_allclose(result['query_xyz'][0], expected)
        self.assertAlmostEqual(result['euclidean_error'][0], .6337464750438865)

    def test_active_formula_detected_from_source(self):
        hashes, kernel_local, formula = module.verify_sources(ROOT)
        self.assertEqual(formula, 'corrected')
        self.assertIn(module.DETECTOR, hashes)
        self.assertIn(module.ENCODER, hashes)

    def test_invalid_padding_and_zero_range_are_excluded(self):
        result = module.audit([[0,53,18],[10,53,0],[10,53,36],[10,53,18]],
                              self.bins, formula='corrected')
        np.testing.assert_array_equal(result['encoder_valid'],[True,False,False,True])
        np.testing.assert_array_equal(result['valid'],[False,False,False,True])
        with self.assertRaises(ValueError):
            module.audit([[256,53,18]],self.bins)
        with self.assertRaises(ValueError):
            module.audit([[10.5,53,18]],self.bins)


if __name__ == '__main__':
    unittest.main()
