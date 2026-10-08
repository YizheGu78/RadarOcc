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
        self.bins = (np.arange(256)*0.462890625, np.arange(-53,54),np.arange(-18,19))

    def test_zero_angle_centres_and_xyz(self):
        result = module.audit([[108,53,18]],self.bins)
        # 0-degree source is padded at A=255,E=27. Two centred stride-2
        # convolutions yield A=63.75,E=6.75; CUDA's reference .5 => 63.5,6.5.
        np.testing.assert_allclose(result['expected_feature'][0],[54,63.75,6.75])
        np.testing.assert_allclose(result['query_feature'][0,1:],[63.5,6.5])
        np.testing.assert_allclose(result['spherical_error'][0,1:],[-.5,-.5])
        # Independent trigonometric oracle for the corresponding query ray.
        r=self.bins[0][108]
        rq=(r/.46-1)*.462890625
        angle=np.pi/360
        expected=[rq*np.cos(angle)**2,rq*np.cos(angle)*(-np.sin(angle)),-rq*np.sin(angle)]
        np.testing.assert_allclose(result['query_xyz'][0],expected)
        self.assertAlmostEqual(result['euclidean_error'][0],.6337464750438865)

    def test_invalid_padding_and_zero_range_are_excluded(self):
        result=module.audit([[0,53,18],[10,53,0],[10,53,36],[10,53,18]],self.bins)
        np.testing.assert_array_equal(result['encoder_valid'],[True,False,False,True])
        np.testing.assert_array_equal(result['valid'],[False,False,False,True])
        with self.assertRaises(ValueError): module.audit([[256,53,18]],self.bins)
        with self.assertRaises(ValueError): module.audit([[10.5,53,18]],self.bins)


if __name__ == '__main__':
    unittest.main()
