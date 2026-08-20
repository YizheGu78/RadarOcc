import numpy as np

from tradition.detection.cfar import NumpyCACFAR


def test_numpy_cfar_peak_exceeds_threshold():
    signal = np.zeros(64, dtype=np.float64)
    signal[20] = 30.0
    cfar = NumpyCACFAR(threshold_offset_db=6.0)
    threshold = cfar.threshold(signal, guard_len=1, noise_len=4)
    assert signal[20] > threshold[20]
