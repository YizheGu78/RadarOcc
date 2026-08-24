import numpy as np

from tradition.evaluation.radarocc_metrics import RadarOccMetricAccumulator


def test_metrics_report_free_background_foreground_and_both_mious():
    gt = np.zeros((128, 128, 14), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[2, 2, 2] = 2
    accumulator = RadarOccMetricAccumulator(ranges_m=(12.8,))
    accumulator.update(gt.copy(), gt)
    result = accumulator.results()[0]

    assert result.sc_iou == 1.0
    assert result.ssc_miou == 1.0
    assert result.three_class_miou == 1.0
    assert result.free_iou == 1.0
    assert result.background_iou == 1.0
    assert result.foreground_iou == 1.0
