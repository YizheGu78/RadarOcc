import numpy as np

from tradition.evaluation.radarocc_metrics import (
    RadarOccMetricAccumulator,
    radarocc_metric_dict,
)


def test_metrics_report_free_background_foreground_and_both_mious():
    gt = np.zeros((128, 128, 14), dtype=np.uint8)
    gt[1, 60, 1] = 1
    gt[2, 61, 2] = 2
    accumulator = RadarOccMetricAccumulator(ranges_m=(12.8,))
    accumulator.update(gt.copy(), gt)
    result = accumulator.results()[0]

    assert result.sc_iou == 1.0
    assert result.ssc_miou == 1.0
    assert result.free_iou == 1.0
    assert result.background_iou == 1.0
    assert result.foreground_iou == 1.0


def test_metrics_use_radarocc_xy_boxes_for_range_crops():
    gt = np.zeros((128, 128, 14), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[10, 10, 1] = 1
    pred[10, 10, 1] = 1
    gt[10, 60, 1] = 1

    accumulator = RadarOccMetricAccumulator()
    accumulator.update(pred, gt)
    by_range = {result.range_m: result for result in accumulator.results()}

    assert by_range[51.2].sc_iou == 0.5
    assert by_range[25.6].sc_iou == 0.0
    assert by_range[12.8].sc_iou == 0.0


def test_metrics_ignore_255_like_radarocc():
    gt = np.zeros((128, 128, 14), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[4, 60, 1] = 1
    pred[4, 60, 1] = 1
    gt[5, 60, 1] = 255
    pred[5, 60, 1] = 2

    accumulator = RadarOccMetricAccumulator()
    accumulator.update(pred, gt)

    for result in accumulator.results():
        assert result.sc_iou == 1.0
        assert result.background_iou == 1.0


def test_radarocc_output_has_exact_metric_keys_and_order():
    gt = np.zeros((128, 128, 14), dtype=np.uint8)
    gt[1, 60, 1] = 1
    gt[2, 61, 2] = 2
    accumulator = RadarOccMetricAccumulator()
    accumulator.update(gt.copy(), gt)

    metrics = radarocc_metric_dict(accumulator.results())

    assert list(metrics) == [
        "SC_non-empty",
        "SC1_non-empty",
        "SC2_non-empty",
        "SSC_free",
        "SSC_Background",
        "SSC_Foreground",
        "SSC_mean",
        "SSC1_free",
        "SSC1_Background",
        "SSC1_Foreground",
        "SSC1_mean",
        "SSC2_free",
        "SSC2_Background",
        "SSC2_Foreground",
        "SSC2_mean",
    ]
