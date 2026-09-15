import csv

from tradition.evaluation.radarocc_metrics import MetricResult
from tradition.reporting.report_writer import MetricsReportWriter


EXPECTED_KEYS = [
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


def test_report_uses_only_radarocc_keys_and_zero_to_one_scale(tmp_path):
    results = [
        MetricResult(
            range_m=distance,
            sc_iou=0.1234,
            ssc_miou=0.2345,
            free_iou=0.3456,
            background_iou=0.4567,
            foreground_iou=0.5678,
        )
        for distance in (51.2, 25.6, 12.8)
    ]

    outputs = MetricsReportWriter().write(results, tmp_path)

    with outputs["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["Metric", "Traditional"]
    assert [row[0] for row in rows[1:]] == EXPECTED_KEYS
    assert len(rows) == 16
    assert rows[1] == ["SC_non-empty", "0.123"]
    assert rows[4] == ["SSC_free", "0.346"]
    assert set(outputs) == {"csv", "markdown", "png"}
    assert all(path.is_file() for path in outputs.values())
