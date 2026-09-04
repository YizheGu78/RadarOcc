import numpy as np

from tradition.visualization.radarocc_video import (
    _index_radarocc_predictions,
    _replace_background_for_visualization,
)


def test_radarocc_supplies_only_visual_background(tmp_path):
    token_dir = tmp_path / "nested" / "3_00000"
    token_dir.mkdir(parents=True)
    prediction_path = token_dir / "pred_c.npy"

    # Saved RadarOcc prediction order is [z, y, x, class]. Its foreground
    # voxel must not leak into the traditional foreground visualization.
    np.save(
        prediction_path,
        np.asarray(
            [
                [2, 20, 10, 1],
                [3, 21, 11, 1],
                [4, 22, 12, 2],
            ],
            dtype=np.int64,
        ),
    )

    traditional = np.zeros((128, 128, 14), dtype=np.uint8)
    traditional[30, 40, 5] = 1
    traditional[10, 20, 2] = 2
    traditional[50, 60, 6] = 2

    indexed = _index_radarocc_predictions(tmp_path)
    assert indexed == {"3_00000": prediction_path.resolve()}

    visual = _replace_background_for_visualization(
        traditional,
        indexed["3_00000"],
    )

    # RadarOcc label-1 becomes the dense blue base.
    assert visual[10, 20, 2] == 2  # Traditional red wins on overlap.
    assert visual[11, 21, 3] == 1
    # RadarOcc foreground is not copied; foreground is traditional-only.
    assert visual[12, 22, 4] == 0
    assert visual[50, 60, 6] == 2
    # The old sparse traditional background is replaced, not merged.
    assert visual[30, 40, 5] == 0
    assert np.count_nonzero(visual == 1) == 1
    assert np.count_nonzero(visual == 2) == 2
