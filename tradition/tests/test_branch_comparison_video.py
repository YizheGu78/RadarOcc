import numpy as np

from tradition.core.config import GridConfig
from tradition.visualization import branch_comparison_video
from tradition.visualization.branch_comparison_video import _points_to_sparse_zyx


def test_branch_points_are_voxelized_in_radarocc_zyx_order():
    points = np.asarray(
        [
            [0.2, -25.4, -2.4],
            [0.2, -25.4, -2.4],
            [51.3, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    coordinates = _points_to_sparse_zyx(points, GridConfig())

    np.testing.assert_array_equal(coordinates, [[0, 0, 0]])


def test_branch_renderer_clears_stale_frames(tmp_path):
    frames_dir = tmp_path / "branch_frames"
    frames_dir.mkdir()
    stale_frame = frames_dir / "frame_99999.png"
    stale_frame.touch()

    branch_comparison_video.BranchComparisonVideoRenderer(
        tmp_path,
        scene="3",
        keep_frames=True,
    )

    assert not stale_frame.exists()
