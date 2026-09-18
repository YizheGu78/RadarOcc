import numpy as np

from tradition.core.config import GridConfig
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
