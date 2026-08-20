import numpy as np

from tradition.core.config import GridConfig, KRadarConfig
from tradition.core.geometry import radar_to_lidar, spherical_to_cartesian, xyz_to_voxel


def test_forward_axis_and_radar_to_lidar_translation():
    radar = spherical_to_cartesian(10.0, 0.0, 0.0)
    lidar = radar_to_lidar(radar, KRadarConfig())
    np.testing.assert_allclose(radar, [10.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(lidar, [12.54, -0.30, -0.70], atol=1e-7)


def test_radarocc_grid_shape_and_voxel_index():
    grid = GridConfig()
    assert grid.shape_xyz == (128, 128, 14)
    assert xyz_to_voxel([0.2, -25.4, -2.4], grid) == (0, 0, 0)
