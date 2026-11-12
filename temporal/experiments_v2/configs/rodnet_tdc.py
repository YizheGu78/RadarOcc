"""Scratch run on the corrected Small projection; previous artifacts retained."""
_base_ = ['../base.py']
model = {'type': 'RadarOccP0Temporal',
 'temporal_cfg': None,
 'p0_temporal_cfg': {'method': 'rodnet_tdc',
                     'bins_path': 'data/K-Radar-official-meta/resources/info_arr.mat',
                     'coordinate_convention': 'kradar_physical_v2',
                     'frame_nums': 4,
                     'channels': 32,
                     'chunk_size': 1024,
                     'bev_bounds': (-4.0, -64.0, 84.0, 64.0),
                     'cell_size': 0.8,
                     'height_bounds': (-4.0, 8.0),
                     'height_cell_size': 1.0,
                     'checkpoint_chunks': True,
                     'max_log_correction': 0.5,
                     'kernel_size': 3}}
data = {'train': {'current_only': False}, 'val': {'current_only': False}, 'test': {'current_only': False}}
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/rodnet_tdc'
