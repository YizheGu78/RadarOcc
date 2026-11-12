"""Scratch run on the corrected Small projection; previous artifacts retained."""
_base_ = ['../base.py']
model = {'temporal_cfg': {'type': 'TemporalPlugin',
                  'mode': 'ra_adta',
                  'position': 'P1',
                  'alignment': {'bins_path': 'data/K-Radar-official-meta/resources/info_arr.mat',
                                'coordinate_convention': 'kradar_physical_v2',
                                'azimuth_sign': -1,
                                'elevation_sign': -1},
                  'attention': {'channels': 8, 'embed_dims': 32, 'num_heads': 4, 'sampling_points_per_head': 2, 'candidate_pool_size': 8, 'search_mode': 'anisotropic_knn', 'alpha_range': 3.0, 'alpha_azimuth': 3.0, 'alpha_elevation': 3.0, 'min_angular_scale_m': .25, 'offset_bound': .5, 'temperature': .5, 'interpolation_cutoff': 2.0, 'chunk_size': 512, 'search_chunk_size': 1024, 'checkpoint_chunks': True, 'time_scale_frames': 3.0, 'max_log_correction': .5},
                  'debug': {'enabled': False}}}
data = {'train': {'current_only': False}, 'val': {'current_only': False}, 'test': {'current_only': False}}
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/ra_adta'
