"""C gates, corrected Small, scratch; Z0 shares the same pairs as Z1-Z3."""
_base_ = ['../base.py']
model = {'type': 'RadarOccBoxZ',
 'temporal_cfg': None,
 'box_z_cfg': {'mode': 'Z0',
               'bins_path': 'data/K-Radar-official-meta/resources/info_arr.mat',
               'coordinate_convention': 'kradar_physical_v2',
               'freeze_original': False,
               'enabled': True,
               'dim': 64,
               'heads': 4,
               'samples': 8,
               'chunk_size': 2048,
               'lambda_scale': 0.001,
               'lambda_shift': 0.001,
               'min_interp_weight': 0.0001,
               'diagnostic_path': 'work_dirs/geometry_v2/projection_fixed_20261112/box_z0_c_20261116/eval_diagnostics.jsonl'}}
data = {'train': {'type': 'BoxZTemporalDatasetV2',
           'current_only': False,
           'box_cache': 'data/geometry_v2/projection_fixed_20261112/box_pairs_c_20261116/train20',
           'expected_y_gates': [2, 2, 1.5],
           'expected_center_gates': [8, 10, 12]},
 'val': {'type': 'BoxZTemporalDatasetV2',
         'current_only': False,
         'box_cache': 'data/geometry_v2/projection_fixed_20261112/box_pairs_c_20261116/val20',
         'expected_y_gates': [2, 2, 1.5],
         'expected_center_gates': [8, 10, 12]},
 'test': {'type': 'BoxZTemporalDatasetV2',
          'current_only': False,
          'box_cache': 'data/geometry_v2/projection_fixed_20261112/box_pairs_c_20261116/val20',
          'expected_y_gates': [2, 2, 1.5],
          'expected_center_gates': [8, 10, 12]}}
custom_hooks = [{'type': 'BoxZEpochHook'}]
# Previous B-gate outputs remain in box_z0; do not mix the two runs.
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/box_z0_c_20261116'
