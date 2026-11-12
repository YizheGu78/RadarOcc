"""Scratch run on the corrected Small projection; previous artifacts retained."""
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
               'diagnostic_path': 'work_dirs/geometry_v2/projection_fixed_20261112/box_z0/eval_diagnostics.jsonl'}}
data = {'train': {'type': 'BoxZTemporalDatasetV2',
           'current_only': False,
           'box_cache': 'data/geometry_v2/box_pairs/train20'},
 'val': {'type': 'BoxZTemporalDatasetV2',
         'current_only': False,
         'box_cache': 'data/geometry_v2/box_pairs/val20'},
 'test': {'type': 'BoxZTemporalDatasetV2',
          'current_only': False,
          'box_cache': 'data/geometry_v2/box_pairs/val20'}}
custom_hooks = [{'type': 'BoxZEpochHook'}]
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/box_z0'
