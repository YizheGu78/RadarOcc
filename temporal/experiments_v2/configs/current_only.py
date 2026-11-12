"""Scratch run on the corrected Small projection; previous artifacts retained."""
_base_ = ['../base.py']
model = {'temporal_cfg': None}
data = {'train': {'current_only': True}, 'val': {'current_only': True}, 'test': {'current_only': True}}
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/current_only'
