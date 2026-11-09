"""Jointly train RadarOcc and P0 Z fusion from random initialization."""
_base_ = ['./RadarOcc_Small_box_z_p0_20pct.py']

model = dict(box_z_cfg=dict(freeze_original=False))
optimizer = dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01)
runner = dict(type='EpochBasedRunner', max_epochs=15)
load_from = None
resume_from = None
work_dir = 'work_dirs/box_z_p0_20pct_scratch_seed42/Z1'
