"""Shared reproduction recipe: 20% contiguous windows, scratch, seed42, 15 epochs."""
_base_ = ['../../projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py']
plugin = True
plugin_dir = 'temporal/experiments_v2/plugin/'
model = dict(occ_encoder_neck=dict(checkpoint_use_reentrant=False))
data = dict(workers_per_gpu=0, samples_per_gpu=1,
    train=dict(ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'),
    val=dict(ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'),
    test=dict(ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'))
optimizer = dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01)
runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(interval=1, save_best='SSC_mean', rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
custom_hooks = []
find_unused_parameters = True
load_from = None
resume_from = None
seed = 42
# Saved in config + checkpoint metadata to refuse accidental old-projection resume.
projection_version = 'small_physical_model_audit_20261112'
experiment_series = 'projection_fixed_20261112'
radar_coordinate_convention = 'kradar_physical_v2'
checkpoint_config['meta'] = dict(projection_version=projection_version,
                                experiment_series=experiment_series,
                                radar_coordinate_convention=radar_coordinate_convention)
