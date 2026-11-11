"""Shared 20% scratch-training recipe; all original configs remain untouched."""
_base_ = ['../projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py']

# Both existing train.py and test.py import this package via plugin_dir.
plugin = True
plugin_dir = 'temporal/p0_plugin/'
model = dict(type='RadarOccP0Temporal', temporal_cfg=None,
             occ_encoder_neck=dict(checkpoint_use_reentrant=False))
data = dict(workers_per_gpu=0, samples_per_gpu=1,
    train=dict(current_only=False,
        ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'),
    val=dict(current_only=False,
        ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'),
    test=dict(current_only=False,
        ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'))
optimizer = dict(_delete_=True, type='AdamW', lr=3e-4, weight_decay=.01)
runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(interval=1, save_best='SSC_mean', rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
custom_hooks = []
find_unused_parameters = True
load_from = None
resume_from = None
seed = 42
