"""Shared Z0-Z3 screening configuration. P0=user name for existing P1 entry."""
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py']

model = dict(type='RadarOccBoxZ', temporal_cfg=None,
    occ_encoder_neck=dict(checkpoint_use_reentrant=False),
    box_z_cfg=dict(mode='Z1', bins_path='data/K-Radar-official-meta/resources/info_arr.mat',
        freeze_original=True, enabled=True, dim=64, heads=4, samples=8, chunk_size=2048,
        lambda_scale=.001, lambda_shift=.001, min_interp_weight=1e-4,
        diagnostic_path='work_dirs/box_z_p0_20pct/Z1/eval_diagnostics.jsonl'))
data = dict(workers_per_gpu=0, samples_per_gpu=1,
    train=dict(type='BoxZTemporalDataset', current_only=False,
        ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl',
        box_cache='data/box_z_pairs/train20'),
    val=dict(type='BoxZTemporalDataset', current_only=False,
        ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl',
        box_cache='data/box_z_pairs/val20'),
    test=dict(type='BoxZTemporalDataset', current_only=False,
        ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl',
        box_cache='data/box_z_pairs/val20'))
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=.01)
runner = dict(type='EpochBasedRunner', max_epochs=10)
evaluation = dict(interval=1, save_best='SSC_mean', rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
custom_hooks = [dict(type='BoxZEpochHook')]
load_from = None
resume_from = None
work_dir = 'work_dirs/box_z_p0_20pct/Z1'
