"""P0 sparse temporal queries + frozen learned Top-8; fixed 20% split."""
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_box64_learned_top8_p1.py']

model = dict(temporal_cfg=dict(query_budget=dict(
    per_range=64, doppler_sign='both_consensus', threshold_mps=.2,
    max_peak_std=.2, min_concentration=.8,
    class_fractions=[.5, .25, .25], angular_cells=[8, 4],
    doppler_bins='data/K-Radar-official-meta/resources/arr_doppler.mat',
    max_ego_dt=1., max_ego_speed=70.,
)))
data = dict(
    train=dict(ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'),
    val=dict(ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'),
    workers_per_gpu=0,
)
runner = dict(type='EpochBasedRunner', max_epochs=15)
evaluation = dict(interval=1, save_best='SSC_mean', rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
load_from = None
resume_from = None
work_dir = 'work_dirs/temporal_query_budget_20pct/Q64'
