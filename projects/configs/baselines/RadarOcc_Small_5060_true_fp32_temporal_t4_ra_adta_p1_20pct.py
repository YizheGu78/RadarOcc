# Same fixed 20% train/val PKLs and initialization as the KNN screening run.
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_ra_adta_p1.py']
data = dict(
    train=dict(ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'),
    val=dict(ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'),
)
work_dir = 'work_dirs/radarocc_small_temporal_t4_ra_adta_p1_20pct'
