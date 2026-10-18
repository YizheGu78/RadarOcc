_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_unaligned_memory_p1.py']
data = dict(
    train=dict(ann_file='data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl'),
    val=dict(ann_file='data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl'),
)
# test inherits the complete official test split, as in the existing 20% configs.
work_dir = 'work_dirs/radarocc_small_temporal_t4_unaligned_memory_p1_20pct'
