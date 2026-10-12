# Optional self-region power-zeroing experiment. Original full-input config retained.
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py']

model = dict(pts_middle_encoder=dict(allow_zero_power=True),
             temporal_cfg=dict(
    power_mask=dict(enabled=True, radius_m=3.7),
))
load_from = None
resume_from = None
work_dir = 'work_dirs/radarocc_small_temporal_t4_ego_attention_p1_mask37'
