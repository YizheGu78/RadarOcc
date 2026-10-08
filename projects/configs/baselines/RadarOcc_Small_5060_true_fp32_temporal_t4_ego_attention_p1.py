_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_identity_p1.py']

model = dict(temporal_cfg=dict(
    _delete_=True,
    type='TemporalPlugin', mode='ego_attention', position='P1',
    alignment=dict(bins_path='data/K-Radar-official-meta/resources/info_arr.mat',
                   azimuth_sign=1, elevation_sign=1),
    attention=dict(channels=8, embed_dims=32, num_heads=4,
                   neighbors_per_frame=8, radius_m=2.0, chunk_size=1024,
                   time_scale_frames=3.0, max_log_correction=0.5),
    debug=dict(enabled=False),
))

# Start a fresh optimizer/schedule while loading the validated single-frame weights.
load_from = 'work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth'
resume_from = None
work_dir = 'work_dirs/radarocc_small_temporal_t4_ego_attention_p1'
# Beginning-of-scene samples can have no valid history.
find_unused_parameters = True
