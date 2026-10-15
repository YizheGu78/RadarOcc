# Full-data RA-ADTA. The original ego_attention (KNN) config is retained.
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py']

model = dict(temporal_cfg=dict(
    mode='ra_adta',
    attention=dict(
        _delete_=True,  # Do not inherit KNN-only radius_m/neighbors_per_frame.
        channels=8, embed_dims=32, num_heads=4,
        sampling_points_per_head=2, candidate_pool_size=24,
        alpha_range=3.0, alpha_azimuth=3.0, alpha_elevation=3.0,
        min_angular_scale_m=0.25, offset_bound=0.5,
        temperature=0.5, interpolation_cutoff=2.0,
        chunk_size=256, search_chunk_size=256, checkpoint_chunks=True,
        time_scale_frames=3.0, max_log_correction=0.5,
    ),
))
load_from = None
resume_from = None
work_dir = 'work_dirs/radarocc_small_temporal_t4_ra_adta_p1'
