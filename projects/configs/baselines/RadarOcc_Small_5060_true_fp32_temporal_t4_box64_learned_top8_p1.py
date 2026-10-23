"""Large symmetric boxes -> frozen pretrained ranker Top-8 -> existing attention."""
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py']

model = dict(temporal_cfg=dict(attention=dict(selection=dict(
    method='learned', frozen=True,
    checkpoint='work_dirs/p1_top8_ranker/best.pth',
    candidate_k=64, top_k=8, x_half_by_history=[4., 6., 8.],
    y_half=1.5, z_half=1., query_chunk_size=512,
))))

work_dir = 'work_dirs/radarocc_small_temporal_t4_box64_learned_top8_p1'
