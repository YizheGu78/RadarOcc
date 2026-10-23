"""Matched baseline: same boxes, pool and attention; geometric-distance Top-8."""
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_box64_learned_top8_p1.py']

model = dict(temporal_cfg=dict(attention=dict(selection=dict(
    method='distance', checkpoint=None,
))))

work_dir = 'work_dirs/radarocc_small_temporal_t4_box64_distance_top8_p1'
