# Geometry debug only: retains exactly the identity P1 feature output.
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_identity_p1.py']
model = dict(temporal_cfg=dict(
    alignment=dict(bins_path='data/radar_geometry.npz', azimuth_sign=1, elevation_sign=1),
    debug=dict(enabled=True, save_every=50, output_dir='outputs/temporal_alignment_debug'),
))
