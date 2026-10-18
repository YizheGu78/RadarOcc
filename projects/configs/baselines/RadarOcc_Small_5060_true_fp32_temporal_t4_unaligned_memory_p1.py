# Opt-in unaligned model. All existing files, configs and commands are unchanged.
_base_ = ['./RadarOcc_Small_5060_true_fp32_temporal_t4_identity_p1.py']

plugin = True
plugin_dir = 'projects/unaligned_temporal/'
model = dict(
    type='RadarOccSmallUnalignedTemporal',
    temporal_cfg=None,
    occ_encoder_neck=dict(checkpoint_use_reentrant=False),
    unaligned_temporal=dict(
        channels=8, embed_dims=64, num_heads=4, num_layers=2,
        memory_tokens=256, chunk_size=1024, ffn_ratio=2,
        coord_scales=(256., 107., 37.), time_scale_frames=3.,
        max_log_correction=0.5,
    ),
)
data = dict(
    workers_per_gpu=0,
    train=dict(type='UnalignedTemporalKRadarDataset', current_only=False),
    val=dict(type='UnalignedTemporalKRadarDataset', current_only=False),
    test=dict(type='UnalignedTemporalKRadarDataset', current_only=False),
)
# Same from-scratch policy as the existing KNN/RA-ADTA comparison configs.
# For fine-tuning, explicitly pass --cfg-options load_from=/path/to/single_frame.pth.
load_from = None
resume_from = None
work_dir = 'work_dirs/radarocc_small_temporal_t4_unaligned_memory_p1'
find_unused_parameters = True
# Existing single-frame forward_dummy efficiency hook has no temporal payload.
custom_hooks = []
