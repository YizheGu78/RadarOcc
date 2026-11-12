"""Historical descriptor memory control on the same corrected Small backbone."""
_base_ = ['../base.py']
model = dict(type='RadarOccSmallUnalignedTemporal', temporal_cfg=None,
             unaligned_temporal=dict(
                 channels=8, embed_dims=64, num_heads=4, num_layers=2,
                 memory_tokens=256, chunk_size=1024, ffn_ratio=2,
                 coord_scales=(256., 107., 37.), time_scale_frames=3.,
                 max_log_correction=0.5))
data = dict(train=dict(type='UnalignedTemporalKRadarDataset', current_only=False),
            val=dict(type='UnalignedTemporalKRadarDataset', current_only=False),
            test=dict(type='UnalignedTemporalKRadarDataset', current_only=False))
work_dir = 'work_dirs/geometry_v2/projection_fixed_20261112/unaligned_memory'
