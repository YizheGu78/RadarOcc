# T=4 Sequential Dataset + current-only regression for Hybrid125.
#
# Use with checkpoints trained from:
#   RadarOcc_Small_5060_true_fp32_hybrid125.py
#
# No temporal fusion is enabled; the model still receives only current frame t.

_base_ = [
    './RadarOcc_Small_5060_true_fp32_hybrid125.py',
]

data = dict(
    train=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_train_official_temporal_hybrid125.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
    val=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_val_official_temporal_hybrid125.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
    test=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_test_official_temporal_hybrid125.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
)
