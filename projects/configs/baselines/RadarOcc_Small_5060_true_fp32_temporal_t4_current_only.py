# T=4 Sequential Dataset + current-only RadarOcc regression config.
#
# Purpose:
#   Verify that the new temporal frame-level PKL and sequence bookkeeping do
#   not change the established single-frame RadarOcc inference path.
#
# IMPORTANT:
#   - No temporal fusion is enabled here.
#   - No retraining is required.
#   - Use the SAME checkpoint as the original RadarOcc baseline.
#   - TemporalKRadarDataset builds [t-3,t-2,t-1,t] indices but only t is
#     passed through the original RadarOcc pipeline.

_base_ = [
    './RadarOcc_Small_5060_true_fp32.py',
]

data = dict(
    train=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_train_official_temporal_doppler8.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
    val=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_val_official_temporal_doppler8.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
    test=dict(
        type='TemporalKRadarDataset',
        ann_file=(
            'data/annotations/'
            'kradar_dict_test_official_temporal_doppler8.pkl'
        ),
        frame_nums=4,
        current_only=True,
    ),
)
