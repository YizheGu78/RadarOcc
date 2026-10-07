# DG-STF Stage 1: real T=4 loading + P1 Adapter + Identity Temporal Plugin.
#
# This stage MUST reproduce the single-frame RadarOcc baseline without
# retraining.  Historical frames are loaded and reach P1, but identity mode
# returns only the current-frame P1 features.

_base_ = [
    './RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py',
]

model = dict(
    temporal_cfg=dict(
        type='identity',
        mode='identity',
        position='P1',
    ),
)

data = dict(
    train=dict(
        current_only=False,
    ),
    val=dict(
        current_only=False,
    ),
    test=dict(
        current_only=False,
    ),
)
