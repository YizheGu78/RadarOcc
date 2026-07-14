_base_ = ['./RadarOcc_Small.py']

# Strict official RadarOcc-S execution path.
# The base config keeps the released Small-model architecture and local data
# paths. Only the registered detector/head classes are switched so the strict
# 800 -> 250 token selection and original FP16/loss behavior are used.
model = dict(
    type='RadarOcc_small_strict',
    pts_bbox_head=dict(
        type='OccHeadStrict',
        fp16=True,
    ),
)

work_dir = 'work_dirs/radarocc_small_5060_strict'
