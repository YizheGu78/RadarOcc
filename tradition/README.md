# Traditional Radar Occupancy Baseline

This folder contains the non-learning CFAR + Doppler + 3D occupancy-grid baseline for the K-Radar/RadarOcc experiment.

## Important: normal experiments do NOT save prediction NPY files

RadarOcc needs persistent `pred_c.npy`/dense tensors because a learned model has a separate train/inference/evaluation workflow. The traditional pipeline does not need that separation. Its occupancy grid can be evaluated and rendered immediately in memory.

The recommended flow is now:

```text
raw K-Radar arrDREA
        |
        v
CFAR + peak extraction
        |
        v
Doppler static/dynamic classification
        |
        v
3D occupancy grid (RAM only)
        |----------------------|
        v                      v
RadarOcc metrics          overlay frame
        |                      |
        v                      v
CSV / MD / PNG            camera + occupancy
                               |
                               v
                            MP4 / GIF
```

No per-frame traditional prediction file is written by `tradition.cli.run`.
Temporary PNG video frames are removed after ffmpeg finishes unless `--keep-frames` is given.

The older `tradition/io/radarocc_writer.py` remains only as an optional compatibility/debug adapter. It is not used by the direct experiment runner.

## Output

A normal run creates only human-facing experiment artifacts:

```text
<output-dir>/
├── traditional_metrics.csv
├── traditional_metrics.md
├── traditional_metrics.png
├── scene_camera_simple_overlay.mp4   # when --video-scene is used
└── scene_camera_simple_overlay.gif   # when --video-scene is used
```

The table contains the same four metric rows used in the RadarOcc comparison:

- SC IoU
- SSC mIoU
- Background IoU
- Foreground IoU

at 12.8 m, 25.6 m and 51.2 m. Metrics are accumulated over the selected dataset before IoU is computed; they are not naive averages of per-frame IoUs.

## Recommended test command

From the RadarOcc repository root:

```bash
PYTHONPATH=$PWD python -m tradition.cli.run \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --radar-root /path/to/K-Radar \
  --output-dir work_dirs/traditional_direct \
  --cfar-backend numpy \
  --ego-speed-mps 0.0
```

To additionally create the same camera + Prediction/GT overlay style already used by `tools/render_scene3_camera_prediction_gt.py`:

```bash
PYTHONPATH=$PWD python -m tradition.cli.run \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --radar-root /path/to/K-Radar \
  --output-dir work_dirs/traditional_direct \
  --cfar-backend numpy \
  --video-scene 3 \
  --camera-dir /path/to/K-Radar/3/cam-front \
  --max-video-frames 100 \
  --fps 10
```

For the OpenRadar backend:

```bash
pip install -e /path/to/OpenRadar
```

then change `--cfar-backend numpy` to `--cfar-backend openradar`.

## Architecture / SOLID

```text
tradition/
├── core/           # config, types, interfaces, geometry
├── io/             # raw reader; optional compatibility writer
├── detection/      # CFAR strategy + target extraction
├── motion/         # ego-compensated Doppler classifier
├── mapping/        # 3D log-odds inverse sensor model
├── pipeline/       # dependency-injected per-frame orchestration
├── evaluation/     # dataset-level RadarOcc metrics
├── reporting/      # CSV/Markdown/PNG result table
├── visualization/  # adapter to existing RadarOcc video style
├── experiment/     # direct dataset runner
├── cli/            # command-line entry points
└── tests/
```

The direct runner consumes `FramePrediction` in memory. Detection, motion classification, mapping, metrics, reporting and visualization remain separate modules.

## Coordinate/output contract

- ROI: x `[0, 51.2)`, y `[-25.6, 25.6)`, z `[-2.6, 3.0)` m
- voxel size: 0.4 m
- in-memory dense grid: `[X,Y,Z] = [128,128,14]`
- classes: 0 free, 1 static/background, 2 dynamic/foreground
- GT default coordinate order: `xyz`; override with `--gt-order zyx` if required by a specific generated GT set

## Experimental cautions

CFAR guard/noise/threshold parameters are validation parameters and must be frozen before final test reporting.

For a moving ego vehicle, `--ego-speed-mps 0` is not a valid static/dynamic compensation. A synchronized ego-speed source should be supplied for the final experiment.

Traditional dynamic/static partition is motion-derived, whereas RadarOcc foreground/background is semantic. Therefore SC IoU is the cleanest primary traditional-vs-RadarOcc comparison; SSC Background/Foreground results should be interpreted with this mismatch stated explicitly.
