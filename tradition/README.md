# Non-learning Radar Occupancy Baseline

This folder implements a traditional, non-learning occupancy baseline for
K-Radar/RadarOcc. Enhanced K-Radar RPC/pc01p point clouds are the default input.
The original 4DRT + CFAR path remains an optional independent-frame strategy.

## Default pose-temporal RPC flow

Each Enhanced K-Radar NPY row must contain:

```text
[x, y, z, power, doppler, range, azimuth, elevation,
 range_index, azimuth_index, elevation_index]
```

Azimuth/elevation are radians and Doppler is already physical radial velocity
in m/s. The code uses Cartesian position and Doppler directly; it does not
repeat CFAR, polar conversion, or Doppler-bin conversion. It also reads
RadarOcc's synchronized `lidar_ego_pose{i}.npy` for every LiDAR token. Pose
indices follow `lidar_token`, not the differently numbered aligned RPC file.

```text
RPC [N,11]
    -> local polar-neighbour power/reliability filtering
    -> adjacent LiDAR poses produce vx, vy and yaw rate
    -> full-vector ego-motion-compensated wrapped Doppler residual
    -> pose-align causal [t-2, t-1, t] points into frame t
    -> 2/3 static support: background
       2/3 dynamic support: foreground
       uncertain/isolated returns: discard
    -> current free rays + current endpoints + historic static endpoints
    -> 0=free, 1=background, 2=foreground
```

Classification fuses pose-compensated wrapped Doppler with pose-aligned
temporal persistence. It does not use DBSCAN/geometric object clustering,
object-shape heuristics, a learned classifier, GT boxes, or RadarOcc neural
predictions. The blue video background comes only from the traditional
temporal OGM.

Detections beyond the evaluation AABB still carve free space up to the grid
boundary but do not create an occupied endpoint outside the grid.

## Run Scene 3

Smoke test:

```bash
cd /home/user1/projects/RadarOcc
export PYTHONPATH=$PWD

python -m tradition.cli.run \
  --annotation data/annotations/kradar_dict_test_official_doppler8.pkl \
  --radar-root data/K-Radar_rpc \
  --pose-root data/K-RadarOcc \
  --pose-dt-s 0.10 \
  --output-dir work_dirs/tradition_rpc_pose_smoke \
  --scene 3 \
  --max-frames 3 \
  --static-residual-threshold-mps 0.50 \
  --dynamic-residual-threshold-mps 0.80 \
  --temporal-window 3 \
  --min-static-support 2 \
  --min-dynamic-support 2
```

Complete official-test Scene 3 metrics and video:

```bash
./run_traditional_scene3_video.sh
```

Three-frame video smoke test:

```bash
OUTPUT_DIR=$PWD/work_dirs/tradition_scene3_pose_smoke \
MAX_FRAMES=3 MAX_VIDEO_FRAMES=3 KEEP_FRAMES=1 \
./run_traditional_scene3_video.sh
```

The RPC resolver uses `radar_frame_idx` from the aligned annotation first,
then the frame number in `radar_path`. Common layouts include:

```text
data/K-Radar_rpc/3/rpc_00042.npy
data/K-Radar_rpc/3/pc01p/rpc_00042.npy
data/K-Radar_rpc/3/pc01p_00042.npy
```

The pose resolver accepts a sequence pose directory, a split root, or the
whole K-RadarOcc root, for example:

```text
data/K-RadarOcc/train/3/pose/lidar_ego_pose0.npy
```

## Legacy independent-frame prediction

```bash
python -m tradition.cli.predict \
  --input data/K-Radar_rpc/3 \
  --output work_dirs/tradition_rpc_predictions_seq3 \
  --ego-speed-mps auto
```

`tradition.cli.predict` is kept as a single-frame compatibility path; it does
not perform pose-temporal fusion. Each output frame contains:

- `pred_dense.npy`: dense `[128,128,14]` uint8 grid;
- `pred_c.npy`: occupied coordinates and class labels;
- `meta.json`: input path, component names, and class convention.

Evaluation reports RadarOcc-compatible `SC IoU`, `SSC mIoU (BG+FG)`,
`Background IoU`, and `Foreground IoU`. It additionally reports `Free IoU` and
`3-class mIoU` (free/background/foreground).

## Optional original 4DRT + CFAR strategy

```bash
python -m tradition.cli.run \
  --annotation "$ANN" \
  --radar-root /path/to/K-Radar \
  --output-dir work_dirs/traditional_raw_cfar \
  --input-mode raw \
  --cfar-backend numpy
```

## Main modules

```text
tradition/
├── core/                              # types, configuration, interfaces
├── io/
│   ├── rpc_radar_reader.py            # Enhanced K-Radar [N,11]
│   ├── pose_reader.py                 # token -> LiDAR-to-world pose
│   └── kradar_reader.py               # optional raw 4DRT
├── detection/
│   ├── rpc_target_detector.py         # direct point-list adapter
│   ├── rpc_reliability_filter.py      # local power/sidelobe gate
│   └── cfar.py                        # optional raw CFAR
├── motion/
│   ├── pose_ego_motion.py             # pose -> velocity and yaw rate
│   ├── doppler_classifier.py          # wrapped residual evidence
│   └── temporal_consistency.py        # pose-aligned persistence
├── mapping/
│   └── temporal_occupancy_grid_3d.py  # static-history log-odds OGM
├── pipeline/
├── evaluation/
├── experiment/
└── cli/
```

## Experimental cautions

- RPC/pc01p is already a density-reduced representation. Describe this as an
  **Enhanced K-Radar RPC pose-temporal traditional OGM baseline**, not as an
  original-4DRT CFAR method.
- Frame 0 uses pose 0->1 (forward difference); later frames use t-1->t (causal
  backward difference). The default interval is 0.10 s (10 Hz).
- Doppler is periodic with 3.84 m/s. Residuals are wrapped into
  `[-1.92, 1.92)`. The supplied RPC convention uses stationary projection sign
  `+1`; use `DOPPLER_SIGN=-1` only for an export with the opposite convention.
- Residuals `<=0.50 m/s` are static evidence, `>=0.80 m/s` are dynamic
  evidence, and values in the dead band are discarded.
- The first two outputs are warm-up frames: required support is reduced to the
  number of frames currently available until the three-frame window is full.
- Static parked objects remain background. This is a motion/background split,
  not object-level semantic recognition.
- Unobserved cells are class 0/free because the requested evaluation has only
  three output classes and no unknown class.
