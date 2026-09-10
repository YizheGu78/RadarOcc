# Classical Object-Aware Radar Occupancy Baseline

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
    -> pose-align a causal five-frame window
    -> persistent stationary points: 2D OGM + 8-connected components
       persistent moving points: scaled XYZ DBSCAN
    -> cluster geometry/power/compensated-residual features
    -> trained classical Random Forest objectness
    -> object cluster: foreground; other occupied structure: background
    -> current free rays + semantically classified temporal endpoints
    -> 0=free, 1=background, 2=foreground
```

Motion is proposal evidence, not the semantic definition. A parked vehicle can
therefore be foreground and a stationary guardrail can remain background. The
classifier is a non-neural Random Forest trained on hand-crafted cluster
features. RadarOcc train GT is used only to construct training labels and is
never read by the inference classifier.

Detections beyond the evaluation AABB still carve free space up to the grid
boundary but do not create an occupied endpoint outside the grid.

## Train the objectness model once

Install `scikit-learn` and `joblib` in `radarocc-vis`; use the same environment
for training and video inference so the serialized estimator stays compatible.

```bash
./train_traditional_object_classifier.sh
```

The default model is written to:

```text
work_dirs/traditional_object_classifier/object_random_forest.joblib
```

The default is `kradar_dict_train_official_doppler8.pkl`. Do not train on
`test_official`; that would leak test labels into the classifier.

Small training smoke test:

```bash
MAX_FRAMES=100 \
OUTPUT_MODEL=$PWD/work_dirs/traditional_object_classifier/smoke.joblib \
./train_traditional_object_classifier.sh
```

## 42-D cluster features and retraining

Training and inference both call `RadarObjectFeatureExtractor.extract` through
`DualBranchCandidateExtractor`. `FEATURE_NAMES` is the ordered schema: the
original 28 entries retain their positions and the following 14 are appended.
These are cluster-level inputs to the binary objectness Random Forest, not
new output classes; occupancy output remains free/background/foreground.

| Appended feature | Definition |
| --- | --- |
| `range_compensated_point_count` | Number of candidate points times their mean measured radar range, `N * mean(range_m)` |
| `power_span` | Maximum minus minimum RPC power (in the supplied power scale, not an amplitude/dB conversion) |
| `oriented_bbox_perimeter_m` | Perimeter of the minimum-area enclosing rectangle of the XY convex hull, allowing rotation |
| `max_line_deviation_m` | Mean perpendicular distance of all XY points to the infinite line through their farthest pair |
| `compactness_m` | Mean Euclidean XY distance to the arithmetic XY centroid |
| `major_doppler_spread_ratio` | Point-coordinate span along the largest-covariance-eigenvalue axis divided by `residual_span + 0.001 m/s` |
| `minor_doppler_spread_ratio` | Corresponding span along the smallest-eigenvalue axis, using the same denominator |
| `range_doppler_correlation` | Signed Pearson correlation of measured radar range and compensated wrapped Doppler residual |
| `z_mean_m` | Mean aligned Cartesian Z coordinate |
| `z_std_m` | Population standard deviation of aligned Z (`ddof=0`) |
| `z_min_m` | Minimum aligned Z coordinate |
| `z_max_m` | Maximum aligned Z coordinate |
| `elevation_mean_rad` | Mean original measured radar elevation angle, in radians |
| `elevation_std_rad` | Population standard deviation of original measured elevation (`ddof=0`) |

Geometry and Z use `xyz_lidar_m` in the **current** LiDAR/vehicle frame after
pose alignment, not world height or height above an estimated road plane.
Range/elevation remain the observations at each source frame's radar pose,
including for historic static points. They are not recomputed from aligned
XYZ. Both training and inference use this convention. Candidate point counts
include any accepted static history, as in the original features.

The caller already replaces `radial_velocity_mps` with the ego-compensated,
wrapped residual before feature extraction. The two principal-axis spans are
computed by projecting the points onto the XY covariance eigenvectors; they
are not the lengths of a fitted confidence ellipse. Their ratios have units
of seconds and are descriptors, not target velocities. The added `0.001 m/s`
regularizer prevents division by zero but can still yield large ratios for
almost constant residuals. It does not unwrap Doppler ambiguities or change
the motion thresholds. Correlation is zero for singleton or constant data.
Single/duplicate XY points have zero geometric spans; collinear points have
zero area and a rectangle perimeter of twice the segment length.

The descriptors are adaptations of feature ideas, not an exact reproduction
of a published 50-D schema. Additional inputs are not guaranteed to improve
accuracy; compare with the 28-D baseline on the same held-out sequences.

**Old 28-D model files cannot be used with the 42-D extractor.** Loading an old
schema raises an explicit retraining error. The training command regenerates
features from RPC inputs, logs the dimension, and stores `feature_names` plus
`metadata.feature_count` in the new bundle. To keep the old model for baseline
comparisons, train to a different path:

```bash
OUTPUT_MODEL="$PWD/work_dirs/traditional_object_classifier/object_random_forest_42d.joblib" \
./train_traditional_object_classifier.sh
```

For the video script, select it with `OBJECT_MODEL`; for the Python inference
CLI, pass the new path via `--object-model`. Use only the official training
split to fit the model, and do not use test GT to choose features/thresholds.

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
  --object-model work_dirs/traditional_object_classifier/object_random_forest.joblib \
  --static-residual-threshold-mps 0.50 \
  --dynamic-residual-threshold-mps 0.80 \
  --temporal-window 5 \
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
├── semantics/
│   ├── object_classifier.py           # OGM/DBSCAN, features, RF inference
│   ├── cluster_geometry.py            # rotated rectangle and diameter-line geometry
│   └── training.py                    # train-GT cluster labels and RF fit
├── mapping/
│   └── temporal_occupancy_grid_3d.py  # semantic-history log-odds OGM
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
- Early outputs are warm-up frames: required support is reduced to the number
  of frames currently available until the five-frame window is full.
- The Random Forest must be trained only on the official training split. GT is
  not loaded during validation/test inference.
- Foreground means a RadarOcc object category, not simply a moving return;
  background means occupied environment structure, not simply a static return.
- Unobserved cells are class 0/free because the requested evaluation has only
  three output classes and no unknown class.
