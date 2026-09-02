# Non-learning Radar Occupancy Baseline

This folder implements a traditional, non-learning occupancy baseline for K-Radar/RadarOcc. Enhanced K-Radar RPC/pc01p point clouds are the default input. The original 4DRT + CFAR path remains an optional alternative strategy.

## Default RPC flow

Each Enhanced K-Radar NPY row must contain:

```text
[x, y, z, power, doppler, range, azimuth, elevation,
 range_index, azimuth_index, elevation_index]
```

Azimuth/elevation are radians and Doppler is already physical radial velocity in m/s. The implementation therefore uses Cartesian position and Doppler directly; it does not repeat CFAR, polar conversion or Doppler-bin conversion.

```text
rpc_*.npy / pc01p_*.npy
        |
        v
validate [N,11] and remove invalid/zero-range points
        |
        v
robust per-frame ego-speed estimation in wrapped-Doppler space
        |
        v
ego-motion-compensated Doppler classification
        |
        v
direct static/background and dynamic/foreground mapping
        |
        v
3D free-ray carving + log-odds occupancy mapping
        |
        v
0=free, 1=background, 2=foreground
```

Compensated Doppler is the complete background/foreground rule: static
detections are background and dynamic detections are foreground. No spatial
clustering, object-shape heuristic, learned classifier or GT box is used.
Detections beyond the evaluation AABB still carve free space up to the grid
boundary, but do not create an occupied endpoint outside the grid.

## Run an aligned evaluation

```bash
cd /home/user1/projects/RadarOcc
export PYTHONPATH=$PWD

ANN=$PWD/data/annotations/kradar_dict_test_official_doppler8.pkl
RPC_ROOT=$PWD/data/K-Radar_rpc

python -m tradition.cli.run \\
  --annotation "$ANN" \\
  --radar-root "$RPC_ROOT" \\
  --output-dir work_dirs/tradition_rpc_smoke_seq3 \\
  --scene 3 \\
  --max-frames 1 \\
  --ego-speed-mps auto \\
  --static-residual-threshold-mps 0.30
```

RPC is the default input mode. Writing `--input-mode rpc` explicitly is optional. The resolver uses `radar_frame_idx` from the aligned annotation first, then the frame number in `radar_path`. Common layouts such as these are accepted:

```text
data/K-Radar_rpc/3/rpc_00042.npy
data/K-Radar_rpc/3/pc01p/rpc_00042.npy
data/K-Radar_rpc/3/pc01p_00042.npy
```

## Save per-frame predictions

```bash
python -m tradition.cli.predict \\
  --input data/K-Radar_rpc/3 \\
  --output work_dirs/tradition_rpc_predictions_seq3 \\
  --ego-speed-mps auto
```

Each output frame contains:

- `pred_dense.npy`: dense `[128,128,14]` uint8 grid;
- `pred_c.npy`: occupied coordinates and class labels;
- `meta.json`: input path, component names and class convention.

Evaluation reports the RadarOcc-compatible `SC IoU`, `SSC mIoU (BG+FG)`,
`Background IoU`, and `Foreground IoU`. It additionally reports `Free IoU` and
`3-class mIoU` (free/background/foreground). The extra mIoU has a separate name
so it cannot be confused with the paper's SSC mIoU definition.

## Optional original 4DRT + CFAR strategy

```bash
python -m tradition.cli.run \\
  --annotation "$ANN" \\
  --radar-root /path/to/K-Radar \\
  --output-dir work_dirs/traditional_raw_cfar \\
  --input-mode raw \\
  --cfar-backend numpy
```

## SOLID structure

- **Single responsibility:** readers parse measurements, detectors create target lists, the motion classifier supplies a Doppler cue, the semantic classifier assigns background/foreground, the mapper updates occupancy, and the writer serializes results.
- **Open/closed:** RPC and raw-CFAR are composed from the same interfaces; adding another radar representation does not require changing the mapper or evaluator.
- **Liskov substitution:** `KRadarRPCReader` and `KRadarTensorReader` implement the representation-neutral `RadarMeasurementReader` interface.
- **Interface segregation:** reader, detector, classifier, mapper and writer contracts remain separate and small.
- **Dependency inversion:** `TraditionalRadarPipeline` depends on these interfaces rather than concrete input formats.

```text
tradition/
├── core/
│   ├── interfaces.py                 # component contracts
│   ├── types.py                      # RPC frame, detection, prediction
│   ├── config.py
│   └── geometry.py
├── io/
│   ├── rpc_radar_reader.py           # default Enhanced K-Radar [N,11]
│   ├── kradar_reader.py              # optional raw 4DRT
│   └── radarocc_writer.py
├── detection/
│   ├── rpc_target_detector.py        # direct point-list adapter
│   ├── target_detector.py            # optional raw CFAR target list
│   └── cfar.py
├── motion/
├── semantics/                        # classical BG/FG classifier
├── mapping/
├── pipeline/
├── evaluation/
├── experiment/
└── cli/
```

## Experimental cautions

- RPC/pc01p is already a density-reduced detection-like representation. Describe it as an **Enhanced K-Radar RPC traditional OGM baseline**, not as original 4DRT + CFAR.
- Background/foreground is a direct compensated-Doppler split, not GT-box, geometric-cluster or learned semantic recognition. Report static parked objects as a known limitation because they remain background.
- The default `--ego-speed-mps auto` estimates speed independently for every frame from the dominant stationary Doppler consensus. Pass a number only when synchronized ego speed is available; `0` is suitable only for a smoke test.
- The default static/background residual threshold is `0.30 m/s`. Increase `--static-residual-threshold-mps` cautiously when static structures remain foreground after ego-motion compensation; a larger value also hides slow-moving targets.
- The requested three-class protocol has no unknown class, so unobserved OGM cells are collapsed into class 0/free.
