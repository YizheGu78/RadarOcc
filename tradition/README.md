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
ego-motion-compensated Doppler classification
        |
        v
3D free-ray carving + log-odds occupancy mapping
        |
        v
0=free, 1=background/static, 2=foreground/dynamic
```

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
  --ego-speed-mps 0.0
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
  --ego-speed-mps 0.0
```

Each output frame contains:

- `pred_dense.npy`: dense `[128,128,14]` uint8 grid;
- `pred_c.npy`: occupied coordinates and class labels;
- `meta.json`: input path, component names and class convention.

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

- **Single responsibility:** readers parse measurements, detectors create target lists, the motion classifier assigns static/dynamic labels, the mapper updates occupancy, and the writer serializes results.
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
├── mapping/
├── pipeline/
├── evaluation/
├── experiment/
└── cli/
```

## Experimental cautions

- RPC/pc01p is already a density-reduced detection-like representation. Describe it as an **Enhanced K-Radar RPC traditional OGM baseline**, not as original 4DRT + CFAR.
- Class 1/2 is a motion-derived static/dynamic proxy, while RadarOcc background/foreground is semantic. Parked vehicles are the main mismatch.
- `--ego-speed-mps 0` is suitable only for a smoke test. Final motion labels require synchronized ego speed; otherwise static structures may be predicted as foreground.
- The requested three-class protocol has no unknown class, so unobserved OGM cells are collapsed into class 0/free.
