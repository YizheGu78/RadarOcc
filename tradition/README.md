# Traditional Radar Occupancy Baseline

This folder adds a **non-learning automotive-radar baseline** that produces the same 3-class 3D occupancy layout used by the current RadarOcc reproduction branch. It is isolated under `tradition/`; no RadarOcc neural-network module is modified.

## 1. Pipeline

RadarOcc avoids the conventional radar point-cloud/target-list bottleneck because CFAR discards weak returns that can be useful for full-scene occupancy. That makes a traditional target-list + occupancy-grid method a useful scientific baseline.

```text
K-Radar arrDREA [D,R,E,A]
        |
        v
Range-Doppler power map (max over E,A)
        |
        v
CA-CFAR in range AND Doppler
        |
        v
Local-maximum grouping / peak selection
        |
        v
Angle estimation from the E-A peak
        |
        v
Target list: range, Doppler, azimuth, elevation, power
        |
        v
Ego-speed compensation + Doppler static/dynamic split
        |
        v
3D inverse sensor model
  - free-space ray carving
  - occupied endpoint evidence
  - finite-resolution hit neighborhood
        |
        v
RadarOcc grid [128,128,14]
  0 = free
  1 = static/background occupied
  2 = dynamic/foreground occupied
        |
        v
<output>/<token>/pred_c.npy
rows = [z, y, x, class]
```

The user's `OGM_radar` fork is used as a design reference for inverse-sensor-model ideas (ray-based free cells and measurement uncertainty), generalized here to a 3D RadarOcc grid. The user's `OpenRadar` fork is supported through an adapter: `OpenRadarCACFAR` calls `mmwave.dsp.cfar.ca_` if that fork is installed. No source file from either external fork is vendored here.

## 2. SOLID structure

```text
tradition/
├── core/        # config, data types, interfaces, geometry
├── io/          # K-Radar reader and RadarOcc-format writer
├── detection/   # CFAR Strategy + target-list extraction
├── motion/      # Doppler/ego-motion static-dynamic classifier
├── mapping/     # 3D log-odds inverse sensor model
├── pipeline/    # dependency-injected orchestration
├── evaluation/  # RadarOcc-style IoU/mIoU helpers
├── cli/         # prediction/evaluation entry points
└── tests/       # focused unit tests
```

- **SRP:** reading, detection, motion classification, mapping, and serialization are separate.
- **OCP:** another CFAR detector, tracker, or mapper can be added behind interfaces.
- **LSP:** NumPy and OpenRadar CFAR backends satisfy the same `CFARBackend`.
- **ISP:** interfaces are small and task-specific.
- **DIP:** `TraditionalRadarPipeline` depends on abstractions; concrete objects are assembled in `build_default_pipeline()`.

## 3. Coordinate/output contract

Aligned to the RadarOcc K-Radar experiment:

- ROI: x `[0, 51.2)`, y `[-25.6, 25.6)`, z `[-2.6, 3.0)` m
- voxel size: `0.4 m`
- dense shape: `[X,Y,Z] = [128,128,14]`
- raw tensor: `arrDREA = [Doppler,Range,Elevation,Azimuth]`
- radar -> LiDAR translation: `[+2.54,-0.30,-0.70]` m, inverse of the transform used in `tools/filter_kradar_fov.py`
- sparse prediction: `[z,y,x,class]`, matching `tools/render_scene3_camera_prediction_gt.py`

Default angular bins are azimuth `-53..+53 deg` and elevation `-18..+18 deg` in 1-degree steps. If calibration differs, change `KRadarConfig`; the reader rejects unexpected tensor shapes rather than silently reinterpret bins.

## 4. Run

From the RadarOcc repository root:

```bash
PYTHONPATH=$PWD python -m tradition.cli.predict \
  --input /path/to/K-Radar/3/radar_tesseract \
  --output work_dirs/traditional_occ \
  --cfar-backend numpy \
  --ego-speed-mps 0.0
```

Use the OpenRadar fork directly:

```bash
pip install -e /path/to/OpenRadar
PYTHONPATH=$PWD python -m tradition.cli.predict \
  --input /path/to/K-Radar/3/radar_tesseract \
  --output work_dirs/traditional_occ \
  --cfar-backend openradar \
  --ego-speed-mps 0.0
```

Evaluate one prediction/GT pair:

```bash
PYTHONPATH=$PWD python -m tradition.cli.evaluate \
  --prediction work_dirs/traditional_occ/<lidar_token>/pred_c.npy \
  --ground-truth /path/to/occupancy_gt_with_semantic_fov.npy \
  --gt-order xyz
```

## 5. Experimental cautions

### Tune CFAR on validation, not test

Guard/noise lengths and the +6 dB threshold offset in `CFARConfig` are starting values, not constants claimed by RadarOcc or the survey paper. Tune them on the validation split before reporting the traditional baseline.

### Supply synchronized ego speed

For moving ego vehicles, stationary targets normally have non-zero radial velocity. The classifier compares measured Doppler with

`v_static ~= sign * v_ego * cos(elevation) * cos(azimuth)`

and wraps the residual with K-Radar's 3.84 m/s Doppler period. `--ego-speed-mps 0` on a moving sequence will over-label dynamic cells. For a full dataset run, pass frame-synchronized CAN/pose speed from a small caller around `TraditionalRadarPipeline.predict_file()`.

### Classical dynamic != RadarOcc semantic foreground

This baseline defines class 2 by residual Doppler. RadarOcc's class 2 represents semantic foreground, so a parked car can be foreground for RadarOcc but static here. For thesis reporting, binary occupied/free IoU is therefore the cleanest traditional-vs-RadarOcc comparison; BG/FG should be reported with this semantic mismatch explicitly stated.
