# Non-learning Radar Occupancy Baseline

This folder provides two non-learning radar occupancy modes for the K-Radar/RadarOcc experiment.

## Recommended mode for the current local dataset: `sparse`

The current local data already contains RadarOcc `EAsparse_*.npz` files produced by Doppler-mean, range-wise Top-K sparsification. The direct runner can now use those files without the original ~500 MB/frame 4DRT.

```text
EAsparse_*.npz
  [range, elevation, azimuth]
  + top-3 Doppler powers/indices
  + mean/variance
        |
        v
sparse candidate target-list extraction
(no CFAR claim)
        |
        v
strongest stored Doppler component
        |
        v
static / dynamic split
        |
        v
3D inverse-sensor-model occupancy grid
        |----------------------|
        v                      v
RadarOcc metrics          overlay/video
```

Important terminology: **EAsparse is not a CFAR output.** True CFAR compares a cell under test with locally estimated noise/reference cells. Those discarded cells are no longer present in EAsparse, so CA-CFAR/OS-CFAR cannot be reconstructed from the NPZ. The sparse mode should be described as a **RadarOcc Top-K sparse non-learning OGM baseline**, not as `CFAR + OGM`.

This mode is nevertheless useful because it starts from exactly the reduced radar representation already used by RadarOcc, avoids storing the original 4DRT, and isolates the effect of a non-learning target-list/Doppler/OGM pipeline from the neural occupancy model.

The stored `power_val` descriptor follows the existing generator:

- rows 0-2: top-3 Doppler powers;
- rows 3-5: matching Doppler-bin indices;
- row 6: mean Doppler power;
- row 7: Doppler variance.

Because the original Top-250 still contains many candidates, sparse mode has a configurable range-wise thinning parameter. Default is 16 candidates per range for a practical first baseline. `--sparse-max-per-range 250` uses every stored Top-250 candidate but is slower and normally much noisier. This parameter must be tuned on validation data and frozen before final test reporting.

## Sparse smoke test

```bash
cd /home/user1/projects/RadarOcc
export PYTHONPATH=$PWD

ANN=$PWD/data/annotations/kradar_dict_test_official_doppler8.pkl
SPARSE_ROOT=$PWD/data/RadarOcc_8doppler

python -m tradition.cli.run \
  --annotation "$ANN" \
  --radar-root "$SPARSE_ROOT" \
  --output-dir work_dirs/tradition_smoke_seq3 \
  --input-mode sparse \
  --scene 3 \
  --max-frames 1 \
  --sparse-max-per-range 16 \
  --ego-speed-mps 0.0
```

## Five requested sequences

```bash
for SEQ in 3 15 22 23 55
do
  python -m tradition.cli.run \
    --annotation "$ANN" \
    --radar-root "$SPARSE_ROOT" \
    --output-dir "work_dirs/tradition_test_seq${SEQ}" \
    --input-mode sparse \
    --scene "$SEQ" \
    --sparse-max-per-range 16 \
    --ego-speed-mps 0.0
done
```

## Optional raw-CFAR mode

If the full original 4DRT is available, the old genuine CFAR path remains available:

```bash
python -m tradition.cli.run \
  --annotation "$ANN" \
  --radar-root /path/to/K-Radar \
  --output-dir work_dirs/traditional_raw_cfar \
  --input-mode raw \
  --cfar-backend numpy
```

## Output

Normal experiments keep predictions in memory and write only human-facing artifacts:

```text
<output-dir>/
├── traditional_metrics.csv
├── traditional_metrics.md
├── traditional_metrics.png
├── scene_camera_simple_overlay.mp4   # when video is requested
└── scene_camera_simple_overlay.gif
```

The metrics are SC IoU, SSC mIoU, Background IoU and Foreground IoU at 12.8 m, 25.6 m and 51.2 m. Confusion matrices are accumulated across the selected dataset before IoU is calculated.

## Architecture

```text
tradition/
├── core/
├── io/
│   ├── kradar_reader.py              # raw 4DRT
│   └── sparse_radar_reader.py        # EAsparse NPZ
├── detection/
│   ├── cfar.py                       # raw mode only
│   ├── target_detector.py            # raw CFAR target list
│   └── sparse_target_detector.py     # sparse candidate target list
├── motion/
├── mapping/
├── pipeline/
├── evaluation/
├── reporting/
├── visualization/
├── experiment/
└── cli/
```

## Experimental cautions

- Sparse Top-K selection and CFAR are different signal-processing operations; do not label sparse-mode results as CFAR results.
- `--sparse-max-per-range` is a validation parameter. Do not choose it using the final test results.
- For a moving ego vehicle, `--ego-speed-mps 0` is only suitable for getting the pipeline running. Final static/dynamic evaluation requires synchronized ego-motion compensation.
- RadarOcc foreground/background is semantic, whereas this baseline's static/dynamic classes are motion-derived. SC IoU is therefore the cleanest primary occupancy comparison; class-wise IoUs require this mismatch to be stated explicitly.
