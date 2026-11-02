# P1 reflection-region inspection

This is a GT-free, single-frame diagnostic. It uses the exact original NPZ
membership of Small P1 (first 175 ranges x 250 returns), stored power channel 2,
and the existing real-bin spherical-to-XYZ conversion. No checkpoint, PyTorch,
MMCV, CUDA, label boxes, object IDs, or occupancy forward is required.

## Run

From the project root in the existing environment:

```bash
source use_radarocc5060.sh
python tools/temporal_vis/vis_p1_reflection_regions.py \
  --scene 3 --start-order 80 --end-order 200 --stride 5 \
  --camera-dir data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched \
  --output-dir outputs/p1_reflection_regions/scene3 --gif
```

For one frame, set `--start-order 80 --end-order 80`. By default the tool finds
the scene in existing full train/val/test temporal PKLs under data/annotations
and prints the selected file and order range. It does not assume scene 3 is in
validation. An explicit `--ann-file` is respected and never silently replaced.
If a scene occurs in more than one split, select one explicitly; split-local
orders must not be merged. Inspect available scenes without radar/RGB loading:

```bash
python tools/temporal_vis/vis_p1_reflection_regions.py --list-scenes
```

These are `order_in_scene` values, NOT radar filenames.
The complete 80..200 step-5 selection produces 25 frames if all orders exist.

The tool also checks the two existing common RGB directory layouts if
`--camera-dir` is omitted. Metadata camera paths take precedence. Otherwise it
reuses `vis_alignment.resolve_rgb`: natural camera ordering against the full
scene annotation; no radar/camera filename-ID equality is assumed. Using a
subset requires `--camera-reference-ann-file` pointing to its complete scene
PKL. A count mismatch is an error, not silently rescaled or nearest-ID matched.
Only use `--camera-offset` for a known ordinal offset. `--rgb-image` is available
for a single manually associated frame. Use `--no-rgb` explicitly for radar only.

RGB is a reference panel, never an algorithm input. Optional projected boxes
require `--projection-npy` containing a calibrated 3x4 radar-XYZ-to-pixel matrix
for the actual displayed camera image. No projection calibration is guessed.
`--rgb-crop-width 1280` keeps the left 1280 columns if the chosen camera format
requires that crop; default is the full image.

## Algorithm and initial settings

1. Read all 43,750 original P1 returns. Exclude non-finite values and points
   outside the explicitly configured analysis ROI. No ego-return mask is added.
2. In 8 m slant-range bands, compute high (95th percentile) and support (80th
   percentile) thresholds. Add median/MAD contrast gates (3 and 0.5 robust sigma)
   so a quantile does not force seeds in a flat/noise-only band. This is NOT a
   calibrated CFAR detector; the input has already been spatially truncated.
3. Rasterize into a temporary XYZ grid: 0.5 m XY, 1 m Z. The rendered BEV is a
   max-strength XY projection, not a learned BEV map. Region extraction keeps Z.
4. Select local strong maxima (1.5 m neighborhood); cap at 128 seeds and report
   any truncation. Grow multiple regions only through lower-threshold supported
   neighboring cells. Seeds remain distinct across weak bridges. Growth is
   bounded by +/-4 m X, +/-2 m Y/Z relative to each seed.
5. Compute observed min/max bounds and a robust center from coordinate 10/90
   percentiles. Draw a fixed 6 x 3 x 3 m fusion box around that center. This
   first diagnostic uses the SAME configurable size in every frame; it is not
   a learned object box, truck-size estimator, or temporal track.
6. Mark proposals with fewer than 6 support points / 3 supported cells, or an
   XY aspect ratio above 8, as low confidence. Retain them in diagnostics; never
   delete their source points. Extremely long structures may be split into
   several bounded reflection proposals, not classified as vehicles.
7. Save ALL original eligible points inside each fixed box, including weak
   returns. Overlaps have a deterministic closest-normalized-center assignment;
   complete per-box membership lists are also preserved.

Important knobs (initial inspection settings, not validated accuracy claims):

```bash
python tools/temporal_vis/vis_p1_reflection_regions.py --help
```

Use `--box-size-m`, `--seed-percentile`, `--support-percentile`, `--seed-mad`,
`--support-mad`, `--peak-separation-m`, `--grow-half-size-m`, `--min-points`,
`--min-cells`, and `--xy-resolution` for a documented diagnostic run. Tune on
training diagnostic frames, not final test results. Use a NEW output directory
for a different parameter set. Default analysis ROI is X [0,85), Y [-60,60),
Z [-5,8) m; exclusions are counted. Display limits never alter the input files.
An existing nonempty output directory is refused unless `--overwrite` is explicit.

## Outputs

- `scene*_order*.png`: four panels: RGB, raw intensity, seed/support cells,
  and colored reflection proposals. Dashed = observed bounds; solid = fixed
  fusion boxes; gray = low confidence. IDs/colors are frame-local, not tracks.
- `scene*_order*.npz`: original XYZ, R/A/E bin indices, stored power,
  original NPZ point IDs, support/seed masks, support-region labels, fixed-box
  labels, all box centers/bounds, and ragged membership arrays in CSR format.
  All arrays load with `allow_pickle=False`.
- `scene*_order*.json`: source/RGB association, thresholds, every proposal,
  exclusions, truncation, and measured geometry/extraction time.
- `frame_summary.csv`, `manifest.json`, `index.html`, and optionally `regions.gif`.

`box_fraction_all_p1` is a fraction of POINTS inside accepted boxes, NOT object
coverage, recall, precision, or occupancy accuracy. NPZ point IDs do not directly
index the GPU power-sorted model tensor; later integration must explicitly map
original indices through the actual P1 sorting. Display caps affect drawing only.

Current scope: no cross-frame association, persistent track IDs, ego warping,
feature interpolation, attention/convolution fusion, or network modifications.
Inspect proposal purity, splitting, merging, missed weak targets, and center
stability against RGB before adding those next stages. Actual target quality and
hardware runtime require the user's real data; synthetic tests prove plumbing.

## Verification

```bash
python -m unittest discover -s tests -p test_p1_reflection_regions.py -v
```
