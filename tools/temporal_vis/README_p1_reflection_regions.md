# P1 reflection-region inspection

This is a GT-free, single-frame diagnostic. It uses the exact original NPZ
membership of Small P1 (first 175 ranges x 250 returns), stored power channel 2,
and the existing real-bin spherical-to-XYZ conversion. No checkpoint, PyTorch,
MMCV, CUDA, or occupancy forward is required. Optional GT boxes/IDs are read only
after region extraction for diagnostic overlays; they never define proposals.

## Run

From the project root in the existing environment:

```bash
source use_radarocc5060.sh
python tools/temporal_vis/vis_p1_reflection_regions.py \
  --ann-file data/annotations/kradar_dict_test_official_temporal_doppler8.pkl \
  --scene 3 --start-order 80 --end-order 200 --stride 5 \
  --seed-percentile 97 --show-gt \
  --merge-space xy --xy-box-gap-m 0.5 0.5 \
  --bbox-root data/K-RadarOcc/train \
  --camera-dir data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched \
  --output-dir outputs/p1_reflection_regions/scene3_p97_gt_xy --gif
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

## GT overlays and fixed image view

The visualization CLI now defaults to seed percentile **97**, matching the
matching benchmark. Older images produced with 95 use different proposals;
pass `--seed-percentile 95` explicitly only to reproduce those older settings.
The reusable `RegionConfig` default is unchanged; this CLI supplies 97 explicitly.

`--show-gt` reads `<bbox-root>/<scene>/bbox/bbox{occ_frame_idx}.npy` and its
`bbox_token{occ_frame_idx}.npy`. The annotation's `occ_frame_idx` is mandatory;
radar filenames and image ordinals are never used to guess GT frame IDs.
Processed boxes use LiDAR bottom-center XYZ, full XYZ dimensions, and yaw in
radians. All eight corners are transformed by the inverse radar-to-LiDAR
extrinsic into the same radar coordinates as the P1 points. The default project
extrinsic is shared with the benchmark; use `--extrinsic radar_to_lidar.npy`
for an explicit calibrated matrix. Wrong/missing GT files fail with their path.

Green polygons show true GT cuboid footprints; `GT:<id>` labels show object IDs.
Duplicate IDs remain visible with `DUP row=...` and cannot supply a reliable
identity. Orange points show ALL support returns; cyan rings show original P1
returns inside unambiguous GT boxes in **3D**; magenta crosses show ambiguous
overlaps. Support and GT-overlap points are never subsampled for display.
The detail panel adds GT height intervals and per-region evidence, for example
`GT:42 8/14 (57%)`. Its denominator is SUPPORT points, not every weak return
inside the enlarged fusion box. Purity defaults to 0.6 and min GT points to 3,
matching the benchmark. `--gt-purity` and `--min-gt-points` affect diagnostics only.

All outputs are static PNG/GIF with a fixed 2D radar XY view: X forward on the
horizontal axis, Y left on the vertical axis. Image/view rotation and RGB EXIF
auto-orientation are disabled; no interactive 3D rotation is provided. GT boxes
retain their physical yaw and sensor coordinate transforms. This does NOT rotate
the picture or reset GT yaw to zero. When no camera projection is supplied, RGB
remains an unprojected reference. With a valid `--projection-npy`, true GT cuboid
edges are also projected onto RGB.

## Algorithm and initial settings

### XY box-footprint merging

Use `--merge-space xy` to ignore height when grouping nearby reflection boxes.
The previous mode remains available as `--merge-space xyz` and is still the
CLI/config default for reproducibility.

1. Keep the existing fine XYZ basins and their original fixed-size boxes. Consider
   ALL fine basins before acceptance filtering, so a height-rejected fragment is
   not discarded before XY grouping.
2. Compare the XY footprints of the original boxes. Overlap is allowed; otherwise
   X and Y EDGE gaps must each be at most `--xy-box-gap-m` (default 0.5/0.5 m).
   This is a box-edge gap, not center distance or point radius. Set `0 0` for
   overlap/touch only. No Z difference or 3D support-cell-contact test is used.
3. Merge nearest footprint links first; break overlap ties by XY center distance
   and then seed IDs. Recheck the WHOLE union's observed support XY span plus
   padding, bounded by the X/Y values of `--max-box-size-m`. This limits chain
   growth; the existing long-thin-chain confidence flag remains active.
4. Rebuild a support-enclosing 3D box from combined original support points. XY
   dimensions remain bounded. Original Z coordinates are retained and enclosed,
   but the Z maximum size is NOT an acceptance gate in this mode: box height may
   exceed 4 m. Point/cell counts and XY shape filters still apply after merging.
   This box-only mode does not recover unassigned support fragments. All eligible
   original returns inside the final box are included in its membership as before.

No GT is used to choose mergers. Same-XY returns at different heights can group
together, without establishing object identity. GT membership/purity remain 3D;
a more complete-looking XY group does not automatically have a higher same-object
GT fraction. Input XYZ, power, NPZ point IDs, and fine-region provenance are kept.

### Original XYZ support-cell merging

The following describes `--merge-space xyz` (the previous default behavior):

1. Read all 43,750 original P1 returns. Exclude non-finite values and points
   outside the explicitly configured analysis ROI. No ego-return mask is added.
2. In 8 m slant-range bands, compute high (97th percentile) and support (80th
   percentile) thresholds. Add median/MAD contrast gates (3 and 0.5 robust sigma)
   so a quantile does not force seeds in a flat/noise-only band. This is NOT a
   calibrated CFAR detector; the input has already been spatially truncated.
3. Rasterize into a temporary XYZ grid: 0.5 m XY, 1 m Z. The rendered BEV is a
   max-strength XY projection, not a learned BEV map. Region extraction keeps Z.
4. Select local strong maxima (1.5 m neighborhood); cap at 128 seeds and report
   any truncation. Grow multiple regions only through lower-threshold supported
   neighboring cells. Seeds remain distinct across weak bridges. Growth is
   bounded by +/-4 m X, +/-2 m Y/Z relative to each seed.
5. Keep the original per-seed regions and fixed 6 x 3 x 3 m boxes for a BEFORE
   panel. Before point-count filtering, merge fine regions with spatial evidence:
   link supported cell centers separated by at most (1.5, 1, 1) m along XYZ.
   Each link needs at least 2 distinct closest-boundary cells on EACH side.
   This permits several peaks on one body while resisting a one-cell bridge.
   Unseeded connected support fragments can also attach to a seeded group across
   small gaps; seedless groups never create standalone proposals. No empty cells
   or extra radar returns are synthesized. Distances are measured on the temporary
   grid, so the grid resolution affects the linkage decision.
6. Process nearest links first, deterministically. Recheck the bounds of the
   ENTIRE union at each step, including padding. Accepted boxes are limited to
   12 x 4 x 4 m, preventing unlimited chaining. Also inspect full linked chains
   before these bounded unions: if a chain exceeds the horizontal size limit and
   its XY aspect ratio exceeds 8, mark its proposals `long_structure_candidate`.
   This is a conservative shape heuristic, not a barrier/vehicle classifier.
7. Create a support-enclosing box: at least 6 x 3 x 3 m, expanding when necessary
   to include all merged support plus padding (0.5, 0.25, 0.5) m per side. Keep
   the robust 10/90 center where possible, adjusting it to enclose support. A fine
   region already larger than the maximum remains visible as `oversized_support`;
   it is NOT silently clipped into an accepted small box. These are single-frame
   sizes, not stabilized sizes across a temporal track.
8. AFTER merging, mark groups with fewer than 6 support points / 3 supported cells,
   an XY aspect ratio above 8, oversized support, or a long-chain flag as low
   confidence. Retain their source points and export their reason. The BEFORE
   and AFTER panels each report counts; accepted counts are not object counts.
9. Save ALL original eligible points inside each box, including weak
   returns. Overlaps have a deterministic closest-normalized-center assignment;
   complete per-box membership lists are also preserved.

Use `--no-merge` to reproduce the original per-seed extraction with exact fixed
`--box-size-m` dimensions. With merging enabled, `--box-size-m` is the MINIMUM
size. `--max-box-size-m` limits accepted groups, and `--box-padding-m` sets XYZ
padding per side. These limits do not establish true object extent or identity;
nearby objects may still merge and sparse single objects may still split.

Important knobs (initial inspection settings, not validated accuracy claims):

```bash
python tools/temporal_vis/vis_p1_reflection_regions.py --help
```

Use `--box-size-m`, `--seed-percentile`, `--support-percentile`, `--seed-mad`,
`--support-mad`, `--peak-separation-m`, `--grow-half-size-m`, `--min-points`,
`--min-cells`, `--merge-gap-m`, `--merge-min-contact-cells`, `--max-box-size-m`,
`--box-padding-m`, `--merge-space`, `--xy-box-gap-m`, and `--xy-resolution` for a documented diagnostic run. Tune on
training diagnostic frames, not final test results. Use a NEW output directory
for a different parameter set. Default analysis ROI is X [0,85), Y [-60,60),
Z [-5,8) m; exclusions are counted. Display limits never alter the input files.
An existing nonempty output directory is refused unless `--overwrite` is explicit.
The logarithmic power color scale is display-only; use `--power-scale linear`
for the original display. The forward-area detail defaults to X [0,60], Y [-10,10]
m; change it with `--detail-x-limits` and `--detail-y-limits`. It does not filter
extraction or exported membership.

## Outputs

- `scene*_order*.png`: six panels: RGB, log-compressed power, seed/support cells,
  original per-seed fixed boxes, merged groups, and a forward-area detail.
  Dashed = observed support bounds; solid = proposal boxes. Only accepted proposal
  boxes are drawn; rejected proposals remain in the array/JSON diagnostics.
  `S` IDs identify original seeds; `R` IDs identify final groups; `n` is the number
  of source seeds in a group, NOT its point count. Detail labels also show center
  height `z`, since distinct height layers can overlap in BEV. IDs/colors are frame-local.
- `scene*_order*.npz`: original XYZ, R/A/E bin indices, stored power,
  original NPZ point IDs, support/seed masks, fine and merged support-region labels,
  box labels, all box centers/bounds/sizes, and ragged membership arrays in CSR
  format. `source_seed_offsets/indices` map final groups to original seeds;
  `fine_center`, `fine_box_min/max` retain the original proposal geometry.
  All arrays load with `allow_pickle=False`.
  With GT overlays, additional arrays preserve `gt_boxes_lidar`, `gt_ids`,
  `gt_corners_radar`, `radar_to_lidar`, and `point_gt_box_rows` (-1 outside GT,
  -2 ambiguous, otherwise the original GT box row).
- `scene*_order*.json`: source/RGB association, thresholds, every proposal,
  exclusions, truncation, merge/recovery counts, and measured geometry/extraction
  time. `fine_regions` retains all original per-seed proposal metadata.
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

`n_fine_regions -> n_regions` measures grouping, not target detection accuracy.
`n_recovered_support` counts existing support returns that were unassigned by the
original growth but attached during merging; `n_unassigned_support` counts support
returns still outside seeded groups. Neither is object recall. Metadata schema
version 3 adds GT overlay provenance and fixed-view settings. This tool does not
modify learned-Top8 caches.

## Verification

```bash
python -m unittest discover -s tests -p test_p1_reflection_regions.py -v
```

