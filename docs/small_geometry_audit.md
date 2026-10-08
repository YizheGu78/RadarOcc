# Small base-coordinate audit

Run from the RadarOcc repository root; no torch, CUDA or checkpoint required:

```bash
python tools/temporal_vis/check_small_geometry.py \
  --bins data/info_arr.mat
```

To restrict statistics to a real training frame's first 175 x 250 candidates:

```bash
python tools/temporal_vis/check_small_geometry.py \
  --bins data/info_arr.mat \
  --radar-npz data/RadarOcc_8doppler/1/radar_tensor_8doppler/EAsparse_00001.npz
```

Writes `work_dirs/small_geometry_audit/summary.json`, `examples.csv`, and
`coordinates.npz`. All coordinate vectors are ordered range/azimuth/elevation.
Errors are signed query minus official position; XYZ distance is in metres.
Statistics exclude invalid padded encoder coordinates, out-of-table queries,
out-of-feature-centre queries and zero range. Counts are reported separately.
NPZ mode does not power-weight statistics or change Small's input selection.

The source guards verify the active Small formulas and stride blocks. The CUDA
centre rule is checked locally when the submodule source exists; otherwise the
pinned source `YizheGu78/VoxFormer@2b25090051643aecf331914bb4bfbe3b5b59e8ff`
is assumed and a warning is printed. An installed binary may differ from source.

Derivation: padded indices R/A/E are `[2*r, 2*a+149, 2*e-9]`. Two k3/p1/s2
convolutions have nominal receptive centres at four times the output index.
CUDA samples at `normalized * feature_size - 0.5`. Invert the encoder's
continuous index map, then use the official physical table to interpret that
sample position. This does not round to the nearest bin and does not include
nearest-grid quantization in the measured bias.

This is a nominal base-reference audit, **not the measured location of learned
feature evidence**, and not prediction-to-GT alignment or an IoU estimate.
Convolutions and self-attention mix features. Cross-attention adds learned
sampling offsets, which require a real checkpoint forward pass to measure.
The zero-degree examples use the source's positive azimuth/elevation convention;
this script does not verify radar preprocessing axis flips or external poses.
The Small model is not modified.
