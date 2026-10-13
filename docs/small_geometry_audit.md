# RadarOcc Small nominal geometry audit

Runs with CPU/NumPy/SciPy only. No checkpoint, model forward, GPU or training.
The script **inspects the ACTIVE detector AST**, detects either the historical
or corrected Small reference formula, and rejects other/mixed formulas.

From the RadarOcc repository root, after updating the existing branch:

```bash
git switch radarocc-5060-reproduction
git pull --ff-only
conda activate radarocc5060

python -m unittest discover -s tests -p 'test_small_geometry_audit.py' -v

python tools/temporal_vis/check_small_geometry.py \
  --bins data/K-Radar-official-meta/resources/info_arr.mat \
  --compare-legacy \
  --out-dir work_dirs/small_geometry_audit_corrected
```

Check a real NPZ's first 175 x 250 radar candidates:

```bash
python tools/temporal_vis/check_small_geometry.py \
  --bins data/K-Radar-official-meta/resources/info_arr.mat \
  --radar-npz data/RadarOcc_8doppler/1/radar_tensor_8doppler/EAsparse_00001.npz \
  --compare-legacy \
  --out-dir work_dirs/small_geometry_audit_corrected_npz
```

Outputs: `summary.json`, `examples.csv`, `coordinates.npz`. Printed
`active_detector_formula` must say `corrected` for commit c26ecfd+.
The top-level `xyz_error_m` and `signed_spherical_error_RAE` are for the
active formula. `comparison_same_valid` compares old and new on the
**same subset** of queries, accounting for differing spatial validity masks.

Geometry:
- Official range step: `0.462890625 m` per raw bin, versus old 0.46 m.
- Small sparse padding in R/A/E: `[2*r,2*a+149,2*e-9]`.
- Two stride-2 sparse convolution blocks imply nominal feature indices
  `[r/2,(2*a+149)/4,(2*e-9)/4]`.
- Current Small spherical query uses
  `r_res = 0.462890625 * scale` and at `scale=2` the pre-normalization
  shifts `[+0.5,+64.25,+7.25]` in R/A/E.
- The 3D VoxFormer sampler maps normalized feature coordinates through
  `feature_index = normalized * feature_size - 0.5`.
  These equations should produce ~zero relative static geometric error
  for valid bins, up to floating-point roundoff.

Statistics exclude invalid padded encoder coordinates, out-of-table queries,
out-of-feature-centre queries and zero range; exclusion counts overlap and
should not be summed. NPZ mode is **not** power-weighted; it does not affect
which candidates are selected during model execution.

**Scope:** This checks theoretical base-reference positions only. It does not
prove that individual sparse-convolution feature evidence is centred exactly
there, that padding out-of-bounds inputs are dropped, that learned attention
offsets are compensated, or that prediction-to-GT IoU improves. Sparse layers
and self-attention mix features. The CUDA centre rule is checked against local
submodule source when available; an installed compiled extension could differ.
