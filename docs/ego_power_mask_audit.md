# Near-origin power masking inspection

This CPU-only script selects one frame for **every scene in each of train, val
and test** from the official temporal annotation files. A scene occurring in
multiple splits receives a separate image in each split. The default frame is
the middle frame within that split/scene. `--order 80` prefers order_in_scene=80,
with an explicitly recorded middle-frame fallback when unavailable.

```bash
python tools/temporal_vis/vis_ego_power_mask.py \
  --bins data/K-Radar-official-meta/resources/info_arr.mat \
  --splits train val test \
  --radii-m 0.5 1.0 2.0 \
  --output-dir outputs/ego_power_mask_audit
```

These are candidate radii, not established vehicle-reflection boundaries.
One selected frame yields three two-column PNGs, one for each radius. To inspect
only a chosen radius, pass e.g. `--radii-m 1.0`. No torch, model or RGB is required.
Use the radarocc5060 environment with NumPy, SciPy and matplotlib.

Masking uses `hypot(x,y) <= radius_m` in the original **local radar coordinates**
of each frame, before pose alignment, and includes all elevations. The official
MAT bin tables convert indices to XYZ. All NPZ returns are inspected, including
ranges beyond Small's P1 crop. Radius is metres; it is not a range-bin index.

The script copies power_val and zeros channels 0,1,2,6,7 inside the mask; it
preserves Doppler IDs 3,4,5 and every coordinate. Source NPZ files are never
modified. Plot channel 2 is the default, matching Small's ranking channel;
`--power-channel` also accepts 0,1,6,7 (7 is variance, not return power).

Each PNG shows original power on the left and zeroed power on the right.
The top row is full extent; bottom row is the near-origin view (default +/-8m).
Color means log10(1+channel value), with exactly the same limits on both sides.
Cells aggregate by max over all elevations and duplicate XY returns. Grey means
zero/no return and cyan circles mark the proposed cutoff. Identical unchanged
outer regions and vanished inner responses can be visually checked.

Outputs: `train/sceneN_r1m.png`, `val/...`, `test/...`, `manifest.csv` and
`manifest.json` and an `index.html` gallery (open in a browser). The manifest records split, token, actual selected order,
source paths, masked counts, zero-range counts, and cutoff definition.

One frame per scene is a **sampled inspection**, not proof that no real object
will be masked in any other frame. A near-origin response is not automatically
a self-vehicle return. Choosing the final fusion mask requires visually checking
the sampled cutoff against the known vehicle-return region. This script does
not enable any mask in TemporalPlugin or change training/testing behavior.
