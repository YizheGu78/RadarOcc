# T-frame overlays for self-region masking inspection

```bash
python tools/temporal_vis/vis_ego_mask_overlay.py \
  --splits train val test --frame-nums 4 \
  --bins data/K-Radar-official-meta/resources/info_arr.mat \
  --radii-m 0.5 1.0 2.0 \
  --camera-root data/K-Radar-RGB/K-Radar/K-Radar-RGB \
  --output-dir outputs/ego_mask_overlay_audit
```

Each split/scene uses one current frame (middle by default), with up to three
past frames from that same split/scene. `--order 80` selects that scene order
when available and falls back to the middle otherwise. Invalid start padding
and invalid historical poses are excluded from plotting; current pose must be
valid. Only Small P1's first 175 range bins are inspected.

Both columns show the **same ego-aligned coordinates**, not before/after pose
alignment. Left: original returns; right: power-zeroed self-region returns no
longer drawn. Self-region XY masks are computed independently in each frame's
raw radar coordinates **before** applying its pose transform. Hence historical
self-return trails can disappear even when displaced from the current origin.
All five power/statistic channels are zeroed in a temporary copy; Doppler IDs
and coordinates remain intact. Data on disk and training behavior are unchanged.

Frame colors, selected IDs, axis bounds and 3D viewpoints are identical across
the two columns. Select original channel-2 top 2000 returns, plus all returns
inside the largest tested local radius to ensure weak self echoes remain
visible on the left. Do not rerank points after masking. Rows show full BEV,
near-origin BEV (+/-10m by default) and 3D. Dashed frame-colored curves indicate
each historical frame's mask circle after transforming its z=0 plane.

Optional current RGB uses full-scene reference metadata merged from available
train/val/test files, deduplicated by frame token. Camera association uses scene
order, never matching numeric filename IDs. Missing camera folders or unequal
camera/reference counts cause an explicit warning and RGB omission, recorded
in the manifest; no arbitrary offset is inferred.

Outputs: per-split PNGs, `manifest.json` and `index.html`. These sampled
overlays help inspect thresholds; one window per scene cannot prove that every
other frame is free from real-target masking. Candidate radius is not an
automatic self-vehicle classification.
