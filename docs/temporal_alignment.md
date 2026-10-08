# P1 ego alignment validation

The T=4 identity regression is the starting point. Alignment changes debug
coordinates only. The plugin still returns the current descriptor unchanged;
there is no historical feature averaging, Doppler correction or target motion
compensation in this stage.

## Geometry contract

P1 indices are [range, azimuth, elevation]. Supply the actual bin tables used
to generate the sparse tensors. `--bins` accepts K-Radar `info_arr.mat` with
`arrRange`, `arrAzimuth`, `arrElevation`, or NPZ with `range_m`, `azimuth_rad`,
`elevation_rad`. Angles must be radians. For cropped/downsampled/flipped tensors,
the lookup arrays must have the same bin ordering as those tensors. Do not use
the network's coarse spherical feature-grid resolution to decode raw P1 bins.

XYZ uses x-forward/y-left/z-up. Default angle signs are +1; set
`--azimuth-sign -1` and/or `--elevation-sign -1` when the source bin arrays use
the opposite angular axes. Verify with a known scene or existing calibrated
RPC; a successful rigid-transform round trip cannot prove the angle convention.

Temporal `ego_pose` is T_world_from_lidar, indexed by the existing occupancy
frame association. E is T_lidar_from_radar. The default E matches the current
repository: identity rotation, translation [2.54, -0.30, -0.70] meters. If the
scene's calibration differs, pass `--extrinsic path/to/T_lidar_from_radar.npy`.
This NPY contains a full 4x4 matrix, not a frame offset or a pose. Alignment is:

    inverse(P_current @ E) @ (P_history @ E)

The result is in the current radar frame, with extrinsics applied once.
Invalid scene-start slots are excluded from rendering and future fusion must
also ignore them. Missing/invalid poses or out-of-bounds bins cause errors.

## Render four frames without a checkpoint

From the repository root in the radarocc5060 environment:

```bash
python tools/temporal_vis/vis_alignment.py \
  --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  --bins /absolute/path/to/info_arr.mat \
  --scene 3 --current-order 80 --frame-nums 4
```

Use a scene present in the selected split. `current-order` is `order_in_scene`,
not the radar filename suffix. By default 2,000 strongest P1 points per frame
are displayed, using ranking channel 2 exactly as the model. Geometry and NPZ
exports include all 43,750 P1 points per frame. Each before/after view uses the
same selected IDs, axis limits and camera view. Outputs under
`outputs/temporal_alignment`: PNG (BEV and 3D), NPZ (full coordinates and
transforms), JSON (frame IDs, mask and geometry settings). No GT is loaded.

Static structures should overlap more after alignment. Moving objects can
remain separated. Inspect multiple current frames, particularly turns. These
plots validate geometry; they do not establish occupancy accuracy gains.

## Optional recording through the model

The alignment config inherits the tested identity P1 config. Put actual lookup
tables at `data/radar_geometry.npz`, or override
`model.temporal_cfg.alignment.bins_path` with your MAT/NPZ path. Override angle
signs and `radar_to_lidar` if needed. Use the existing dist_test entrypoint with
this config and the same epoch_4 checkpoint. It records geometry every 50
forwards while preserving identity features. Use one GPU for this debug path.

```bash
python tools/temporal_vis/vis_alignment.py \
  --debug-file outputs/temporal_alignment_debug/alignment_000000.npz
```

Recording copies coordinates to CPU, so disable it for runtime benchmarks.
After geometry validation, implement local retrieval and masked temporal
averaging as a separate stage.
