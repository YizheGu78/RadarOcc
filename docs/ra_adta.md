# P1 RA-ADTA and the retained KNN baseline

`TemporalPlugin` accepts `identity`, `ego_attention` (the existing fixed-radius
KNN attention), or `ra_adta`. The KNN implementation, full-data config and state
dict keys are unchanged. Select a complete config when changing architecture;
changing only `mode` would leave incompatible attention options in the config.

| Experiment | Config under `projects/configs/baselines/` |
| --- | --- |
| KNN, full train/val | `RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py` |
| RA-ADTA, full train/val | `RadarOcc_Small_5060_true_fp32_temporal_t4_ra_adta_p1.py` |
| KNN, fixed 20% train/val | `RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1_20pct.py` |
| RA-ADTA, fixed 20% train/val | `RadarOcc_Small_5060_true_fp32_temporal_t4_ra_adta_p1_20pct.py` |

All four configs keep T=4, P1 Top-K, the current sparse indices, the downstream
RadarOcc model, 15 epochs, from-scratch initialization, per-epoch validation
and `save_best='SSC_mean'`. Test annotations remain the full official test set.
Use the same training seed and the same existing subset PKLs for both runs.

## Implemented method

1. Reuse physical bin lookup and `inverse(P_current @ E) @ (P_history @ E)`.
   The Small model's feature-sampling approximation is not used for matching.
2. Build orthonormal radial/azimuth/elevation directions at each current query.
   Local intervals are read from the physical bin tables (radians for angles),
   including locally varying intervals. Scales are `3*dr`, `3*r*cos(el)*daz`,
   `3*r*del`. Angular metric scales have a configurable 0.25 m floor.
   RA-ADTA requires at least two distinct physical bins on each axis.
3. For each history frame, search a coarse ball with query-specific radius
   `beta*norm(scales) + kappa*max(scales)`. Retain candidates within normalized
   distance `kappa` of the entire allowed offset cube `[-beta,beta]^3`.
   A deterministic set of spread cube anchors selects spatial support before
   remaining slots are filled by normalized proximity. The 24-point pool is a
   bounded approximation; it does not guarantee retaining every useful token.
4. Predict offsets from current features, a weighted history-candidate summary,
   and the same simple relative frame offset used by the KNN path. Four heads
   each predict two samples per history frame. `beta*tanh(offset)` is expressed
   in the current query's local normalized coordinates, so the world location
   is `q + basis @ (scales * offset)`.
5. Compute Gaussian interpolation over candidates in torch. The normalized
   squared distance uses the **same query basis/scales** at all sample locations.
   `temperature=0.5`, `kappa=2.0`; candidates farther than kappa from an individual
   sample have zero weight. All-masked samples are zero and excluded from
   attention. The fixed candidate selection is discrete; interpolation weights
   are differentiable with respect to continuous offsets.
6. Project history K/V once, then attend over valid history frames and samples.
   Shared small position/frame-offset encoding is retained; no FPS or timestamp
   network is introduced. Query chunks default to 256. Non-reentrant checkpoint
   recomputation avoids keeping all expanded candidate tensors for backward.
7. Apply `power + power*expm1(0.5*tanh(residual))` to channels 0,1,2,6,7 only.
   Bin IDs 3:6 are unchanged. The residual projection is zero-initialized for
   exact identity. It gates upstream gradients at the first optimizer step;
   offset/Q/K/V gradients are checked after that first update. Queries without
   usable samples return current features exactly, even after training.

Optional `power_mask` behavior is shared with the KNN path: excluded current
queries and excluded historical candidates cannot be reintroduced by fusion.
For a masked experiment set both `model.temporal_cfg.power_mask` and
`model.pts_middle_encoder.allow_zero_power=True`, as in the existing KNN mask37
config. Apply the same mask setting to both methods for a fair comparison.

The coarse search and candidate organization run on CPU with SciPy. Learned
projections, offsets, interpolation, attention and residuals run on the tensor's
device. CPU search time is included in the benchmark below.

Implementation checks on 2026-10-15 passed 11 RA-ADTA/config tests plus 19 existing
temporal tests. A CPU-only synthetic T=4, N=43,750 run completed forward/backward
and produced nonzero offset gradients after the first residual update. With
PyTorch 2.14.1 and OMP/MKL threads limited to one, the measured post-warmup step
took about 44.2 s forward and 16.1 s backward. This is not an RTX 5060 measurement;
CPU candidate search remains a potential training bottleneck. CUDA memory, real
RadarOcc training/IoU, and full-model DDP were not verified in this environment.

## Checks before training

From the RadarOcc repository root in the existing `radarocc5060` environment:

```bash
python -m unittest discover -s tests -p 'test_ra_adta.py' -v
python -m unittest discover -s tests -p 'test_temporal*.py' -v

python tools/temporal/benchmark_p1_fusion.py \
  --bins-path data/K-Radar-official-meta/resources/info_arr.mat \
  --mode ra_adta --device cuda --points 43750 --steps 2
```

The benchmark uses synthetic P1 inputs and reports forward/backward wall time,
peak GPU allocated memory and parameter gradient norms after warmup. It measures
the plugin only; real Top-K distribution, data loading and the full backbone
require a real training run. It does not establish an IoU improvement. Use
`--mode ego_attention` for the existing KNN cost on the same synthetic input.

## Train the two 20% experiments

If subset PKLs do not already exist, generate them once using the existing tool
(reuse existing files when already generated):

```bash
python tools/temporal/create_temporal_subsets.py --dry-run
python tools/temporal/create_temporal_subsets.py --seed 42
```

Run these GPU jobs sequentially on a single training GPU:

```bash
bash tools/dist_train.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ra_adta_p1_20pct.py \
  1 --seed 42

bash tools/dist_train.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1_20pct.py \
  1 --seed 42
```

These are the actual training entry points (`tools/train.py`), not an isolated
implementation check. Both configs retain `find_unused_parameters=True` for
scene starts without history and the existing FPN non-reentrant checkpoint fix.
Use each config's own work directory for resume; KNN and RA-ADTA have different
fusion parameters, so do not resume one architecture's checkpoint into the other.

## Evaluate on the same 20% validation set

Training already evaluates this split each epoch. To explicitly re-evaluate a
selected RA-ADTA checkpoint, point the test loader at the validation subset:

```bash
bash tools/dist_test.sh \
  projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ra_adta_p1_20pct.py \
  work_dirs/radarocc_small_temporal_t4_ra_adta_p1_20pct/best_SSC_mean_epoch_N.pth \
  1 --cfg-options \
  data.test.ann_file=data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl
```

Replace `best_SSC_mean_epoch_N.pth` with the actual saved filename. For KNN use
its matching config and work directory, with the same validation override.
Compare SSC_mean, SSC_Foreground, training time and GPU memory on this split.
After architecture selection, use the corresponding full-data config to train
on 100% train and select by full validation. Omitting the test-loader override
evaluates the official full test set, which is reserved for final reporting.
