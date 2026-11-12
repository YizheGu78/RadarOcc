> 坐标修正版请使用仓库根目录的 `temporal/experiments_v2/EXPERIMENTS.md`。本页旧命令保留 legacy 坐标以便复现；支持的离线工具需显式传入 `--coordinate-convention kradar_physical_v2`。旧权重/缓存不能作为 v2 训练产物。

# P2 Region Matching weight benchmark (offline; no Learned KNN)

This experiment evaluates **region-to-region** matches before adding interpolation or Temporal Cross-Attention. It uses the same geometry-only reflection proposal algorithm as `vis_p1_reflection_regions.py`, with no model, Ranker, checkpoint, or GPU dependency.

## Inputs

- Full temporal scene PKL for train/val/test. History is selected by **scene-local row position**, not radar filename number.
- K-Radar P1 radar NPZ and calibrated radar bin geometry.
- GT `bbox{occ_frame_idx}.npy` and `bbox_token{occ_frame_idx}.npy`, used **only for scoring**. At inference the matching algorithm does not access GT.
- Current accepted reflection regions and prior 1, 2, 3 accepted reflection regions.

**Important:** This benchmark **recomputes** the reflection regions with the specified parameters and does not read `outputs/p1_reflection_regions/...` PNG/JSON/NPZ. By default it uses the agreed proposal settings: `seed-percentile=97`, `seed-mad=3`, `box-size-m=6 3 3`, `max-box-size-m=12 4 4`. Override the flags if you want to reproduce another proposal configuration.

## Fast smoke run

From the repository root:

```bash
source use_radarocc5060.sh

python tools/temporal_vis/benchmark_region_matching.py \
  --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
  --max-frames-per-scene 10 \
  --y-gates 0.5 1.0 1.5 \
  --center-gates 4 6 8 \
  --output-dir outputs/region_matching_benchmark/smoke
```

## Validation sweep

```bash
python tools/temporal_vis/benchmark_region_matching.py \
  --ann-file data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl \
  --max-frames-per-scene 100 \
  --y-gates 0.5 1.0 1.5 \
  --center-gates 4 6 8 \
  --output-dir outputs/region_matching_benchmark/val20
```

For scene 3 use the **split containing scene 3**, not the val annotation that contains only scenes 1 and 2. Inspect available scenes with `python tools/temporal_vis/vis_p1_reflection_regions.py --list-scenes`, then pass `--scenes 3` if desired.

If GT bbox files live elsewhere, pass `--bbox-root /path/to/K-RadarOcc/{split}`. Default is `data/K-RadarOcc/train`, consistent with the existing Top-8 GT evaluation tools.

## XY association experiment without height gates

Three flags control independent stages. `--merge-space xy` groups proposal boxes
in XY. `--gt-space xy` evaluates original SUPPORT points against the GT cuboid's
XY footprint in radar coordinates (the convex hull of all eight calibrated,
yaw-preserving corners), ignoring token height. `--match-space xy` uses only XY
center distance and X/Y dimensions for temporal gates and costs. Y gates remain
active; ego alignment still uses the complete calibrated 3D pose transform.
Defaults for both GT membership and temporal matching remain `xyz`.

For the permissive Scene 3 association experiment:

```bash
python tools/temporal_vis/benchmark_region_matching.py \
  --ann-file data/annotations/kradar_dict_test_official_temporal_doppler8.pkl \
  --scenes 3 --start-order 80 --end-order 200 --stride 5 \
  --seed-percentile 97 --seed-mad 3 \
  --merge-space xy --xy-box-gap-m 0.5 0.5 \
  --box-size-m 6 3 3 --max-box-size-m 20 6 6 \
  --gt-space xy --match-space xy --gt-purity 0 --min-gt-points 1 \
  --bbox-root data/K-RadarOcc/train \
  --y-gates 0.5 1.0 1.5 --center-gates 4 6 8 \
  --output-dir outputs/region_matching_benchmark/scene3_p97_xy20_loose_gt
```

Purity zero removes the 60% requirement. A region still needs at least one
unambiguous original SUPPORT point inside a unique-ID GT footprint; merely
touching a GT with an enlarged fusion box is insufficient. Among multiple GT
identities with support, choose the largest support-point count. In XY mode,
equal dominant counts (`ambiguous_dominant`), duplicate GT IDs, and individual
points inside overlapping GT footprints do not establish identity. Diagnostics
retain dominant count/fraction, so a 1/100 association is visible as such.

This is **projected region association**, not a claim of 3D point purity or
occupancy accuracy. Changing only `--gt-space`/GT thresholds cannot change actual
geometric matches. To isolate height effects in GT scoring, use `--gt-space xy`
alone with the old thresholds and keep `--match-space xyz`. Reports record all
spaces and thresholds; do not mix the old and new evaluation denominators.

## Matching algorithm

To evaluate proposals from the XY visualization, pass the SAME
`--merge-space xy --xy-box-gap-m 0.5 0.5` parameters to this benchmark. The default
remains `xyz` for reproducibility. XY mode FIRST filters fine boxes by their
existing radar support point/cell/shape quality, then merges only accepted footprints
without a Z gate or 3D contact requirement, bounding only the union's X/Y size.
Raw point heights and default 3D GT scoring are preserved. This option changes proposal
grouping only: by default the temporal matcher below still uses its existing 3D center and
size costs. It does not silently turn GT scoring into a 2D evaluation.
Rejected fine boxes cannot bridge accepted ones or contribute merged support.
GT label purity does not determine the pre-merge filter; it remains scoring-only.

To reproduce the uncapped XY visualization experiment, also pass
`--no-xy-size-limit` and use a separate output directory. This disables the
whole-union/final XY size caps and their size-dependent long-chain flag.
Pre-merge quality filtering and independent post-merge point/cell/aspect-ratio
checks remain active. The flag requires `--merge-space xy`; GT scoring and the
temporal matching gates below are unchanged. The report records the flag in
`config.no_xy_size_limit`.

1. Extract accepted radar regions independently in every frame. Number of regions is **not fixed**.
2. Transform historical radar boxes into the **current radar coordinate system** using ego poses and radar-to-LiDAR extrinsic. Historical box **AABB** dimensions are conservatively rotated into current axes.
3. For each current region independently in each historical frame, apply strict absolute Y-center gates **0.5, 1.0, 1.5 m** for t−1/t−2/t−3.
4. Apply Euclidean center-distance gates **4, 6, 8 m** in `--match-space` (default XYZ).
5. For remaining candidates rank by:
   ```text
   cost = w_center * (center_distance / center_gate)
        + w_size * (mean(abs(log(history_size/current_size))) / size_scale)
   ```
   `size_scale` defaults to 0.5. Search always picks at most one history region **per current region per history frame**. Many-to-one matching is permitted.
6. Sweep six weight pairs: **1/0, .9/.1, .75/.25, .5/.5, .25/.75, 0/1**. No weights are trained or inferred from GT.

## Offline GT scoring

By default, for each accepted reflection region, assign a GT target token **only if** at least 3 original support points belong to the same unambiguous GT cuboid **and** those points make up at least 60% of all support points. `--gt-space`, `--gt-purity`, and `--min-gt-points` explicitly select the alternative experiment above. Unlabeled, overlapping, and duplicate-ID support counts stay in the purity denominator.

All accepted current regions are now matched and written to `matches.csv`, even
when no current region receives a GT ID. GT scoring is applied separately:
`current_scorable=False` rows are excluded from GT precision/recall. Their
`correct`, `has_true_history`, and GT gate-rejection verdicts are blank (unknown),
not False. A matched history region with no GT ID also has an unknown correctness
verdict. This prevents unknown IDs from being compared as if `None == None` were
a correct same-object match. Proposal and matching parameters are unchanged.

- **Precision**: correct same-ID predicted links / predicted links where the history region has a scorable GT ID. Predictions to unlabeled/ambiguous historical regions are counted separately as `unscorable_prediction`.
- **Recall**: correct same-ID predicted links / current GT-labeled regions with at least one same-ID GT-labeled history region. This denominator **includes** queries rejected by geometry gates or left unmatched.
- **GT Y rejection rate**: fraction of recall-eligible current regions where **all** same-ID history regions fail the hard Y gate.
- **GT center rejection rate**: fraction where Y gate permits a same-ID history region but the selected XY/XYZ center gate eliminates them all.
- `prediction_rate` and `false_match_rate` are also reported.
- `all_queries`, `all_predictions`, and `all_prediction_rate` include ALL accepted
  current-region/history queries, while `current_unlabeled` counts the queries
  excluded from GT scoring. These are geometry matching counts, not accuracy.

Metrics are computed for each individual temporal slot and for all slots combined. Undefined ratios are `null` in JSON, blank in CSV. If the number of GT-labeled regions is very low, treat the results as **inconclusive** and adjust proposal thresholds or GT purity with an explicitly documented experiment.

## Outputs

```text
outputs/region_matching_benchmark/val20/
  metrics.csv     # 6 weights × (t-1, t-2, t-3, all)
  matches.csv     # ALL accepted query-region matches, including unknown GT IDs
  frame_diagnostics.csv      # loaded current/history frames, proposal and GT counts
  region_gt_diagnostics.csv  # GT labeling evidence and reason for each accepted region
  report.json     # config, metrics, runtime, scoring definitions
```

## Empty or unscorable results

Schema version 2 and later always write the full matches header, even if there are no
queries. If accepted regions exist but GT labeling fails, `matches.csv` still
contains geometric matches. Only zero accepted current regions, no selected
frames, or unavailable history positions can leave it with a header only.

Inspect `frame_diagnostics.csv` first: it records exact radar/bbox paths,
`occ_frame_idx`, GT box and duplicate-ID counts, P1 points inside unambiguous GT
boxes, all proposals, accepted proposals, and labeled proposals for every loaded
current/history frame. `proposal_status_counts` also shows why proposals were
rejected (for example `oversized_support`). History frames are included even when not selected for
evaluation. Counts are per frame, not repeated for six weight settings.

`region_gt_diagnostics.csv` separates these labeling outcomes:

| reason | Meaning |
| --- | --- |
| `labeled` | A unique GT ID passed point-count and purity requirements |
| `no_support` | No support points |
| `no_gt_overlap` | No support points lie in an unambiguous GT box |
| `duplicate_id_only` | All overlapping support points belong to duplicate-ID boxes |
| `insufficient_gt_points` | Dominant unique-ID box has fewer than min_gt_points points |
| `low_purity` | Enough dominant points, but their fraction is below gt_purity |
| `ambiguous_dominant` | In XY scoring, multiple unique IDs have equal maximum support counts |

The table includes overlap counts, dominant ID/count/fraction and duplicate-ID
point counts. A zero-GT result remains inconclusive; diagnostic output does not
relax GT thresholds or establish correct matches. Check geometry, frame pairing,
and proposal quality before tuning GT purity. Wider matching gates cannot change
GT labeling. Match rows also include current center XYZ, history accepted/labeled
counts, and candidate counts after Y and 3D-center gates.

The visualization CLI and this benchmark now both default to
`--seed-percentile 97`. Older visualizations used 95; reproduce those explicitly
or regenerate them at 97 with GT overlays. Check recorded configurations before
comparing results.

To view all-slot results:

```bash
python - <<'PY'
import csv
p='outputs/region_matching_benchmark/val20/metrics.csv'
for r in csv.DictReader(open(p)):
    if r['slot']=='all':
        print(r['weight_center'], r['weight_size'], 'precision=', r['precision'],
              'recall=', r['recall'], 'GT-Y-rejection=', r['gt_y_rejection_rate'])
PY
```

The test is CPU-only:

```bash
python -m unittest tests.test_region_matching_benchmark -v
```

**Scope:** This is **matching verification only**, not full RadarOcc IoU, attention feature interpolation, or inference-latency benchmarking. Choose weights on validation and confirm on independent held-out scenes before the final P2 training benchmark. If you optimize proposal thresholds, do not use test GT for tuning.

