# P2 Region Matching weight benchmark (offline; no Learned KNN)

This experiment evaluates **region-to-region** matches before adding interpolation or Temporal Cross-Attention. It uses the same geometry-only reflection proposal algorithm as `vis_p1_reflection_regions.py`, with no model, Ranker, checkpoint, or GPU dependency.

## Inputs

- Full temporal scene PKL for train/val/test. History is selected by **scene-local row position**, not radar filename number.
- K-Radar P1 radar NPZ and calibrated radar bin geometry.
- GT `bbox{occ_frame_idx}.npy` and `bbox_token{occ_frame_idx}.npy`, used **only for scoring**. At inference the matching algorithm does not access GT.
- Current accepted reflection regions and prior 1, 2, 3 accepted reflection regions.

**Important:** This benchmark **recomputes** the reflection regions with the specified parameters and does not read `outputs/p1_reflection_regions/...` PNG/JSON/NPZ. By default it uses the recent visualization settings: `seed-percentile=98`, `seed-mad=4`, `box-size-m=6 3 3`, `max-box-size-m=12 5 5`. Override the flags if you want to reproduce another proposal configuration.

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

## Matching algorithm

1. Extract accepted radar regions independently in every frame. Number of regions is **not fixed**.
2. Transform historical radar boxes into the **current radar coordinate system** using ego poses and radar-to-LiDAR extrinsic. Historical box **AABB** dimensions are conservatively rotated into current axes.
3. For each current region independently in each historical frame, apply strict absolute Y-center gates **0.5, 1.0, 1.5 m** for t−1/t−2/t−3.
4. Apply 3D Euclidean center-distance gates **4, 6, 8 m**.
5. For remaining candidates rank by:
   ```text
   cost = w_center * (center_distance / center_gate)
        + w_size * (mean(abs(log(history_size/current_size))) / size_scale)
   ```
   `size_scale` defaults to 0.5. Search always picks at most one history region **per current region per history frame**. Many-to-one matching is permitted.
6. Sweep six weight pairs: **1/0, .9/.1, .75/.25, .5/.5, .25/.75, 0/1**. No weights are trained or inferred from GT.

## Offline GT scoring

For each accepted reflection region, assign a GT target token **only if** at least 3 original support points belong to the same unambiguous GT box **and** those points make up at least 60% of all support points. Unlabeled, overlapping, and duplicate-ID support counts stay in the purity denominator. Unscorable current proposals are excluded from matching metrics, but they still exist as candidate history regions.

- **Precision**: correct same-ID predicted links / predicted links where the history region has a scorable GT ID. Predictions to unlabeled/ambiguous historical regions are counted separately as `unscorable_prediction`.
- **Recall**: correct same-ID predicted links / current GT-labeled regions with at least one same-ID GT-labeled history region. This denominator **includes** queries rejected by geometry gates or left unmatched.
- **GT Y rejection rate**: fraction of recall-eligible current regions where **all** same-ID history regions fail the hard Y gate.
- **GT center rejection rate**: fraction where Y gate permits a same-ID history region but the 3D center gate eliminates them all.
- `prediction_rate` and `false_match_rate` are also reported.

Metrics are computed for each individual temporal slot and for all slots combined. Undefined ratios are `null` in JSON, blank in CSV. If the number of GT-labeled regions is very low, treat the results as **inconclusive** and adjust proposal thresholds or GT purity with an explicitly documented experiment.

## Outputs

```text
outputs/region_matching_benchmark/val20/
  metrics.csv     # 6 weights × (t-1, t-2, t-3, all)
  matches.csv     # individual query-region matches with offline GT verdicts
  report.json     # config, metrics, runtime, scoring definitions
```

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
