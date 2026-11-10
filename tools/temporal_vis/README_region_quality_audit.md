# P1 Reflection-Region Quality Audit: GO / NO-GO screen

This standalone audit evaluates whether extracted reflection regions and
their historical geometry matches are sufficiently reliable to justify
new temporal-fusion experiments. It does not change the training pipeline.

## Why the previous Recall is insufficient

The old benchmark reports Recall only for current regions with an assigned
permissive XY GT ID and a same-ID historical candidate. GT is assigned using
purity=0 and min_gt_points=1. A single target-associated support point can
therefore suffice even if many search-box points are unrelated. Conditional
Precision ignores history-identity-unknown predictions.

This audit reuses the same proposal code, ego alignment, and region matching;
ground truth is examined only AFTER predictions are selected.

## Execute on the machine containing K-Radar data

From repository root on the radarocc-5060-reproduction branch:

~~~bash
git checkout radarocc-5060-reproduction
git pull origin radarocc-5060-reproduction
source use_radarocc5060.sh

python tools/temporal_vis/audit_region_quality.py \
  --ann-files \
      data/annotations/kradar_dict_val_official_temporal_doppler8.pkl \
      data/annotations/kradar_dict_test_official_temporal_doppler8.pkl \
  --bbox-root data/K-RadarOcc/train \
  --start-order 80 --end-order 200 --stride 5 \
  --output-dir outputs/region_quality_audit/val_test_v1
~~~

Scene IDs are discovered from the two annotations. To reproduce just the six
external validation scenes, append: --scenes 1 2 15 22 23 55 .
For scene 3 use the annotation containing it, add --scenes 3, and specify
a DIFFERENT output directory. A smoke run can append --max-frames-per-scene 3.
The script refuses an already-existing output directory.

## Output files

| Output | What it measures |
|---|---|
| summary.md / comparison.csv | A/B/C comparison and preliminary GO/PAUSE/REVISE screen |
| regions.csv | All accepted regions, support/box purity XYZ and XY, projected XY GT polygon IoU, approximate 3D GT AABB IoU |
| objects.csv | Per-annotated-object coverage in P1 and whether any accepted region got its GT ID |
| matches.csv | All current-region/history queries, chosen matches, unknown IDs, same-ID and strict-3D checks |
| inspect_low_purity_links.csv | Worst 100 same-ID historical links for manual examination (biased sample) |
| report.json | Complete settings, counts, caveats and aggregate metrics |

### Score definitions

Support Purity XYZ = P1 support points inside the assigned object's
3D GT cuboid / all region support points.

Box Purity XYZ = all valid P1 radar returns inside the final proposed box
that are also inside the assigned object's 3D GT cuboid / all valid P1
radar returns inside that proposed box.

Importantly, non-target returns may be true road, barrier, building, or
unannotated objects. They are NOT automatically radar clutter.

Strict XYZ ID = support-point GT assignment with true 3D cuboids,
minimum 3 same-target points and support purity >=60%; ties follow the
existing XYZ benchmark policy. A loose same-ID link is strictly confirmed
only if BOTH its current and historical regions independently receive
the SAME strict XYZ ID. Strict Recall has its OWN denominator and must
not be directly subtracted from loose Recall as if the populations match.

3D point evidence confirmation = among loose same-ID links, BOTH boxes
contain >=3 original P1 returns assigned to the claimed 3D GT object.

XY GT IoU is computed against the true projected yaw/tilt GT hull.
3D IoU column compares the region cuboid to an axis-aligned envelope of
the 3D GT; it is an AABB PROXY, not oriented 3D IoU. Region-pair XY IoU
uses the ego-aligned historical region AABB, and may be low for moving
targets without constituting a wrong target ID.

Object coverage is tabulated both for all annotated objects and, separately,
the subset with at least one genuine 3D-GT-associated P1 return in the
analysis RoI; samples are object-by-frame counts, not independent objects.

### Predeclared screening, not a statistical proof

Screening first requires >=30 loose same-ID links and >=30 GT object-frame
observations with a genuine XYZ radar return. The method is PAUSED for
calibration/example review if fewer than 25% of these radar-visible GT
objects get any accepted loose-ID region. It also emits PAUSE when all
three association-quality measures are severely weak: strict ID
confirmation <25%, >=3 3D target points in both boxes <35%, and median
minimum(current,history) 3D Box Purity <15%.

It emits PILOT when object assignment is >=50% and the three association
measures reach >=60%, >=70%, and >=35%. Otherwise it emits
REVISE/INSPECT or INCONCLUSIVE. These heuristics are
intended to triage engineering work. They are not calibrated scientific
cutoffs and cannot establish downstream improvement by themselves.

Before finally abandoning the method, inspect low-purity examples and
verify radar-to-LiDAR calibration, frame association, target movement,
and the foreground-only nature of GT. A positive verdict means: run
a small controlled RadarOcc fusion ablation versus unchanged single-frame
baseline and check actual occupancy IoU / runtime. Do NOT claim that
the audit itself proves an occupancy gain.

## Reproducibility

The region extraction parameters are fixed to:
- seed percentile 97 / seed MAD 3;
- XY merge with xy gap (0.5,0.5)m;
- nominal box (6,3,3)m and XY maximum (20,6,6)m;
- loose XY GT membership, purity 0, min 1 for exact old scoring;
- A: Y (.5,1,1.5), center (4,6,8)m;
- B: Y (4,5,3), center (4,6,8)m;
- C: Y (4,5,3), center (4,12,16)m;
- pure center-distance cost (1/0), XY matching, history t-1..t-3.

The original GT-free proposal and matching routines are imported rather
than reimplemented here. There is no model or GPU dependency, but the
original radar geometry and dataset Python dependencies must be installed.

CPU-only synthetic tests:

~~~bash
python -m unittest tests.test_region_quality_audit -v
~~~
