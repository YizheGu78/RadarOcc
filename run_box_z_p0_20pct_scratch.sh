#!/usr/bin/env bash
# Same random initialization and joint-training budget for Single-frame/Z0-Z3.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
set +u
source "$TASK_ROOT/use_radarocc5060.sh"
set -u
EPOCHS="${2:-15}"
SEED="${3:-42}"
RUN_DIR="${1:-work_dirs/box_z_p0_20pct_scratch_seed${SEED}}"
CACHE_ROOT="${BOX_Z_CACHE_ROOT:-data/box_z_pairs}"
CONFIG=projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct_scratch.py
TRAIN_ANN=data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl
VAL_ANN=data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'Epochs must be positive'; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo 'Seed must be nonnegative'; exit 1; }
if [[ ! -f "$TRAIN_ANN" && ! -f "$VAL_ANN" ]]; then
  python tools/temporal/create_temporal_subsets.py --seed 42
fi
[[ -f "$TRAIN_ANN" && -f "$VAL_ANN" ]] || { echo 'Both fixed 20% subset PKLs are required'; exit 1; }
VAL_COUNT=$(python -c 'import pickle,sys; print(len(pickle.load(open(sys.argv[1],"rb"))["infos"]))' "$VAL_ANN")
python - "$RUN_DIR" "$EPOCHS" "$SEED" "$CACHE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
from mmcv import Config
root,epochs,seed,cache=sys.argv[1:]; root=Path(root)
config='projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct_scratch.py'
cfg=Config.fromfile(config)
assert cfg.load_from is None and cfg.resume_from is None
assert cfg.model.box_z_cfg.freeze_original is False
files=[config,'projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct.py',
 'projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py',
 'projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py',
 'temporal/fusion/box_z_p0.py','projects/occ_plugin/occupancy/detectors/radarocc_box_z.py',
 'projects/occ_plugin/occupancy/detectors/radarocc_self_small.py',
 'projects/occ_plugin/occupancy/voxel_encoder/sparse_lidar_enc.py',
 'projects/occ_plugin/datasets/box_z_dataset.py',
 'projects/occ_plugin/core/evaluation/eval_hooks.py','projects/occ_plugin/utils/formating.py',
 'tools/temporal/check_box_z_scratch.py','run_box_z_p0_20pct_scratch.sh',
 cfg.data.train.ann_file,cfg.data.val.ann_file,cfg.model.box_z_cfg.bins_path]
spec=dict(initialization='random',freeze_original=False,epochs=int(epochs),seed=int(seed),
 methods=['Single-frame','Z1','Z2','Z3','Z0'],cache_root=str(Path(cache).resolve()),
 resolved_config=cfg.pretty_text,
 files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
manifest=root/'experiment.json'
if manifest.exists():
 if json.loads(manifest.read_text())!=spec:
  raise SystemExit('Experiment inputs/code/budget changed; use a new RUN_DIR')
elif root.exists() and any(root.iterdir()):
 raise SystemExit('Nonempty RUN_DIR has no matching scratch manifest; use a new directory')
root.mkdir(parents=True,exist_ok=True)
manifest.write_text(json.dumps(spec,indent=2)+'\n')
PY
python tools/temporal/cache_box_z_pairs.py --ann-file "$TRAIN_ANN" --output-dir "$CACHE_ROOT/train20" --resume
python tools/temporal/cache_box_z_pairs.py --ann-file "$VAL_ANN" --output-dir "$CACHE_ROOT/val20" --resume
if [[ ! -f "$RUN_DIR/preflight.json" ]]; then
  python tools/temporal/check_box_z_scratch.py --config "$CONFIG" --seed "$SEED" \
    --ann-file "$VAL_ANN" --cache "$CACHE_ROOT/val20" --output "$RUN_DIR/preflight.json" \
    2>&1 | tee "$RUN_DIR/preflight.log"
fi
if [[ "${BOX_Z_CHECK_ONLY:-0}" == 1 ]]; then
  echo "Scratch preflight passed. No formal training started: $RUN_DIR/preflight.json"
  exit 0
fi
trap 'python tools/temporal/summarize_box_z_ablation.py --root "$RUN_DIR" --val-frames "$VAL_COUNT"' EXIT
for METHOD in Single-frame Z1 Z2 Z3 Z0; do
  if [[ -f "$RUN_DIR/$METHOD/completed.json" ]]; then
    echo "Reuse completed scratch experiment: $METHOD"
    continue
  fi
  mkdir -p "$RUN_DIR/$METHOD"
  MODE="$METHOD"
  ENABLED=True
  if [[ "$METHOD" == Single-frame ]]; then
    MODE=Z1
    ENABLED=False
  fi
  RESUME_ARGS=()
  if [[ -f "$RUN_DIR/$METHOD/latest.pth" ]]; then
    RESUME_ARGS=(--resume-from "$RUN_DIR/$METHOD/latest.pth")
  fi
  python -u tools/train.py "$CONFIG" --seed "$SEED" --work-dir "$RUN_DIR/$METHOD" \
    "${RESUME_ARGS[@]}" --cfg-options \
    model.box_z_cfg.mode="$MODE" model.box_z_cfg.enabled="$ENABLED" \
    model.box_z_cfg.freeze_original=False \
    model.box_z_cfg.diagnostic_path="$RUN_DIR/$METHOD/eval_diagnostics.jsonl" \
    data.train.box_cache="$CACHE_ROOT/train20" \
    data.val.box_cache="$CACHE_ROOT/val20" \
    data.test.box_cache="$CACHE_ROOT/val20" \
    runner.max_epochs="$EPOCHS" \
    2>&1 | tee -a "$RUN_DIR/$METHOD/console.log"
  python tools/temporal/summarize_box_z_ablation.py --root "$RUN_DIR" --val-frames "$VAL_COUNT"
  python - "$RUN_DIR" "$METHOD" "$EPOCHS" <<'PY'
import csv,json,sys
from pathlib import Path
root,method,epochs=sys.argv[1:]; root=Path(root)
rows=list(csv.DictReader((root/'epoch_iou.csv').open()))
if not any(r['method']==method and int(r['epoch'])==int(epochs) for r in rows):
 raise SystemExit('Final validation missing; experiment not marked complete')
(root/method/'completed.json').write_text(json.dumps(dict(method=method,epochs=int(epochs),initialization='random'))+'\n')
PY
done
echo "All scratch experiments complete: $RUN_DIR/comparison.md"
