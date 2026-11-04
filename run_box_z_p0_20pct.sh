#!/usr/bin/env bash
# P0 = user's post-selection, pre-Range Attention entry (named P1 in old code).
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
# Existing environment setup references unset variables; enable nounset afterwards.
set +u
source "$TASK_ROOT/use_radarocc5060.sh"
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONHASHSEED=42
INITIAL="${1:-work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth}"
RUN_DIR="${2:-work_dirs/box_z_p0_20pct}"
EPOCHS="${3:-10}"
CACHE_ROOT="${BOX_Z_CACHE_ROOT:-data/box_z_pairs}"
CONFIG=projects/configs/baselines/RadarOcc_Small_box_z_p0_20pct.py
TRAIN_ANN=data/annotations/kradar_dict_train_official_temporal_doppler8_20pct.pkl
VAL_ANN=data/annotations/kradar_dict_val_official_temporal_doppler8_20pct.pkl
[[ -f "$INITIAL" ]] || { echo "Missing single-frame initialization: $INITIAL"; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'Epochs must be positive'; exit 1; }
if [[ ! -f "$TRAIN_ANN" && ! -f "$VAL_ANN" ]]; then
  python tools/temporal/create_temporal_subsets.py --seed 42
fi
[[ -f "$TRAIN_ANN" && -f "$VAL_ANN" ]] || { echo 'Both fixed 20% subset PKLs are required'; exit 1; }
VAL_COUNT=$(python -c 'import pickle,sys; d=pickle.load(open(sys.argv[1],"rb")); print(len(d["infos"]))' "$VAL_ANN")
mkdir -p "$RUN_DIR"
# Refuse to mix different initializations or training budgets in one comparison.
python - "$INITIAL" "$RUN_DIR" "$EPOCHS" "$TRAIN_ANN" "$VAL_ANN" "$CONFIG" <<'PY'
import hashlib,json,sys
from pathlib import Path
initial,root,epochs,train,val,config=sys.argv[1:]
files=[initial,train,val,config,'temporal/fusion/box_z_p0.py',
       'projects/occ_plugin/occupancy/detectors/radarocc_box_z.py',
       'projects/occ_plugin/datasets/box_z_dataset.py']
spec=dict(epochs=int(epochs),seed=42,files={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
p=Path(root)/'experiment.json'
if p.exists() and json.loads(p.read_text()) != spec:
    raise SystemExit('Experiment inputs/code/budget changed; use a new RUN_DIR')
p.write_text(json.dumps(spec,indent=2)+'\n')
PY
python tools/temporal/cache_box_z_pairs.py --ann-file "$TRAIN_ANN" --output-dir "$CACHE_ROOT/train20" --resume
python tools/temporal/cache_box_z_pairs.py --ann-file "$VAL_ANN" --output-dir "$CACHE_ROOT/val20" --resume
if [[ ! -f "$RUN_DIR/preflight.json" ]]; then
  python tools/temporal/check_box_z_integration.py --checkpoint "$INITIAL" \
    --ann-file "$VAL_ANN" --cache "$CACHE_ROOT/val20" --output "$RUN_DIR/preflight.json" \
    2>&1 | tee "$RUN_DIR/preflight.log"
fi
# Update comparison files even if a later training process fails/interruption occurs.
trap 'python tools/temporal/summarize_box_z_ablation.py --root "$RUN_DIR" --val-frames "$VAL_COUNT"' EXIT
if [[ ! -f "$RUN_DIR/single_frame_metrics.json" ]]; then
  python tools/test.py "$CONFIG" "$INITIAL" --eval SSC \
    --metrics-json "$RUN_DIR/single_frame_metrics.json" --seed 42 \
    --cfg-options model.box_z_cfg.enabled=False \
    data.test.box_cache="$CACHE_ROOT/val20" \
    > "$RUN_DIR/single_frame_eval.log" 2>&1
fi
for MODE in Z1 Z2 Z3 Z0; do
  if [[ -f "$RUN_DIR/$MODE/completed.json" ]]; then
    echo "Reuse completed experiment: $MODE"
    continue
  fi
  mkdir -p "$RUN_DIR/$MODE"
  RESUME_ARGS=()
  if [[ -f "$RUN_DIR/$MODE/latest.pth" ]]; then
    RESUME_ARGS=(--resume-from "$RUN_DIR/$MODE/latest.pth")
  fi
  python tools/train.py "$CONFIG" --seed 42 --work-dir "$RUN_DIR/$MODE" \
    "${RESUME_ARGS[@]}" --cfg-options \
    model.box_z_cfg.mode="$MODE" \
    model.box_z_cfg.diagnostic_path="$RUN_DIR/$MODE/eval_diagnostics.jsonl" \
    data.train.box_cache="$CACHE_ROOT/train20" \
    data.val.box_cache="$CACHE_ROOT/val20" \
    data.test.box_cache="$CACHE_ROOT/val20" \
    load_from="$INITIAL" runner.max_epochs="$EPOCHS" \
    2>&1 | tee -a "$RUN_DIR/$MODE/console.log"
  python tools/temporal/summarize_box_z_ablation.py --root "$RUN_DIR" --val-frames "$VAL_COUNT"
  python - "$RUN_DIR" "$MODE" "$EPOCHS" <<'PY'
import csv,json,sys
from pathlib import Path
root,mode,epochs=sys.argv[1:]; root=Path(root)
rows=list(csv.DictReader((root/'epoch_iou.csv').open()))
if not any(r['method']==mode and int(r['epoch'])==int(epochs) for r in rows):
    raise SystemExit('Final validation IoU missing; experiment not marked complete')
(root/mode/'completed.json').write_text(json.dumps(dict(mode=mode,epochs=int(epochs)))+'\n')
PY
done
echo "All experiments complete. Comparison: $RUN_DIR/comparison.md"
