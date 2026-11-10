#!/usr/bin/env bash
# Visualization and training are intentionally separate actions.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
set +u
source "$TASK_ROOT/use_radarocc5060.sh"
set -u
export PYTHONPATH="$TASK_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
ACTION="${1:-visualize}"
REVIEW_DIR="${QUERY_REVIEW_DIR:-outputs/query_budget/five_scenes_64_32}"
RUN_DIR="${QUERY_WORK_DIR:-work_dirs/temporal_query_budget_20pct_seed${QUERY_SEED:-42}}"
CONFIG=projects/configs/baselines/RadarOcc_Small_5060_temporal_query_budget.py
case "$ACTION" in
  visualize)
    if [[ "$#" -gt 0 ]]; then shift; fi
    python tools/temporal_vis/visualize_query_budget.py --output-dir "$REVIEW_DIR" "$@"
    echo "Only visualization completed. Review $REVIEW_DIR/index.html before choosing train."
    ;;
  summarize)
    python tools/temporal/summarize_query_budget.py --root "$RUN_DIR"
    ;;
  train)
    [[ "${2:-}" == --visual-reviewed ]] || {
      echo 'Review the visualization first, then explicitly run: bash run_temporal_query_budget.sh train --visual-reviewed'
      exit 1
    }
    EPOCHS="${QUERY_EPOCHS:-15}"
    SEED="${QUERY_SEED:-42}"
    RANKER="${QUERY_RANKER:-work_dirs/p1_top8_ranker/best.pth}"
    [[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || { echo 'Invalid epoch/seed'; exit 1; }
    [[ -f "$RANKER" ]] || { echo "Ranker missing: $RANKER"; exit 1; }
    python - "$REVIEW_DIR" "$RUN_DIR" "$CONFIG" "$RANKER" "$EPOCHS" "$SEED" "${QUERY_INIT:-}" "${TOP8_CACHE_DIR:-}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
import torch
from mmcv import Config
from tools.temporal_vis.visualize_query_budget import POLICY, source_hashes
from temporal.core.learned_topk import load_ranker, DEFAULT_POLICY
review, root, config, ranker, epochs, seed, init, cache = sys.argv[1:]
manifest = json.loads((Path(review)/'manifest.json').read_text())
if not manifest.get('complete') or not manifest.get('plots_generated') or manifest['budgets'] != [64, 32]:
    raise SystemExit('Complete Q64/Q32 visualizations are required before training')
if manifest['policy'] != POLICY or manifest['source_sha256'] != source_hashes():
    raise SystemExit('Code/policy changed since visualization; generate a fresh review')
cfg = Config.fromfile(config)
if {k:v for k,v in cfg.model.temporal_cfg.query_budget.items() if k != 'per_range'} != POLICY:
    raise SystemExit('Visualization and resolved training policy disagree')
if cfg.model.temporal_cfg.get('power_mask', {}).get('enabled', False):
    raise SystemExit('Power masking was not included in this visualization')
load_ranker(ranker, DEFAULT_POLICY)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; no training started')
files = [cfg.data.train.ann_file, cfg.data.val.ann_file, ranker]
if init:
    files.append(init)
    state = torch.load(init, map_location='cpu', weights_only=False)
    if any('temporal_plugin.' in k for k in state.get('state_dict', state)):
        raise SystemExit('QUERY_INIT must be a single-frame checkpoint, so temporal weights start identically')
spec = dict(epochs=int(epochs), seed=int(seed), ranker=str(Path(ranker).resolve()),
    initialization=str(Path(init).resolve()) if init else 'random', freeze_ranker=True,
    train_ann=cfg.data.train.ann_file, val_ann=cfg.data.val.ann_file,
    validation='fixed 20% val split, every epoch, best SSC_mean', budgets=[64, 32], policy=POLICY,
    full_query_cache=str(Path(cache).resolve()) if cache else None,
    sources=source_hashes(), inputs={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
root = Path(root)
path = root/'experiment.json'
if path.exists():
    if json.loads(path.read_text()) != spec:
        raise SystemExit('Experiment changed; use a new QUERY_WORK_DIR')
elif root.exists() and any(root.iterdir()):
    raise SystemExit('Use a new work directory')
root.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(spec, indent=2)+'\n')
PY
    trap 'python tools/temporal/summarize_query_budget.py --root "$RUN_DIR"' EXIT
    for BUDGET in 64 32; do
      DIR="$RUN_DIR/Q${BUDGET}"
      if [[ -f "$DIR/completed.json" ]]; then
        echo "Reuse completed experiment Q${BUDGET}"
        continue
      fi
      mkdir -p "$DIR"
      OPTIONS=("model.temporal_cfg.query_budget.per_range=$BUDGET"
               "model.temporal_cfg.attention.selection.checkpoint=$RANKER" "runner.max_epochs=$EPOCHS")
      if [[ -n "${QUERY_INIT:-}" ]]; then OPTIONS+=("load_from=$QUERY_INIT"); fi
      if [[ -n "${TOP8_CACHE_DIR:-}" ]]; then
        OPTIONS+=("model.temporal_cfg.attention.selection.cache_dir=$TOP8_CACHE_DIR"
                   'model.temporal_cfg.attention.selection.cache_mode=read')
      fi
      RESUME=()
      if [[ -f "$DIR/latest.pth" ]]; then RESUME=(--resume-from "$DIR/latest.pth"); fi
      # Both runs use the same seed, trainable architecture and occupancy init.
      python -u tools/train.py "$CONFIG" --seed "$SEED" --work-dir "$DIR" "${RESUME[@]}" \
        --cfg-options "${OPTIONS[@]}" 2>&1 | tee -a "$DIR/console.log"
      python tools/temporal/summarize_query_budget.py --root "$RUN_DIR"
      python - "$DIR" "$EPOCHS" <<'PY'
import json, sys
from pathlib import Path
path, epochs = Path(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(line) for line in (path/'val_metrics.jsonl').read_text().splitlines()]
if not any(r['epoch'] == epochs and r.get('SSC_mean') is not None for r in rows):
    raise SystemExit('Final exact validation metrics missing; not marking complete')
(path/'completed.json').write_text(json.dumps(dict(epochs=epochs, complete=True))+'\n')
PY
    done
    echo "Q64 and Q32 completed: $RUN_DIR/comparison.md"
    ;;
  *) echo 'Usage: bash run_temporal_query_budget.sh visualize | train --visual-reviewed | summarize'; exit 1 ;;
esac
