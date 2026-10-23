#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/use_radarocc5060.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
ACTION="${1:-train}"
RANKER="${2:-work_dirs/p1_top8_ranker/best.pth}"
OCCUPANCY="${3:-}"
METHOD="${4:-learned}"
case "$METHOD" in
  learned)
    [ -f "$RANKER" ] || { echo "Pretrained ranker not found: $RANKER"; exit 1; }
    ;;
  distance) ;;
  *) echo "Method must be learned or distance"; exit 1 ;;
esac
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_box64_${METHOD}_top8_p1.py"
OPTIONS=("model.temporal_cfg.attention.selection.checkpoint=$RANKER")
case "$ACTION" in
  train)
    if [ -n "$OCCUPANCY" ]; then
      [ -f "$OCCUPANCY" ] || { echo "Occupancy checkpoint not found: $OCCUPANCY"; exit 1; }
      OPTIONS+=("load_from=$OCCUPANCY")
    fi
    bash tools/dist_train.sh "$CONFIG" 1 --cfg-options "${OPTIONS[@]}"
    ;;
  val|test)
    [ -n "$OCCUPANCY" ] && [ -f "$OCCUPANCY" ] || { echo "val/test needs an occupancy checkpoint as argument 3"; exit 1; }
    bash tools/dist_test.sh "$CONFIG" "$OCCUPANCY" 1 --cfg-options "${OPTIONS[@]}" \
      "data.test.ann_file=data/annotations/kradar_dict_${ACTION}_official_temporal_doppler8.pkl" \
      "data.workers_per_gpu=0"
    ;;
  *) echo "Usage: bash $0 train|val|test ranker.pth [occupancy.pth] [learned|distance]"; exit 1 ;;
esac
