#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/use_radarocc5060.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_ego_attention_p1.py"
ACTION="${1:-train}"
CHECKPOINT="${2:-}"
case "$ACTION" in
  train)
    if [ -n "$CHECKPOINT" ]; then
      if [ ! -f "$CHECKPOINT" ]; then
        echo "Checkpoint not found: $CHECKPOINT"; exit 1
      fi
      bash tools/dist_train.sh "$CONFIG" 1 --cfg-options "load_from=$CHECKPOINT"
    else
      bash tools/dist_train.sh "$CONFIG" 1
    fi
    ;;
  val|test)
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
      echo "val/test requires an existing checkpoint as the second argument"; exit 1
    fi
    bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 1 --cfg-options \
      "data.test.ann_file=data/annotations/kradar_dict_${ACTION}_official_temporal_doppler8.pkl" \
      "data.workers_per_gpu=0"
    ;;
  *) echo "Usage: bash $0 train|val|test [checkpoint]"; exit 1 ;;
esac
