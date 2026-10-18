#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/use_radarocc5060.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
SUFFIX=""
case "${SUBSET:-full}" in
  full) ;;
  20pct) SUFFIX="_20pct" ;;
  *) echo 'SUBSET must be full or 20pct'; exit 1 ;;
esac
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_unaligned_memory_p1${SUFFIX}.py"
ACTION="${1:-train}"
CHECKPOINT="${2:-}"
GPUS="${GPUS:-1}"
case "$ACTION" in
  train)
    if [ -n "$CHECKPOINT" ]; then
      if [ ! -f "$CHECKPOINT" ]; then
        echo "Checkpoint not found: $CHECKPOINT"; exit 1
      fi
      bash tools/dist_train.sh "$CONFIG" "$GPUS" --cfg-options "load_from=$CHECKPOINT"
    else
      bash tools/dist_train.sh "$CONFIG" "$GPUS"
    fi
    ;;
  val|test)
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
      echo 'val/test requires an existing checkpoint as the second argument'; exit 1
    fi
    ANN_SUFFIX=""
    if [ "$ACTION" = val ]; then ANN_SUFFIX="$SUFFIX"; fi
    bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --cfg-options \
      "data.test.ann_file=data/annotations/kradar_dict_${ACTION}_official_temporal_doppler8${ANN_SUFFIX}.pkl" \
      'data.workers_per_gpu=0'
    ;;
  *) echo "Usage: [SUBSET=20pct] [GPUS=1] bash $0 train|val|test [checkpoint]"; exit 1 ;;
esac
