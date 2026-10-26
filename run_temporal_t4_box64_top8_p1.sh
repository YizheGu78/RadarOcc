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
if [ -n "${TOP8_CACHE_DIR:-}" ]; then
  OPTIONS+=("model.temporal_cfg.attention.selection.cache_dir=$TOP8_CACHE_DIR"
            "model.temporal_cfg.attention.selection.cache_mode=${TOP8_CACHE_MODE:-read}")
fi
case "$ACTION" in
  cache|cache-check)
    [ "$METHOD" = learned ] || { echo "Cache warmup requires learned selection"; exit 1; }
    CACHE_ARGS=(--checkpoint "$RANKER" --cache-dir "${TOP8_CACHE_DIR:-data/top8_selection}")
    read -r -a SPLITS <<< "${TOP8_CACHE_SPLITS:-train val}"
    CACHE_ARGS+=(--splits "${SPLITS[@]}")
    if [ "$ACTION" = cache-check ]; then
      CACHE_ARGS+=(--verify --max-samples "${TOP8_CACHE_SAMPLES:-5}")
    else
      CACHE_ARGS+=(--max-samples "${TOP8_CACHE_SAMPLES:-0}")
    fi
    python tools/temporal/cache_top8_selection.py "$CONFIG" "${CACHE_ARGS[@]}"
    ;;
  train)
    if [ -n "$OCCUPANCY" ]; then
      [ -f "$OCCUPANCY" ] || { echo "Occupancy checkpoint not found: $OCCUPANCY"; exit 1; }
      OPTIONS+=("load_from=$OCCUPANCY")
    fi
    if [ -n "${TOP8_RESUME_FROM:-}" ]; then
      [ -f "$TOP8_RESUME_FROM" ] || { echo "Resume checkpoint not found: $TOP8_RESUME_FROM"; exit 1; }
      OPTIONS+=("resume_from=$TOP8_RESUME_FROM")
    fi
    bash tools/dist_train.sh "$CONFIG" 1 --cfg-options "${OPTIONS[@]}"
    ;;
  val|test)
    [ -n "$OCCUPANCY" ] && [ -f "$OCCUPANCY" ] || { echo "val/test needs an occupancy checkpoint as argument 3"; exit 1; }
    bash tools/dist_test.sh "$CONFIG" "$OCCUPANCY" 1 --cfg-options "${OPTIONS[@]}" \
      "data.test.ann_file=data/annotations/kradar_dict_${ACTION}_official_temporal_doppler8.pkl" \
      "data.workers_per_gpu=0"
    ;;
  *) echo "Usage: bash $0 train|val|test|cache|cache-check ranker.pth [occupancy.pth] [learned|distance]"; exit 1 ;;
esac
