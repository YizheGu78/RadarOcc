#!/usr/bin/env bash
set -eo pipefail
RADAROCC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RADAROCC_ROOT"
METHOD="${1:?Usage: bash temporal/run_p0.sh centerformer|rodnet train|val [checkpoint] [extra arguments]}"
ACTION="${2:-train}"
CHECKPOINT="${3:-}"
case "$METHOD" in centerformer|rodnet) ;; *) echo "Unknown method: $METHOD"; exit 1 ;; esac
CONFIG="temporal/$METHOD/config_p0.py"
if [ "$#" -ge 3 ]; then shift 3; else shift "$#"; fi
source "$RADAROCC_ROOT/use_radarocc5060.sh"
case "$ACTION" in
  train)
    if [ -n "$CHECKPOINT" ]; then
      test -f "$CHECKPOINT"
      bash tools/dist_train.sh "$CONFIG" 1 --seed 42 --resume-from "$CHECKPOINT" "$@"
    else
      bash tools/dist_train.sh "$CONFIG" 1 --seed 42 "$@"
    fi
    ;;
  val)
    test -n "$CHECKPOINT" && test -f "$CHECKPOINT"
    bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 1 "$@"
    ;;
  *) echo "Supported actions: train, val (same held-out 20% validation set)"; exit 1 ;;
esac
