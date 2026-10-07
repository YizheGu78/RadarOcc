#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage:"
  echo "  bash run_temporal_t4_current_only_regression.sh CHECKPOINT [val|test]"
  exit 1
fi

CHECKPOINT="$1"
SPLIT="${2:-val}"

if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
  echo "split must be val or test"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

source "$ROOT/use_radarocc5060.sh"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OLD_CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py"
NEW_CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_current_only.py"

OLD_PKL="data/annotations/kradar_dict_${SPLIT}_official_doppler8.pkl"
NEW_PKL="data/annotations/kradar_dict_${SPLIT}_official_temporal_doppler8.pkl"

OUT_DIR="outputs/temporal_t4_current_only_regression/${SPLIT}"
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "1/3 Validate temporal PKL and T=4 sequence bookkeeping"
echo "======================================================================"
python tools/local/check_temporal_t4_regression.py \
  --split "$SPLIT" \
  --frame-nums 4

echo
echo "======================================================================"
echo "2/3 Original RadarOcc evaluation"
echo "======================================================================"
PORT="${PORT_OLD:-29514}" \
bash tools/dist_test.sh \
  "$OLD_CONFIG" \
  "$CHECKPOINT" \
  1 \
  --cfg-options \
  "data.test.ann_file=$OLD_PKL" \
  2>&1 | tee "$OUT_DIR/original.log"

echo
echo "======================================================================"
echo "3/3 T=4 TemporalDataset current-only evaluation"
echo "======================================================================"
PORT="${PORT_NEW:-29515}" \
bash tools/dist_test.sh \
  "$NEW_CONFIG" \
  "$CHECKPOINT" \
  1 \
  --cfg-options \
  "data.test.ann_file=$NEW_PKL" \
  2>&1 | tee "$OUT_DIR/temporal_t4_current_only.log"

echo
echo "======================================================================"
echo "Finished"
echo "======================================================================"
echo "Original log:"
echo "  $OUT_DIR/original.log"
echo "Temporal T=4 current-only log:"
echo "  $OUT_DIR/temporal_t4_current_only.log"
echo
echo "Compare the final evaluation dictionaries. They should be identical"
echo "apart from harmless last-digit floating-point noise."
echo
echo "Quick view:"
grep -E "SSC_|SC_" "$OUT_DIR/original.log" | tail -n 5 || true
grep -E "SSC_|SC_" "$OUT_DIR/temporal_t4_current_only.log" | tail -n 5 || true
