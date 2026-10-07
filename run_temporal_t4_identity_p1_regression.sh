#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECKPOINT="${1:-work_dirs/radarocc_small_fp32_idfix_timealign_v2/epoch_4.pth}"
SPLIT="${2:-val}"

if [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
  echo "split must be val or test"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

set +u
source "$ROOT/use_radarocc5060.sh"
set -u
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OLD_CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py"
IDENTITY_CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32_temporal_t4_identity_p1.py"

OLD_PKL="data/annotations/kradar_dict_${SPLIT}_official_doppler8.pkl"
TEMPORAL_PKL="data/annotations/kradar_dict_${SPLIT}_official_temporal_doppler8.pkl"

CKPT_PARENT="$(basename "$(dirname "$CHECKPOINT")")"
CKPT_FILE="$(basename "$CHECKPOINT" .pth)"
RUN_TAG="${CKPT_PARENT}_${CKPT_FILE}"

OUT_DIR="outputs/temporal_t4_identity_p1_regression/${RUN_TAG}/${SPLIT}"
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "DG-STF P1 Identity Regression"
echo "Checkpoint: $CHECKPOINT"
echo "Split     : $SPLIT"
echo "======================================================================"

echo
echo "1/3 Validate temporal metadata and T=4 bookkeeping"
python tools/local/check_temporal_t4_regression.py \
  --split "$SPLIT" \
  --frame-nums 4

echo
echo "2/3 Original single-frame RadarOcc"
PORT="${PORT_OLD:-29534}" \
bash tools/dist_test.sh \
  "$OLD_CONFIG" \
  "$CHECKPOINT" \
  1 \
  --cfg-options \
  "data.test.ann_file=$OLD_PKL" \
  "data.workers_per_gpu=0" \
  2>&1 | tee "$OUT_DIR/original.log"

echo
echo "3/3 T=4 real loading + P1 Adapter + Identity Plugin"
PORT="${PORT_IDENTITY:-29535}" \
bash tools/dist_test.sh \
  "$IDENTITY_CONFIG" \
  "$CHECKPOINT" \
  1 \
  --cfg-options \
  "data.test.ann_file=$TEMPORAL_PKL" \
  "data.workers_per_gpu=0" \
  2>&1 | tee "$OUT_DIR/identity_p1.log"

echo
echo "======================================================================"
echo "Finished"
echo "Original : $OUT_DIR/original.log"
echo "Identity : $OUT_DIR/identity_p1.log"
echo "The final SC/SSC dictionaries must match."
echo "======================================================================"
