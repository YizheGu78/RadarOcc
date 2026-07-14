#!/usr/bin/env bash
set -o pipefail

PROJECT_DIR="$HOME/projects/RadarOcc"
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_full.py"
CHECKPOINT="work_dirs/radarocc_small_5060_full/epoch_3.pth"
WORK_DIR="work_dirs/radarocc_small_5060_strict_resume"

cd "$PROJECT_DIR"
source "$PROJECT_DIR/use_radarocc5060.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

for file in "$CONFIG" "$CHECKPOINT" \
    "data/annotations/kradar_dict_train_doppler8.pkl"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] 缺少文件：$file"
        exit 1
    fi
done

if pgrep -f '[p]ython.*tools/train.py' >/dev/null; then
    echo "[ERROR] 已存在训练进程："
    pgrep -af '[p]ython.*tools/train.py'
    exit 1
fi

mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/train_console.log"

echo "============================================================"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Work dir:   $WORK_DIR"
echo "============================================================"

python -u tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --gpu-ids 0 \
    --no-validate \
    --resume-from "$CHECKPOINT" \
    2>&1 | tee -a "$LOG_FILE"

status=${PIPESTATUS[0]}
exit "$status"
