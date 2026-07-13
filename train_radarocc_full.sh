#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/projects/RadarOcc"
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_full.py"
WORK_DIR="work_dirs/radarocc_small_5060_full"

cd "$PROJECT_DIR"

# 激活已验证通过的 RadarOcc 环境
source "$PROJECT_DIR/use_radarocc5060.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$WORK_DIR"

# 防止误启动第二份训练
if pgrep -f "python.*tools/train.py.*RadarOcc_Small_5060_full.py" >/dev/null; then
    echo "[ERROR] 已检测到 RadarOcc 完整训练进程正在运行。"
    echo "请先执行："
    echo "  pgrep -af \"python.*tools/train.py\""
    echo "  nvidia-smi"
    echo "确认旧任务结束后再重新启动。"
    exit 1
fi

# 检查必要文件
required_files=(
    "$CONFIG"
    "data/annotations/kradar_dict_train_doppler8.pkl"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] 缺少文件：$PROJECT_DIR/$file"
        exit 1
    fi
done

RESUME_ARGS=()

# 优先从 latest.pth 恢复
if [[ -f "$WORK_DIR/latest.pth" ]]; then
    echo "[INFO] 从 $WORK_DIR/latest.pth 恢复训练"
    RESUME_ARGS=(--resume-from "$WORK_DIR/latest.pth")
else
    # 如果 latest.pth 不存在，则尝试寻找编号最大的 epoch checkpoint
    latest_epoch="$(
        find "$WORK_DIR" -maxdepth 1 -type f -name 'epoch_*.pth' -printf '%f\n' \
        | sort -V \
        | tail -n 1
    )"

    if [[ -n "$latest_epoch" ]]; then
        echo "[INFO] 从 $WORK_DIR/$latest_epoch 恢复训练"
        RESUME_ARGS=(--resume-from "$WORK_DIR/$latest_epoch")
    else
        echo "[INFO] 未发现 checkpoint，将从头开始训练"
    fi
fi

LOG_FILE="$WORK_DIR/train_console.log"

echo "============================================================"
echo "Start time: $(date)"
echo "Host:       $(hostname)"
echo "Project:    $PROJECT_DIR"
echo "Config:     $CONFIG"
echo "Work dir:   $WORK_DIR"
echo "Log file:   $LOG_FILE"
echo "============================================================"

python -u tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --gpu-ids 0 \
    --no-validate \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE"

status=${PIPESTATUS[0]}

echo "============================================================"
echo "End time:  $(date)"
echo "Exit code: $status"
echo "============================================================"

exit "$status"
