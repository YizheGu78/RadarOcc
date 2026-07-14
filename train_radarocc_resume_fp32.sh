#!/usr/bin/env bash
set -o pipefail

PROJECT_DIR="$HOME/projects/RadarOcc"

# 真正的 FP32 配置
CONFIG="projects/configs/baselines/RadarOcc_Small_5060_true_fp32.py"

# 新训练结果保存目录
WORK_DIR="work_dirs/radarocc_small_5060_true_fp32_v4"

# 明确指定原训练完成 Epoch 3 后生成的 checkpoint
CHECKPOINT="work_dirs/radarocc_small_5060_full/epoch_3_lr1e-4.pth"

cd "$PROJECT_DIR"

# 激活 radarocc5060 环境
source "$PROJECT_DIR/use_radarocc5060.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 检查必要文件
required_files=(
    "$CONFIG"
    "$CHECKPOINT"
    "data/annotations/kradar_dict_train_doppler8.pkl"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] 缺少文件：$PROJECT_DIR/$file"
        exit 1
    fi
done

# 防止同时运行两份训练
if pgrep -f '[p]ython.*tools/train.py' >/dev/null; then
    echo "[ERROR] 已经存在训练进程："
    pgrep -af '[p]ython.*tools/train.py'
    echo "请先停止旧训练后再运行。"
    exit 1
fi

mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/train_console.log"

echo "============================================================"
echo "Start time: $(date)"
echo "Host:       $(hostname)"
echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Work dir:   $WORK_DIR"
echo "Log file:   $LOG_FILE"
echo "============================================================"

python -u tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --gpu-ids 0 \
    --no-validate \
    --resume-from "$CHECKPOINT" \
    2>&1 | tee -a "$LOG_FILE"

status=${PIPESTATUS[0]}

echo "============================================================"
echo "End time:  $(date)"
echo "Exit code: $status"
echo "============================================================"

exit "$status"
