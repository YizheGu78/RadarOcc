#!/usr/bin/env bash
set -euo pipefail

SRC="/mnt/elab-share/Datasets"
DST="$HOME/projects/RadarOcc/data"

RADAR_SRC="$SRC/RadarOcc_8doppler"
GT_SRC="$SRC/K-RadarOcc/train"

RADAR_DST="$DST/RadarOcc_8doppler"
GT_DST="$DST/K-RadarOcc/train"

LOG_DIR="$DST/logs"
mkdir -p "$RADAR_DST" "$GT_DST" "$LOG_DIR"

LOG_FILE="$LOG_DIR/download_radarocc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========== RadarOcc data download =========="
echo "SRC=$SRC"
echo "DST=$DST"
echo "LOG=$LOG_FILE"
echo

echo "========== check mount =========="
if ! mount | grep -q "/mnt/elab-share"; then
  echo "[ERROR] /mnt/elab-share is not mounted."
  echo "Please mount SMB first."
  exit 1
fi

echo "========== check source dirs =========="
test -d "$RADAR_SRC" || { echo "[ERROR] missing $RADAR_SRC"; exit 1; }
test -d "$GT_SRC" || { echo "[ERROR] missing $GT_SRC"; exit 1; }

echo "========== disk space =========="
df -h "$DST"
echo

echo "========== copy all sparse radar npz =========="
for seqdir in "$RADAR_SRC"/*; do
  [ -d "$seqdir" ] || continue
  seq="$(basename "$seqdir")"

  if [ ! -d "$seqdir/radar_tensor_8doppler" ]; then
    echo "[SKIP] radar seq $seq has no radar_tensor_8doppler"
    continue
  fi

  echo
  echo "===== radar seq $seq ====="
  mkdir -p "$RADAR_DST/$seq/radar_tensor_8doppler"

  rsync -ah --partial --info=progress2 \
    "$seqdir/radar_tensor_8doppler/" \
    "$RADAR_DST/$seq/radar_tensor_8doppler/"
done

echo
echo "========== copy all semantic occupancy fov gt npy =========="
for seqdir in "$GT_SRC"/*; do
  [ -d "$seqdir" ] || continue
  seq="$(basename "$seqdir")"

  if [ ! -d "$seqdir/semantic_occupancy_gt_fov" ]; then
    echo "[SKIP] GT seq $seq has no semantic_occupancy_gt_fov"
    continue
  fi

  echo
  echo "===== GT seq $seq ====="
  mkdir -p "$GT_DST/$seq/semantic_occupancy_gt_fov"

  rsync -ah --partial --info=progress2 \
    "$seqdir/semantic_occupancy_gt_fov/" \
    "$GT_DST/$seq/semantic_occupancy_gt_fov/"
done

echo
echo "========== final check =========="
echo "Radar npz count:"
find "$RADAR_DST" -type f -name "*.npz" | wc -l

echo "GT fov npy count:"
find "$GT_DST" -type f -path "*/semantic_occupancy_gt_fov/*.npy" | wc -l

echo "Data size:"
du -sh "$RADAR_DST" "$GT_DST"

echo
echo "DONE."
