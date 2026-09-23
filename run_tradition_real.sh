#!/usr/bin/env bash
# Run from any directory; uses the currently activated Python environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

stage="${1:-help}"
if (($#)); then shift; fi
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_ANNOTATION="${TRAIN_ANNOTATION:-data/annotations/kradar_dict_train_official_doppler8.pkl}"
TEST_ANNOTATION="${TEST_ANNOTATION:-data/annotations/kradar_dict_test_official_doppler8.pkl}"
CACHE_ROOT="${CACHE_ROOT:-data/frame_fusion}"
TRAIN_CACHE="${TRAIN_CACHE:-$CACHE_ROOT/train_official}"
TEST_CACHE="${TEST_CACHE:-$CACHE_ROOT/test_official}"
N_ESTIMATORS="${N_ESTIMATORS:-200}"
RUN_DIR="${RUN_DIR:-work_dirs/tradition_real/rf_$N_ESTIMATORS}"
TEST_OUTPUT="${TEST_OUTPUT:-${RUN_DIR}_test}"
MODEL="${MODEL:-$RUN_DIR/random_forest.joblib}"

config_args=()
if [[ -n "${CONFIG:-}" ]]; then config_args+=(--config "$CONFIG"); fi
if [[ -n "${TEMPORAL_WINDOW:-}" ]]; then config_args+=(--temporal-window "$TEMPORAL_WINDOW"); fi
if [[ -n "${RADAR_Z:-}" ]]; then config_args+=(--radar-z "$RADAR_Z"); fi
gt_args=(--gt-order "${GT_ORDER:-xyz}")
if [[ -n "${GT_ROOT:-}" ]]; then gt_args+=(--gt-root "$GT_ROOT"); fi

preprocess_split() {
  local annotation="$1" destination="$2"
  shift 2
  "$PYTHON_BIN" -m tradition_real preprocess \
    --annotation "$annotation" --output "$destination" \
    --radar-root "${RADAR_ROOT:-data/K-Radar_rpc}" \
    --pose-root "${POSE_ROOT:-data/K-RadarOcc}" \
    --calib-root "${CALIB_ROOT:-data/K-Radar_calib}" \
    "${config_args[@]}" "$@"
}

train_rf() {
  "$PYTHON_BIN" -m tradition_real train \
    --annotation "$TRAIN_ANNOTATION" --frame-fusion-root "$TRAIN_CACHE" \
    --output "$RUN_DIR" --model "$MODEL" \
    --n-estimators "$N_ESTIMATORS" --max-depth "${MAX_DEPTH:-18}" \
    --min-samples-leaf "${MIN_SAMPLES_LEAF:-2}" \
    --class-weight "${CLASS_WEIGHT:-balanced_subsample}" \
    --n-jobs "${N_JOBS:--1}" --seed "${SEED:-13}" \
    --positive-fraction "${POSITIVE_FRACTION:-0.20}" \
    --negative-fraction "${NEGATIVE_FRACTION:-0.05}" \
    "${config_args[@]}" "${gt_args[@]}" "$@"
}

evaluate_rf() {
  local video_args=() prediction_args=()
  if [[ -n "${CAMERA_DIR:-}" ]]; then
    video_args+=(--camera-dir "$CAMERA_DIR" --video-scene "${VIDEO_SCENE:-3}" --video-fps "${VIDEO_FPS:-10}")
    if [[ -n "${VIDEO_START:-}" ]]; then video_args+=(--video-start "$VIDEO_START"); fi
    if [[ -n "${VIDEO_END:-}" ]]; then video_args+=(--video-end "$VIDEO_END"); fi
  fi
  if [[ "${SAVE_PREDICTIONS:-1}" == 1 ]]; then prediction_args+=(--save-predictions); fi
  "$PYTHON_BIN" -m tradition_real evaluate \
    --annotation "$TEST_ANNOTATION" --frame-fusion-root "$TEST_CACHE" \
    --model "$MODEL" --output "$TEST_OUTPUT" \
    "${config_args[@]}" "${gt_args[@]}" "${prediction_args[@]}" "${video_args[@]}" "$@"
}

case "$stage" in
  preprocess-train) preprocess_split "$TRAIN_ANNOTATION" "$TRAIN_CACHE" "$@" ;;
  preprocess-test) preprocess_split "$TEST_ANNOTATION" "$TEST_CACHE" "$@" ;;
  preprocess)
    preprocess_split "$TRAIN_ANNOTATION" "$TRAIN_CACHE" "$@"
    preprocess_split "$TEST_ANNOTATION" "$TEST_CACHE" "$@"
    ;;
  train) train_rf "$@" ;;
  evaluate) evaluate_rf "$@" ;;
  all)
    if (($#)); then
      echo 'all accepts environment settings only; use individual stages for extra CLI arguments.' >&2
      exit 2
    fi
    preprocess_split "$TRAIN_ANNOTATION" "$TRAIN_CACHE"
    preprocess_split "$TEST_ANNOTATION" "$TEST_CACHE"
    train_rf
    evaluate_rf
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: bash run_tradition_real.sh STAGE [extra Python CLI arguments]
Stages: preprocess-train | preprocess-test | preprocess | train | evaluate | all

First run:  bash run_tradition_real.sh preprocess
            bash run_tradition_real.sh train
            bash run_tradition_real.sh evaluate
RF tuning:  N_ESTIMATORS=500 bash run_tradition_real.sh train
            N_ESTIMATORS=500 bash run_tradition_real.sh evaluate
Scene 3:    CAMERA_DIR=/path/to/scene3/rgb bash run_tradition_real.sh evaluate

Raw paths: TRAIN_ANNOTATION TEST_ANNOTATION RADAR_ROOT POSE_ROOT CALIB_ROOT
Cache/output: CACHE_ROOT TRAIN_CACHE TEST_CACHE RUN_DIR TEST_OUTPUT MODEL
Config: CONFIG TEMPORAL_WINDOW RADAR_Z GT_ROOT GT_ORDER
RF: N_ESTIMATORS MAX_DEPTH MIN_SAMPLES_LEAF CLASS_WEIGHT N_JOBS SEED
    POSITIVE_FRACTION NEGATIVE_FRACTION
Video: CAMERA_DIR VIDEO_SCENE VIDEO_START VIDEO_END VIDEO_FPS
Other: PYTHON_BIN SAVE_PREDICTIONS (default 1)

Preprocess refuses to overwrite an existing cache. For changed preprocessing,
use a new CACHE_ROOT and rebuild both splits. Train/evaluate never rebuild it.
HELP
    ;;
  *) echo "Unknown stage: $stage (use help)" >&2; exit 2 ;;
esac
