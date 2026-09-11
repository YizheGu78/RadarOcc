#!/usr/bin/env bash

set -Eeuo pipefail

# Pose-compensated, object-aware traditional RPC OGM evaluation + video.
# Metrics cover every scene in official test by default; visualization uses
# Scene 3 only. Train the Random Forest once before running this script.
#   FPS=5 KEEP_FRAMES=1 ./run_traditional_scene3_video.sh
# Set EVAL_SCENE=3 only for a quick Scene-3-only smoke test.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-radarocc-vis}"
QT_PLATFORM="${QT_PLATFORM:-offscreen}"

ANNOTATION="${ANNOTATION:-$REPO_ROOT/data/annotations/kradar_dict_test_official_doppler8.pkl}"
RADAR_ROOT="${RADAR_ROOT:-$REPO_ROOT/data/K-Radar_rpc}"
POSE_ROOT="${POSE_ROOT:-$REPO_ROOT/data/K-RadarOcc}"
CAMERA_DIR="${CAMERA_DIR:-$REPO_ROOT/data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/work_dirs/tradition_rpc_test_official_all_video_scene3}"
OBJECT_MODEL="${OBJECT_MODEL:-$REPO_ROOT/work_dirs/traditional_object_classifier/object_random_forest.joblib}"
VIDEO_BACKGROUND_PREDICTION_ROOT="${VIDEO_BACKGROUND_PREDICTION_ROOT:-$REPO_ROOT/work_dirs/radarocc_small_fp32_idfix_timealign_v2/visualization_epoch4_test}"

EVAL_SCENE="${EVAL_SCENE:-}"
VIDEO_SCENE="${VIDEO_SCENE:-3}"
CAMERA_OFFSET="${CAMERA_OFFSET:-0}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-1000}"
MAX_FRAMES="${MAX_FRAMES:-}"
FPS="${FPS:-10}"
POSE_DT_S="${POSE_DT_S:-0.10}"
STATIC_RESIDUAL_THRESHOLD_MPS="${STATIC_RESIDUAL_THRESHOLD_MPS:-0.50}"
DYNAMIC_RESIDUAL_THRESHOLD_MPS="${DYNAMIC_RESIDUAL_THRESHOLD_MPS:-0.80}"
DOPPLER_SIGN="${DOPPLER_SIGN:-1}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-5}"
MIN_STATIC_SUPPORT="${MIN_STATIC_SUPPORT:-2}"
MIN_DYNAMIC_SUPPORT="${MIN_DYNAMIC_SUPPORT:-2}"
STATIC_MATCH_RADIUS_M="${STATIC_MATCH_RADIUS_M:-0.60}"
DYNAMIC_MATCH_RADIUS_M="${DYNAMIC_MATCH_RADIUS_M:-2.00}"
MIN_LOCAL_POWER_RATIO="${MIN_LOCAL_POWER_RATIO:-0.25}"
MIN_LOCAL_NEIGHBORS="${MIN_LOCAL_NEIGHBORS:-1}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"
OBJECT_PROBABILITY_THRESHOLD="${OBJECT_PROBABILITY_THRESHOLD:-0.50}"
STATIC_OBJECT_DILATION_CELLS="${STATIC_OBJECT_DILATION_CELLS:-1}"
STATIC_OBJECT_MIN_POINTS="${STATIC_OBJECT_MIN_POINTS:-3}"
STATIC_OBJECT_MIN_CELLS="${STATIC_OBJECT_MIN_CELLS:-2}"
DYNAMIC_OBJECT_EPS_XY_M="${DYNAMIC_OBJECT_EPS_XY_M:-2.00}"
DYNAMIC_OBJECT_EPS_Z_M="${DYNAMIC_OBJECT_EPS_Z_M:-1.00}"
DYNAMIC_OBJECT_MIN_POINTS="${DYNAMIC_OBJECT_MIN_POINTS:-2}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

activate_conda_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        return
    fi

    # Conda's activate/deactivate hooks reference variables that may be unset.
    # Temporarily disable nounset so this script also works when launched from
    # another active conda environment (e.g. radarocc5060).
    set +u
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -f /home/user1/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck source=/dev/null
        source /home/user1/miniconda3/etc/profile.d/conda.sh
    else
        fail "Conda was not found. Activate '$CONDA_ENV' before running this script."
    fi
    conda activate "$CONDA_ENV"
    set -u
}

activate_conda_env

[[ -f "$ANNOTATION" ]] || fail "Annotation not found: $ANNOTATION"
[[ -d "$RADAR_ROOT" ]] || fail "RPC radar root not found: $RADAR_ROOT"
[[ -d "$POSE_ROOT" ]] || fail "RadarOcc pose root not found: $POSE_ROOT"
[[ -d "$CAMERA_DIR" ]] || fail "Camera directory not found: $CAMERA_DIR"
[[ -f "$OBJECT_MODEL" ]] || fail \
    "Object model not found: $OBJECT_MODEL. Run ./train_traditional_object_classifier.sh first."
[[ -d "$VIDEO_BACKGROUND_PREDICTION_ROOT" ]] || fail \
    "RadarOcc background prediction root not found: $VIDEO_BACKGROUND_PREDICTION_ROOT"
[[ -f "$REPO_ROOT/tradition/cli/run.py" ]] || fail "Not a RadarOcc repository: $REPO_ROOT"
python -c "import sklearn, joblib" >/dev/null 2>&1 || fail \
    "Install scikit-learn and joblib in '$CONDA_ENV'. Train and run with the same environment."

command -v xvfb-run >/dev/null 2>&1 || fail \
    "xvfb-run is missing. Install it with: sudo apt-get install -y xvfb"
command -v ffmpeg >/dev/null 2>&1 || fail \
    "ffmpeg is missing. Install it with: sudo apt-get install -y ffmpeg"

mkdir -p "$OUTPUT_DIR"

KEEP_ARGS=()
if [[ "$KEEP_FRAMES" == "1" ]]; then
    KEEP_ARGS+=(--keep-frames)
fi

FRAME_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
    FRAME_ARGS+=(--max-frames "$MAX_FRAMES")
fi

EVAL_ARGS=()
if [[ -n "$EVAL_SCENE" ]]; then
    EVAL_ARGS+=(--scene "$EVAL_SCENE")
fi

echo "Object-aware dual-branch traditional RPC OGM + video"
echo "  conda env : $CONDA_ENV"
echo "  Qt backend: $QT_PLATFORM (headless rendering)"
echo "  annotation: $ANNOTATION"
echo "  radar root: $RADAR_ROOT"
echo "  pose root : $POSE_ROOT"
echo "  camera dir: $CAMERA_DIR"
if [[ -n "$EVAL_SCENE" ]]; then
    echo "  metrics   : scene $EVAL_SCENE only"
else
    echo "  metrics   : all scenes in test_official annotation"
fi
echo "  video     : scene $VIDEO_SCENE only"
echo "  output    : $OUTPUT_DIR"
echo "  RF model  : $OBJECT_MODEL"
echo "  blue base : $VIDEO_BACKGROUND_PREDICTION_ROOT (visualization only)"
echo "  fps       : $FPS"
echo "  pose dt   : $POSE_DT_S s"
echo "  residuals : static <= $STATIC_RESIDUAL_THRESHOLD_MPS m/s; dynamic >= $DYNAMIC_RESIDUAL_THRESHOLD_MPS m/s"
echo "  Doppler   : stationary projection sign $DOPPLER_SIGN"
echo "  temporal  : $MIN_STATIC_SUPPORT/$TEMPORAL_WINDOW static; $MIN_DYNAMIC_SUPPORT/$TEMPORAL_WINDOW dynamic"
echo "  matching  : static $STATIC_MATCH_RADIUS_M m; dynamic $DYNAMIC_MATCH_RADIUS_M m"
echo "  RPC filter: power ratio >= $MIN_LOCAL_POWER_RATIO; neighbors >= $MIN_LOCAL_NEIGHBORS"
echo "  semantics : stationary OGM components + moving DBSCAN -> Random Forest objectness"

cd "$REPO_ROOT"

xvfb-run -a -s "-screen 0 1920x1080x24 -ac +extension GLX +render -noreset" \
    env QT_QPA_PLATFORM="$QT_PLATFORM" MPLBACKEND=Agg PYTHONPATH="$REPO_ROOT" \
    python -u -m tradition.cli.run \
    --annotation "$ANNOTATION" \
    --radar-root "$RADAR_ROOT" \
    --pose-root "$POSE_ROOT" \
    --pose-dt-s "$POSE_DT_S" \
    --output-dir "$OUTPUT_DIR" \
    --input-mode rpc \
    "${EVAL_ARGS[@]}" \
    "${FRAME_ARGS[@]}" \
    --gt-order xyz \
    --static-residual-threshold-mps "$STATIC_RESIDUAL_THRESHOLD_MPS" \
    --dynamic-residual-threshold-mps "$DYNAMIC_RESIDUAL_THRESHOLD_MPS" \
    --stationary-velocity-sign "$DOPPLER_SIGN" \
    --temporal-window "$TEMPORAL_WINDOW" \
    --min-static-support "$MIN_STATIC_SUPPORT" \
    --min-dynamic-support "$MIN_DYNAMIC_SUPPORT" \
    --static-match-radius-m "$STATIC_MATCH_RADIUS_M" \
    --dynamic-match-radius-m "$DYNAMIC_MATCH_RADIUS_M" \
    --min-local-power-ratio "$MIN_LOCAL_POWER_RATIO" \
    --min-local-neighbors "$MIN_LOCAL_NEIGHBORS" \
    --object-model "$OBJECT_MODEL" \
    --object-probability-threshold "$OBJECT_PROBABILITY_THRESHOLD" \
    --static-object-dilation-cells "$STATIC_OBJECT_DILATION_CELLS" \
    --static-object-min-points "$STATIC_OBJECT_MIN_POINTS" \
    --static-object-min-cells "$STATIC_OBJECT_MIN_CELLS" \
    --dynamic-object-eps-xy-m "$DYNAMIC_OBJECT_EPS_XY_M" \
    --dynamic-object-eps-z-m "$DYNAMIC_OBJECT_EPS_Z_M" \
    --dynamic-object-min-points "$DYNAMIC_OBJECT_MIN_POINTS" \
    --video-scene "$VIDEO_SCENE" \
    --camera-dir "$CAMERA_DIR" \
    --camera-offset "$CAMERA_OFFSET" \
    --video-background-prediction-root "$VIDEO_BACKGROUND_PREDICTION_ROOT" \
    --max-video-frames "$MAX_VIDEO_FRAMES" \
    --fps "$FPS" \
    --no-rotate \
    "${KEEP_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo
echo "Finished. Results are under: $OUTPUT_DIR"
