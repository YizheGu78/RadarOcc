#!/usr/bin/env bash

set -Eeuo pipefail

# Complete traditional RPC OGM evaluation + RadarOcc-style video for official
# test Scene 3. Defaults can be overridden with environment variables, e.g.:
#   FPS=5 KEEP_FRAMES=1 ./run_traditional_scene3_video.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-radarocc-vis}"

ANNOTATION="${ANNOTATION:-$REPO_ROOT/data/annotations/kradar_dict_test_official_doppler8.pkl}"
RADAR_ROOT="${RADAR_ROOT:-$REPO_ROOT/data/K-Radar_rpc}"
CAMERA_DIR="${CAMERA_DIR:-$REPO_ROOT/data/K-Radar-RGB/K-Radar/K-Radar-RGB/3/images_rb_switched}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/work_dirs/tradition_rpc_test_official_scene3_video}"
VIDEO_BACKGROUND_PREDICTION_ROOT="${VIDEO_BACKGROUND_PREDICTION_ROOT:-$REPO_ROOT/work_dirs/radarocc_small_fp32_idfix_timealign_v2/visualization_epoch4_test}"

SCENE="${SCENE:-3}"
CAMERA_OFFSET="${CAMERA_OFFSET:-0}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-1000}"
MAX_FRAMES="${MAX_FRAMES:-}"
FPS="${FPS:-10}"
EGO_SPEED_MPS="${EGO_SPEED_MPS:-auto}"
STATIC_RESIDUAL_THRESHOLD_MPS="${STATIC_RESIDUAL_THRESHOLD_MPS:-0.30}"
KEEP_FRAMES="${KEEP_FRAMES:-0}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

activate_conda_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        return
    fi

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -f /home/user1/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck source=/dev/null
        source /home/user1/miniconda3/etc/profile.d/conda.sh
    else
        fail "Conda was not found. Activate '$CONDA_ENV' before running this script."
    fi
    conda activate "$CONDA_ENV"
}

activate_conda_env

[[ -f "$ANNOTATION" ]] || fail "Annotation not found: $ANNOTATION"
[[ -d "$RADAR_ROOT" ]] || fail "RPC radar root not found: $RADAR_ROOT"
[[ -d "$CAMERA_DIR" ]] || fail "Camera directory not found: $CAMERA_DIR"
[[ -d "$VIDEO_BACKGROUND_PREDICTION_ROOT" ]] || fail \
    "RadarOcc background prediction root not found: $VIDEO_BACKGROUND_PREDICTION_ROOT"
[[ -f "$REPO_ROOT/tradition/cli/run.py" ]] || fail "Not a RadarOcc repository: $REPO_ROOT"

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

echo "Traditional RPC OGM + video"
echo "  conda env : $CONDA_ENV"
echo "  annotation: $ANNOTATION"
echo "  radar root: $RADAR_ROOT"
echo "  camera dir: $CAMERA_DIR"
echo "  blue base : $VIDEO_BACKGROUND_PREDICTION_ROOT"
echo "  scene     : $SCENE"
echo "  output    : $OUTPUT_DIR"
echo "  fps       : $FPS"
echo "  ego speed : $EGO_SPEED_MPS"
echo "  static thr: $STATIC_RESIDUAL_THRESHOLD_MPS m/s"

cd "$REPO_ROOT"

xvfb-run -a -s "-screen 0 1920x1080x24" \
    env QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO_ROOT" \
    python -u -m tradition.cli.run \
    --annotation "$ANNOTATION" \
    --radar-root "$RADAR_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --input-mode rpc \
    --scene "$SCENE" \
    "${FRAME_ARGS[@]}" \
    --gt-order xyz \
    --ego-speed-mps "$EGO_SPEED_MPS" \
    --static-residual-threshold-mps "$STATIC_RESIDUAL_THRESHOLD_MPS" \
    --video-scene "$SCENE" \
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
