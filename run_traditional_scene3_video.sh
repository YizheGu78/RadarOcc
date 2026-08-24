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

SCENE="${SCENE:-3}"
CAMERA_OFFSET="${CAMERA_OFFSET:-0}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-1000}"
FPS="${FPS:-10}"
EGO_SPEED_MPS="${EGO_SPEED_MPS:-auto}"
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

echo "Traditional RPC OGM + video"
echo "  conda env : $CONDA_ENV"
echo "  annotation: $ANNOTATION"
echo "  radar root: $RADAR_ROOT"
echo "  camera dir: $CAMERA_DIR"
echo "  scene     : $SCENE"
echo "  output    : $OUTPUT_DIR"
echo "  fps       : $FPS"
echo "  ego speed : $EGO_SPEED_MPS"

cd "$REPO_ROOT"

xvfb-run -a -s "-screen 0 1920x1080x24" \
    env QT_QPA_PLATFORM=xcb PYTHONPATH="$REPO_ROOT" \
    python -u -m tradition.cli.run \
    --annotation "$ANNOTATION" \
    --radar-root "$RADAR_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --input-mode rpc \
    --scene "$SCENE" \
    --gt-order xyz \
    --ego-speed-mps "$EGO_SPEED_MPS" \
    --video-scene "$SCENE" \
    --camera-dir "$CAMERA_DIR" \
    --camera-offset "$CAMERA_OFFSET" \
    --max-video-frames "$MAX_VIDEO_FRAMES" \
    --fps "$FPS" \
    --no-rotate \
    "${KEEP_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/run.log"

echo
echo "Finished. Results are under: $OUTPUT_DIR"
