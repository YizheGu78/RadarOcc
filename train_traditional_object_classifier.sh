#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-radarocc-vis}"

ANNOTATION="${ANNOTATION:-$REPO_ROOT/data/annotations/kradar_dict_train_official_doppler8.pkl}"
RADAR_ROOT="${RADAR_ROOT:-$REPO_ROOT/data/K-Radar_rpc}"
POSE_ROOT="${POSE_ROOT:-$REPO_ROOT/data/K-RadarOcc}"
OUTPUT_MODEL="${OUTPUT_MODEL:-$REPO_ROOT/work_dirs/traditional_object_classifier/object_random_forest.joblib}"
MAX_FRAMES="${MAX_FRAMES:-}"
POSE_DT_S="${POSE_DT_S:-0.10}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-5}"
N_ESTIMATORS="${N_ESTIMATORS:-200}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

activate_conda_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
        return
    fi
    set +u
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    elif [[ -f /home/user1/miniconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck source=/dev/null
        source /home/user1/miniconda3/etc/profile.d/conda.sh
    else
        fail "Conda was not found. Activate '$CONDA_ENV' first."
    fi
    conda activate "$CONDA_ENV"
    set -u
}

activate_conda_env

[[ -f "$ANNOTATION" ]] || fail "Training annotation not found: $ANNOTATION"
[[ -d "$RADAR_ROOT" ]] || fail "RPC radar root not found: $RADAR_ROOT"
[[ -d "$POSE_ROOT" ]] || fail "Pose root not found: $POSE_ROOT"
python -c "import sklearn, joblib" >/dev/null 2>&1 || fail \
    "Install scikit-learn and joblib in '$CONDA_ENV'."

FRAME_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
    FRAME_ARGS+=(--max-frames "$MAX_FRAMES")
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" python -u -m tradition.cli.train_object_classifier \
    --annotation "$ANNOTATION" \
    --radar-root "$RADAR_ROOT" \
    --pose-root "$POSE_ROOT" \
    --pose-dt-s "$POSE_DT_S" \
    --output-model "$OUTPUT_MODEL" \
    --temporal-window "$TEMPORAL_WINDOW" \
    --n-estimators "$N_ESTIMATORS" \
    "${FRAME_ARGS[@]}"

echo "Object classifier ready: $OUTPUT_MODEL"
