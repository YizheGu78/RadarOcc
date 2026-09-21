#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-radarocc-vis}"

ANNOTATION="${ANNOTATION:-$REPO_ROOT/data/annotations/kradar_dict_train_official_doppler8.pkl}"
RADAR_ROOT="${RADAR_ROOT:-$REPO_ROOT/data/K-Radar_rpc}"
CALIB_ROOT="${CALIB_ROOT:-$REPO_ROOT/data/K-Radar_calib}"
POSE_ROOT="${POSE_ROOT:-$REPO_ROOT/data/K-RadarOcc}"
OUTPUT_MODEL="${OUTPUT_MODEL:-$REPO_ROOT/work_dirs/traditional_object_classifier/object_random_forest_occupancy_first_42d.joblib}"
MAX_FRAMES="${MAX_FRAMES:-}"
POSE_DT_S="${POSE_DT_S:-0.10}"
TEMPORAL_WINDOW="${TEMPORAL_WINDOW:-5}"
MIN_PERSISTENT_SUPPORT="${MIN_PERSISTENT_SUPPORT:-3}"
MIN_MOTION_SUPPORT="${MIN_MOTION_SUPPORT:-2}"
OCCUPANCY_CELL_SIZE_M="${OCCUPANCY_CELL_SIZE_M:-0.40}"
OCCUPANCY_DILATION_CELLS="${OCCUPANCY_DILATION_CELLS:-1}"
VELOCITY_UNWRAPPING="${VELOCITY_UNWRAPPING:-range-kalman}"
UNWRAP_RANGE_STD_M="${UNWRAP_RANGE_STD_M:-0.20}"
UNWRAP_CLUSTER_RADIUS_M="${UNWRAP_CLUSTER_RADIUS_M:-1.50}"
STATIC_RESIDUAL_THRESHOLD_MPS="${STATIC_RESIDUAL_THRESHOLD_MPS:-0.50}"
DYNAMIC_RESIDUAL_THRESHOLD_MPS="${DYNAMIC_RESIDUAL_THRESHOLD_MPS:-0.80}"
DOPPLER_SIGN="${DOPPLER_SIGN:-1}"
STATIC_MATCH_RADIUS_M="${STATIC_MATCH_RADIUS_M:-0.60}"
MOTION_MATCH_RADIUS_M="${MOTION_MATCH_RADIUS_M:-2.00}"
MIN_LOCAL_POWER_RATIO="${MIN_LOCAL_POWER_RATIO:-0.25}"
MIN_LOCAL_NEIGHBORS="${MIN_LOCAL_NEIGHBORS:-1}"
N_ESTIMATORS="${N_ESTIMATORS:-200}"
RANDOM_STATE="${RANDOM_STATE:-13}"

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
[[ -d "$CALIB_ROOT" ]] || fail "Calibration root not found: $CALIB_ROOT"
[[ -d "$POSE_ROOT" ]] || fail "Pose root not found: $POSE_ROOT"
python -c "import sklearn, joblib" >/dev/null 2>&1 || fail \
    "Install scikit-learn and joblib in '$CONDA_ENV'."

FRAME_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
    FRAME_ARGS+=(--max-frames "$MAX_FRAMES")
fi

mkdir -p "$(dirname "$OUTPUT_MODEL")"

echo "Training occupancy-first traditional object classifier"
echo "  annotation : $ANNOTATION"
echo "  radar root : $RADAR_ROOT"
echo "  calib root : $CALIB_ROOT"
echo "  pose root  : $POSE_ROOT"
echo "  output     : $OUTPUT_MODEL"
echo "  velocity   : $VELOCITY_UNWRAPPING"
echo "  persistence: $MIN_PERSISTENT_SUPPORT/$TEMPORAL_WINDOW"
echo "  motion     : $MIN_MOTION_SUPPORT/$TEMPORAL_WINDOW"
echo "  estimators : $N_ESTIMATORS"
echo

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT" python -u -m tradition.cli.train_object_classifier \
    --annotation "$ANNOTATION" \
    --radar-root "$RADAR_ROOT" \
    --calib-root "$CALIB_ROOT" \
    --pose-root "$POSE_ROOT" \
    --pose-dt-s "$POSE_DT_S" \
    --output-model "$OUTPUT_MODEL" \
    --temporal-window "$TEMPORAL_WINDOW" \
    --min-persistent-support "$MIN_PERSISTENT_SUPPORT" \
    --min-motion-support "$MIN_MOTION_SUPPORT" \
    --occupancy-cell-size-m "$OCCUPANCY_CELL_SIZE_M" \
    --occupancy-dilation-cells "$OCCUPANCY_DILATION_CELLS" \
    --velocity-unwrapping "$VELOCITY_UNWRAPPING" \
    --unwrap-range-std-m "$UNWRAP_RANGE_STD_M" \
    --unwrap-cluster-radius-m "$UNWRAP_CLUSTER_RADIUS_M" \
    --static-residual-threshold-mps "$STATIC_RESIDUAL_THRESHOLD_MPS" \
    --dynamic-residual-threshold-mps "$DYNAMIC_RESIDUAL_THRESHOLD_MPS" \
    --stationary-velocity-sign "$DOPPLER_SIGN" \
    --static-match-radius-m "$STATIC_MATCH_RADIUS_M" \
    --dynamic-match-radius-m "$MOTION_MATCH_RADIUS_M" \
    --min-local-power-ratio "$MIN_LOCAL_POWER_RATIO" \
    --min-local-neighbors "$MIN_LOCAL_NEIGHBORS" \
    --n-estimators "$N_ESTIMATORS" \
    --random-state "$RANDOM_STATE" \
    "${FRAME_ARGS[@]}"

PYTHONPATH="$REPO_ROOT" python - "$OUTPUT_MODEL" <<'PY'
import sys

from tradition.semantics.object_classifier import (
    FEATURE_NAMES,
    ObjectAwareSemanticClassifier,
)

model = ObjectAwareSemanticClassifier.from_file(sys.argv[1])
metadata = model.last_diagnostics.get("training_metadata", {})
print(
    "Verified object model:"
    f" features={len(FEATURE_NAMES)},"
    f" oob_score={metadata.get('oob_score')}"
)
PY

echo "Object classifier ready: $OUTPUT_MODEL"
