#!/usr/bin/env bash
# Serial full-val on/off comparisons. No training or checkpoint updates.
# Examples:
#   bash tools/temporal/run_all_fusion_ablation.sh --dry-run
#   bash tools/temporal/run_all_fusion_ablation.sh
#   bash tools/temporal/run_all_fusion_ablation.sh --ids T00 T01 T03 T10 T11
# Repeat with the same output directory to reuse completed, identical arms.
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${PYTHON:-}" && "${CONDA_DEFAULT_ENV:-}" != "radarocc5060" ]]; then
    source "$REPO_ROOT/use_radarocc5060.sh"
fi
exec "${PYTHON:-python}" tools/temporal/ablate_all_fusion.py "$@"
