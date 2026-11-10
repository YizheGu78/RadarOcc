#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/use_radarocc5060.sh"
OUTPUT_DIR="${1:-outputs/doppler_query_audit/five_scenes_25frames}"
if [ "$#" -gt 0 ]; then shift; fi
python tools/temporal_vis/audit_doppler_queries.py --output-dir "$OUTPUT_DIR" "$@"
