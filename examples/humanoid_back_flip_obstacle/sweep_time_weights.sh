#!/usr/bin/env bash

# Sweep TIME_WEIGHT from 2e4 to 6e4 in increments of 1000.
#
# Usage:
#   ./sweep_time_weights.sh path/to/backflip.py [results_directory]
#
# Optional:
#   PYTHON_BIN=/path/to/python ./sweep_time_weights.sh path/to/backflip.py
#
# Outputs:
#   <results_directory>/summary.csv
#   <results_directory>/weight_<W>/run.log
#   <results_directory>/weight_<W>/humanoid_backflip_obstacle_min_time.npz

set -u
set -o pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 path/to/backflip.py [results_directory]" >&2
    exit 2
fi

SOURCE_SCRIPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
RESULTS_DIR="${2:-time_weight_sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$SOURCE_SCRIPT" ]]; then
    echo "Error: Python script not found: $SOURCE_SCRIPT" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"

SCRIPT_DIR="$(dirname "$SOURCE_SCRIPT")"

# Keep the temporary Python file in the same directory as the original so that
# imports such as `import config_h1` and other relative project assumptions
# continue to work.
TMP_SCRIPT="$(mktemp "${SCRIPT_DIR}/.time_weight_sweep.XXXXXX")"

cleanup() {
    rm -f "$TMP_SCRIPT"
}
trap cleanup EXIT INT TERM

CSV="${RESULTS_DIR}/summary.csv"
echo "time_weight,minimum_time_s,max_dynamics_defect,max_constraint_violation,status" > "$CSV"

echo "Source:  $SOURCE_SCRIPT"
echo "Results: $RESULTS_DIR"
echo "CSV:     $CSV"
echo

for weight in $(seq 20000 2000 60000); do
    RUN_DIR="${RESULTS_DIR}/weight_${weight}"
    LOG="${RUN_DIR}/run.log"
    mkdir -p "$RUN_DIR"

    # Generate a temporary copy with only TIME_WEIGHT changed.
    # The original Python source is never edited.
    awk -v w="$weight" '
        /^[[:space:]]*TIME_WEIGHT[[:space:]]*=/ {
            print "TIME_WEIGHT = " w ".0"
            next
        }
        { print }
    ' "$SOURCE_SCRIPT" > "$TMP_SCRIPT"

    echo "============================================================"
    echo "Running TIME_WEIGHT=${weight}"
    echo "============================================================"

    if "$PYTHON_BIN" "$TMP_SCRIPT" --output-dir "$RUN_DIR" > "$LOG" 2>&1; then
        status="ok"
    else
        status="failed"
    fi

    # Parse the diagnostics already printed by the solver.
    min_time="$(
        sed -n 's/.*Computed minimum backflip time: \([^ ]*\) s.*/\1/p' "$LOG" |
        tail -n 1
    )"

    dyn_defect="$(
        sed -n 's/.*Maximum dynamics defect: \([^ ]*\).*/\1/p' "$LOG" |
        tail -n 1
    )"

    constraint_violation="$(
        sed -n 's/.*Maximum constraint violation: \([^ ]*\).*/\1/p' "$LOG" |
        tail -n 1
    )"

    [[ -n "$min_time" ]] || min_time="NA"
    [[ -n "$dyn_defect" ]] || dyn_defect="NA"
    [[ -n "$constraint_violation" ]] || constraint_violation="NA"

    echo "${weight},${min_time},${dyn_defect},${constraint_violation},${status}" >> "$CSV"

    printf "weight=%-6s  min_time=%-12s  dyn_defect=%-12s  constraint=%-12s  %s\n" \
        "$weight" "$min_time" "$dyn_defect" "$constraint_violation" "$status"

    if [[ "$status" != "ok" ]]; then
        echo "  Run failed; see: $LOG"
    fi
done

echo
echo "Finished sweep."
echo "Summary: $CSV"

if command -v column >/dev/null 2>&1; then
    echo
    column -s, -t "$CSV"
fi
