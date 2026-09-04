#!/usr/bin/env bash

# Usage:
#   ./run_20.sh <tracker_script.py> <plan.npz> [extra tracker arguments]
#
# Example:
#   ./run_20.sh fatrop_rti.py fatrop_plan.npz --horizon 20

set -u

NUM_RUNS=20

SCRIPT="$1"
PLAN="$2"
shift 2

RESULT_DIR="monte_carlo_results"
mkdir -p "$RESULT_DIR"

echo "========================================"
echo "Running $NUM_RUNS trials"
echo "Script: $SCRIPT"
echo "Plan:   $PLAN"
echo "========================================"

successful_runs=0

for i in $(seq 1 "$NUM_RUNS"); do
    printf "\n========== RUN %02d / %02d ==========\n" "$i" "$NUM_RUNS"

    LOG="$RESULT_DIR/run_$(printf "%02d" "$i").log"

    python3 "$SCRIPT" "$PLAN" \
        "$@" \
        2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [ "$status" -eq 0 ]; then
        successful_runs=$((successful_runs + 1))
    else
        echo "WARNING: run $i exited with status $status"
    fi
done


echo
echo "========================================"
echo "AGGREGATING RESULTS"
echo "========================================"

python3 - "$RESULT_DIR" <<'PY'
import re
import sys
from pathlib import Path

import numpy as np


result_dir = Path(sys.argv[1])
files = sorted(result_dir.glob("run_*.log"))

if not files:
    raise RuntimeError("No run log files found.")


def extract(text, pattern, cast=float, default=None):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        return default

    value = match.group(1).strip()

    if cast is bool:
        return value.lower() == "true"

    return cast(value)


results = []

for filename in files:
    text = filename.read_text(errors="replace")

    # Only count logs containing the final tracking summary.
    if "================ TRACKING RESULT =====================" not in text:
        print(f"WARNING: no tracking result found in {filename.name}")
        continue

    reached_goal = extract(
        text,
        r"^Reached final goal:\s*(True|False)",
        bool,
    )

    steps = extract(
        text,
        r"^Controller steps executed:\s*(\d+)",
        int,
    )

    progress_match = re.search(
        r"^Final progress index:\s*(\d+)\s*/\s*(\d+)",
        text,
        flags=re.MULTILINE,
    )

    if progress_match:
        progress_index = int(progress_match.group(1))
        progress_total = int(progress_match.group(2))
    else:
        progress_index = None
        progress_total = None

    final_position_error = extract(
        text,
        r"^Final position error \[m\]:\s*([0-9eE+\-.]+)",
        float,
    )

    max_position_error = extract(
        text,
        r"^Max progress-matched position error \[m\]:\s*([0-9eE+\-.]+)",
        float,
    )

    mean_position_error = extract(
        text,
        r"^Mean progress-matched position error \[m\]:\s*([0-9eE+\-.]+)",
        float,
    )

    initial_solve_ms = extract(
        text,
        r"^Initial full GPU-SLS solve \[ms\]:\s*([0-9eE+\-.]+)",
        float,
    )

    mean_rti_ms = extract(
        text,
        r"^Mean GPU-SLS RTI solve \[ms\]:\s*([0-9eE+\-.]+)",
        float,
    )

    max_rti_ms = extract(
        text,
        r"^Max GPU-SLS RTI solve \[ms\]:\s*([0-9eE+\-.]+)",
        float,
    )

    fallback_controls = extract(
        text,
        r"^Fallback controls used:\s*(\d+)",
        int,
    )

    collision = extract(
        text,
        r"^Obstacle collision:\s*(True|False)",
        bool,
    )

    collision_step = extract(
        text,
        r"^Collision controller step:\s*(-?\d+)",
        int,
    )

    collision_gate = extract(
        text,
        r"^Collision gate:\s*(.+)$",
        str,
    )

    collision_bar = extract(
        text,
        r"^Collision bar:\s*(.+)$",
        str,
    )

    results.append({
        "filename": filename.name,
        "reached_goal": reached_goal,
        "steps": steps,
        "progress_index": progress_index,
        "progress_total": progress_total,
        "final_position_error": final_position_error,
        "max_position_error": max_position_error,
        "mean_position_error": mean_position_error,
        "initial_solve_ms": initial_solve_ms,
        "mean_rti_ms": mean_rti_ms,
        "max_rti_ms": max_rti_ms,
        "fallback_controls": fallback_controls,
        "collision": collision,
        "collision_step": collision_step,
        "collision_gate": collision_gate,
        "collision_bar": collision_bar,
    })


if not results:
    raise RuntimeError("No valid tracking summaries were found.")


# ---------------------------------------------------------
# Per-run output
# ---------------------------------------------------------

print()
print("Per-run results")
print("-" * 110)
print(
    f"{'Run':>5} "
    f"{'Goal':>7} "
    f"{'Collision':>10} "
    f"{'Steps':>7} "
    f"{'Mean RTI [ms]':>15} "
    f"{'Max RTI [ms]':>14} "
    f"{'Mean err [m]':>14} "
    f"{'Max err [m]':>13} "
    f"{'Fallback':>10}"
)
print("-" * 110)

for i, r in enumerate(results, start=1):
    print(
        f"{i:5d} "
        f"{str(r['reached_goal']):>7} "
        f"{str(r['collision']):>10} "
        f"{r['steps'] if r['steps'] is not None else -1:7d} "
        f"{r['mean_rti_ms'] if r['mean_rti_ms'] is not None else np.nan:15.3f} "
        f"{r['max_rti_ms'] if r['max_rti_ms'] is not None else np.nan:14.3f} "
        f"{r['mean_position_error'] if r['mean_position_error'] is not None else np.nan:14.6f} "
        f"{r['max_position_error'] if r['max_position_error'] is not None else np.nan:13.6f} "
        f"{r['fallback_controls'] if r['fallback_controls'] is not None else -1:10d}"
    )

print("-" * 110)


# ---------------------------------------------------------
# Helper for safe statistics
# ---------------------------------------------------------

def values(key):
    return np.asarray(
        [
            r[key]
            for r in results
            if r[key] is not None and np.isfinite(r[key])
        ],
        dtype=float,
    )


num_runs = len(results)

collisions = sum(r["collision"] is True for r in results)
goals_reached = sum(r["reached_goal"] is True for r in results)

collision_rate = collisions / num_runs
goal_rate = goals_reached / num_runs

mean_rti = values("mean_rti_ms")
max_rti = values("max_rti_ms")
initial_solve = values("initial_solve_ms")
steps = values("steps")

mean_err = values("mean_position_error")
max_err = values("max_position_error")
final_err = values("final_position_error")

fallbacks = values("fallback_controls")


# ---------------------------------------------------------
# Final aggregate statistics
# ---------------------------------------------------------

print()
print("================ FINAL RESULTS ================")

print(f"Parsed runs:                       {num_runs}")
print(f"Goals reached:                     {goals_reached}/{num_runs}")
print(f"Goal success rate:                 {100.0 * goal_rate:.2f} %")

print(f"Collisions:                        {collisions}/{num_runs}")
print(f"Collision rate:                    {100.0 * collision_rate:.2f} %")

if len(mean_rti):
    print(
        f"Average mean RTI solve time:       "
        f"{np.mean(mean_rti):.3f} ms"
    )
    print(
        f"Std. dev. mean RTI solve time:     "
        f"{np.std(mean_rti):.3f} ms"
    )

if len(max_rti):
    print(
        f"Average max RTI solve time:        "
        f"{np.mean(max_rti):.3f} ms"
    )
    print(
        f"Worst RTI solve time:              "
        f"{np.max(max_rti):.3f} ms"
    )

if len(initial_solve):
    print(
        f"Average initial full solve:        "
        f"{np.mean(initial_solve):.3f} ms"
    )

if len(steps):
    print(
        f"Average controller steps:          "
        f"{np.mean(steps):.2f}"
    )

if len(mean_err):
    print(
        f"Average mean tracking error:       "
        f"{np.mean(mean_err):.6f} m"
    )

if len(max_err):
    print(
        f"Average max tracking error:        "
        f"{np.mean(max_err):.6f} m"
    )

if len(final_err):
    print(
        f"Average final position error:      "
        f"{np.mean(final_err):.6f} m"
    )

if len(fallbacks):
    print(
        f"Total fallback controls used:      "
        f"{int(np.sum(fallbacks))}"
    )
    print(
        f"Average fallback controls/run:     "
        f"{np.mean(fallbacks):.2f}"
    )

print("===============================================")


# ---------------------------------------------------------
# Collision breakdown
# ---------------------------------------------------------

collision_results = [r for r in results if r["collision"] is True]

if collision_results:
    print()
    print("Collision details")
    print("-" * 60)

    for i, r in enumerate(results, start=1):
        if r["collision"] is not True:
            continue

        print(
            f"Run {i:02d}: "
            f"step={r['collision_step']}, "
            f"gate={r['collision_gate']}, "
            f"bar={r['collision_bar']}"
        )

    print("-" * 60)
PY