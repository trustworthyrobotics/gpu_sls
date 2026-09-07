#!/usr/bin/env bash

set -u

TRACKER_SCRIPT="follow_no_obstacle.py"

SWEEP_DIR="/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/gpu_sls/examples/drone_following"

E_MAGS=(0 1 2.5 5 7.5 10 15)
# E_MAGS=(7.5)

TRACKING_ROOT="$SWEEP_DIR/tracking_results"
mkdir -p "$TRACKING_ROOT"

TRACKER_ABS="$(realpath "$TRACKER_SCRIPT")"
TRACKER_DIR="$(dirname "$TRACKER_ABS")"

echo "======================================================"
echo "GPU-SLS TRACKING E_MAG SWEEP"
echo "======================================================"
echo "Tracker: $TRACKER_ABS"
echo "Plans:   $SWEEP_DIR"
echo "Results: $TRACKING_ROOT"
echo "E_MAG values: ${E_MAGS[*]}"
echo "======================================================"

for E_MAG in "${E_MAGS[@]}"; do

    TAG="${E_MAG//./p}"
    PLAN="$SWEEP_DIR/multiphase_trajectory_dist_nom.npz"

    RUN_DIR="$TRACKING_ROOT/E_mag_${TAG}"
    mkdir -p "$RUN_DIR"

    OUTPUT="$RUN_DIR/tracking_E_mag_${TAG}.npz"
    LOG="$RUN_DIR/tracking_E_mag_${TAG}.log"

    echo
    echo "======================================================"
    echo "E_MAG = $E_MAG"
    echo "Plan: $PLAN"
    echo "Tracking DISTURBANCE_MAG = $E_MAG"
    echo "Output: $OUTPUT"
    echo "Log:    $LOG"
    echo "======================================================"

    if [ ! -f "$PLAN" ]; then
        echo "WARNING: missing plan:"
        echo "  $PLAN"
        echo "Skipping."

        echo "MISSING_PLAN" > "$RUN_DIR/exit_status.txt"
        continue
    fi

    # ----------------------------------------------------------
    # Create temporary tracker with matching disturbance magnitude
    # ----------------------------------------------------------
    MODIFIED_TRACKER="$(
        mktemp "$TRACKER_DIR/.follow_no_obstacle_emag_${TAG}_XXXXXX.py"
    )"

    sed \
        "s/^DISTURBANCE_MAG = .*/DISTURBANCE_MAG = $E_MAG/" \
        "$TRACKER_ABS" > "$MODIFIED_TRACKER"

    # Save exact source used for reproducibility.
    cp \
        "$MODIFIED_TRACKER" \
        "$RUN_DIR/follow_no_obstacle_E_mag_${TAG}.py"

    # ----------------------------------------------------------
    # Run tracker
    # ----------------------------------------------------------
    (
        cd "$RUN_DIR"

        python3 "$MODIFIED_TRACKER" \
            "$PLAN" \
            --output "$OUTPUT" \
            2>&1 | tee "$LOG"

        exit "${PIPESTATUS[0]}"
    )

    STATUS=$?

    rm -f "$MODIFIED_TRACKER"

    echo "$STATUS" > "$RUN_DIR/exit_status.txt"

    if [ "$STATUS" -eq 0 ]; then
        echo "SUCCESS: E_MAG=$E_MAG"
    else
        echo "WARNING: E_MAG=$E_MAG exited with status $STATUS"
    fi

done


# ==============================================================
# Aggregate rollout durations directly from tracker logs
# ==============================================================

echo
echo "======================================================"
echo "ROLLOUT DURATION SUMMARY"
echo "======================================================"

printf "%10s %12s %15s %10s %14s\n" \
    "E_MAG" \
    "Status" \
    "Rollout [s]" \
    "Steps" \
    "Reached Goal"

printf '%*s\n' 70 '' | tr ' ' '-'

ROLLOUT_VALUES=()

for E_MAG in "${E_MAGS[@]}"; do

    TAG="${E_MAG//./p}"

    RUN_DIR="$TRACKING_ROOT/E_mag_${TAG}"
    LOG="$RUN_DIR/tracking_E_mag_${TAG}.log"
    STATUS_FILE="$RUN_DIR/exit_status.txt"

    # ----------------------------------------------------------
    # Status
    # ----------------------------------------------------------
    if [ -f "$STATUS_FILE" ]; then
        STATUS_RAW="$(cat "$STATUS_FILE")"

        if [ "$STATUS_RAW" = "0" ]; then
            STATUS="SUCCESS"
        elif [ "$STATUS_RAW" = "MISSING_PLAN" ]; then
            STATUS="NO PLAN"
        else
            STATUS="FAIL($STATUS_RAW)"
        fi
    else
        STATUS="MISSING"
    fi

    # Defaults
    ROLLOUT_DURATION="--"
    STEPS="--"
    REACHED_GOAL="--"

    if [ -f "$LOG" ]; then

        # ------------------------------------------------------
        # Rollout duration
        #
        # Expected:
        # Rollout duration [s]: 6.609667
        # ------------------------------------------------------
        PARSED_ROLLOUT="$(
            grep -E 'Rollout duration \[s\]:' "$LOG" \
                | tail -n 1 \
                | sed -E \
                    's/.*Rollout duration \[s\]:[[:space:]]*([0-9.eE+-]+).*/\1/'
        )"

        if [ -n "$PARSED_ROLLOUT" ]; then
            ROLLOUT_DURATION="$PARSED_ROLLOUT"
            ROLLOUT_VALUES+=("$PARSED_ROLLOUT")
        fi

        # ------------------------------------------------------
        # Controller steps
        #
        # Expected:
        # Controller steps executed: 116
        # ------------------------------------------------------
        PARSED_STEPS="$(
            grep -E 'Controller steps executed:' "$LOG" \
                | tail -n 1 \
                | sed -E \
                    's/.*Controller steps executed:[[:space:]]*([0-9]+).*/\1/'
        )"

        if [ -n "$PARSED_STEPS" ]; then
            STEPS="$PARSED_STEPS"
        fi

        # ------------------------------------------------------
        # Reached final goal
        #
        # Expected:
        # Reached final goal: True
        # ------------------------------------------------------
        PARSED_GOAL="$(
            grep -E 'Reached final goal:' "$LOG" \
                | tail -n 1 \
                | sed -E \
                    's/.*Reached final goal:[[:space:]]*(True|False).*/\1/'
        )"

        if [ -n "$PARSED_GOAL" ]; then
            REACHED_GOAL="$PARSED_GOAL"
        fi
    fi

    printf "%10s %12s %15s %10s %14s\n" \
        "$E_MAG" \
        "$STATUS" \
        "$ROLLOUT_DURATION" \
        "$STEPS" \
        "$REACHED_GOAL"

done


# ==============================================================
# Aggregate rollout statistics
# ==============================================================

if [ "${#ROLLOUT_VALUES[@]}" -gt 0 ]; then

    echo
    echo "Rollout duration statistics"
    echo "----------------------------------------"

    printf "%s\n" "${ROLLOUT_VALUES[@]}" | awk '
    BEGIN {
        n = 0
        sum = 0
    }

    {
        x = $1

        if (n == 0 || x < min)
            min = x

        if (n == 0 || x > max)
            max = x

        sum += x
        n++
    }

    END {
        if (n > 0) {
            printf "Mean   [s]: %.6f\n", sum / n
            printf "Min    [s]: %.6f\n", min
            printf "Max    [s]: %.6f\n", max
        }
    }'

fi


echo
echo "======================================================"
echo "TRACKING SWEEP COMPLETE"
echo "======================================================"