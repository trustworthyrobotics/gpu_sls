#!/usr/bin/env bash

set -u

SCRIPT="plan_drone_obstacle.py"

E_MAGS=(0 1 2.5 5 7.5 10 15)

RESULT_DIR="e_mag_sweep"
mkdir -p "$RESULT_DIR"

# Resolve everything to absolute paths so each run can use its own working dir.
SCRIPT_ABS="$(realpath "$SCRIPT")"
SCRIPT_DIR="$(dirname "$SCRIPT_ABS")"
RESULT_DIR_ABS="$(realpath "$RESULT_DIR")"

echo "========================================"
echo "E_MAG PLANNER SWEEP"
echo "Script: $SCRIPT_ABS"
echo "Values: ${E_MAGS[*]}"
echo "Results: $RESULT_DIR_ABS"
echo "========================================"

for E_MAG in "${E_MAGS[@]}"; do

    TAG="${E_MAG//./p}"
    RUN_DIR="$RESULT_DIR_ABS/E_mag_${TAG}"

    mkdir -p "$RUN_DIR"

    echo
    echo "========================================"
    echo "Running E_MAG = $E_MAG"
    echo "Output directory:"
    echo "  $RUN_DIR"
    echo "========================================"

    # Make a temporary modified copy NEXT TO the original script.
    #
    # Keeping it in SCRIPT_DIR means imports such as
    # visualize_drone_racing continue to work normally.
    MODIFIED_SCRIPT="$(mktemp "$SCRIPT_DIR/.plan_emag_${TAG}_XXXXXX.py")"

    sed \
        "s/^E_MAG = .*/E_MAG = $E_MAG/" \
        "$SCRIPT_ABS" > "$MODIFIED_SCRIPT"

    # Save the exact modified source used for this experiment.
    cp "$MODIFIED_SCRIPT" "$RUN_DIR/plan_drone_obnstacle_E_mag_${TAG}.py"

    # Run from inside the result directory.
    #
    # Your planner currently saves:
    #   multiphase_trajectory.npz
    #   multiphase_trajectory.png
    #   trajectory_topdown.png
    #
    # Since each run gets a separate working directory, none of these
    # files overwrite one another.
    (
        cd "$RUN_DIR"

        python3 "$MODIFIED_SCRIPT" \
            2>&1 | tee "run.log"

        exit "${PIPESTATUS[0]}"
    )

    status=$?

    rm -f "$MODIFIED_SCRIPT"

    if [ "$status" -eq 0 ]; then
        echo "Finished E_MAG=$E_MAG successfully."
        echo "$status" > "$RUN_DIR/exit_status.txt"
    else
        echo "WARNING: E_MAG=$E_MAG exited with status $status"
        echo "$status" > "$RUN_DIR/exit_status.txt"
    fi

done


echo
echo "========================================"
echo "SWEEP COMPLETE"
echo "========================================"

printf "%-10s %-12s %-20s\n" "E_MAG" "STATUS" "RESULT DIRECTORY"
printf "%-10s %-12s %-20s\n" "-----" "------" "----------------"

for E_MAG in "${E_MAGS[@]}"; do

    TAG="${E_MAG//./p}"
    RUN_DIR="$RESULT_DIR_ABS/E_mag_${TAG}"

    if [ -f "$RUN_DIR/exit_status.txt" ]; then
        STATUS="$(cat "$RUN_DIR/exit_status.txt")"

        if [ "$STATUS" -eq 0 ]; then
            STATUS_TEXT="SUCCESS"
        else
            STATUS_TEXT="FAILED($STATUS)"
        fi
    else
        STATUS_TEXT="UNKNOWN"
    fi

    printf "%-10s %-12s %s\n" \
        "$E_MAG" \
        "$STATUS_TEXT" \
        "$RUN_DIR"
done