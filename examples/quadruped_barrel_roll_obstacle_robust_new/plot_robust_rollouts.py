#!/usr/bin/env python3
"""
Plot the nominal Go2 barrel-roll trajectory, SLS tube envelope, and all
random-disturbance feedback rollouts on the same state-by-state graphs.

For each state coordinate i, this plots

    nominal:        X[k, i]
    tube envelope:  X[k, i] +/- trajectory_tubes[k, i]
    rollouts:       X_random_feedback_rollouts[r, k, i]

The script does not rerun the optimizer or dynamics.

Run:
    python3 plot_nominal_random_rollouts_with_tubes.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Hardcoded NPZ
# ---------------------------------------------------------------------

NPZ_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/"
    "gpu_sls/examples/quadruped_barrel_roll_obstacle_robust_new/"
    "disturbance_rollouts.npz"
)

OUTPUT_PATH = (
    NPZ_PATH.parent
    / "quadruped_barrel_roll_nominal_random_rollouts_with_tubes.png"
)


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------

N_COLS = 5
FIG_DPI = 180

# Plot every rollout by default.
MAX_ROLLOUTS_TO_PLOT = None

# Transparency for individual Monte Carlo rollouts.
ROLLOUT_ALPHA = 0.22
ROLLOUT_LINEWIDTH = 0.8

N_JOINTS = 12
N_CONTACT = 4

PHASE_NAMES = (
    "stance",
    "lateral_launch",
    "rolling_flight",
    "post_roll_flight",
    "terminal_posture",
    "posture_hold",
)


def state_dimension_names(n_states: int) -> list[str]:
    """Human-readable names matching the augmented Go2 state."""

    names = [
        "base_x",
        "base_y",
        "base_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]

    names += [f"q_{i}" for i in range(N_JOINTS)]

    names += [
        "v_x",
        "v_y",
        "v_z",
        "omega_x",
        "omega_y",
        "omega_z",
    ]

    names += [f"dq_{i}" for i in range(N_JOINTS)]

    names += [
        f"foot_{leg}_{axis}"
        for leg in range(N_CONTACT)
        for axis in ("x", "y", "z")
    ]

    names += [
        f"grf_{leg}_{axis}"
        for leg in range(N_CONTACT)
        for axis in ("x", "y", "z")
    ]

    names += [
        f"duration_{phase}"
        for phase in PHASE_NAMES
    ]

    if len(names) != n_states:
        return [f"state_{i}" for i in range(n_states)]

    return names


def align_tubes(
    tubes: np.ndarray,
    num_nodes: int,
    num_states: int,
) -> np.ndarray:
    """Align saved tube radii to x_0,...,x_N."""

    tubes = np.asarray(tubes, dtype=float)

    if tubes.ndim != 2:
        raise ValueError(
            f"trajectory_tubes must be rank 2; got {tubes.shape}"
        )

    if tubes.shape[1] != num_states:
        raise ValueError(
            "Tube state dimension does not match nominal trajectory: "
            f"tubes={tubes.shape}, states={num_states}"
        )

    if tubes.shape[0] == num_nodes:
        return tubes

    if tubes.shape[0] == num_nodes - 1:
        # Fixed initial state has zero tube.
        return np.concatenate(
            [
                np.zeros(
                    (1, num_states),
                    dtype=tubes.dtype,
                ),
                tubes,
            ],
            axis=0,
        )

    raise ValueError(
        "Cannot align trajectory_tubes with nominal trajectory: "
        f"tube nodes={tubes.shape[0]}, nominal nodes={num_nodes}"
    )


def load_data():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(
            f"NPZ does not exist:\n{NPZ_PATH}"
        )

    with np.load(NPZ_PATH, allow_pickle=False) as data:

        if "X" not in data.files:
            raise KeyError(
                f"NPZ does not contain X. Keys: {data.files}"
            )

        if "trajectory_tubes" not in data.files:
            raise KeyError(
                "NPZ does not contain trajectory_tubes."
            )

        X_nominal = np.asarray(
            data["X"],
            dtype=float,
        )

        # Prefer the complete Monte Carlo set.
        if "X_random_feedback_rollouts" in data.files:
            X_rollouts = np.asarray(
                data["X_random_feedback_rollouts"],
                dtype=float,
            )

        elif "X_state_feedback_rollout" in data.files:
            # Fallback for a single feedback rollout.
            X_rollouts = np.asarray(
                data["X_state_feedback_rollout"],
                dtype=float,
            )[None, ...]

        else:
            raise KeyError(
                "NPZ contains neither X_random_feedback_rollouts nor "
                "X_state_feedback_rollout."
            )

        tubes = align_tubes(
            np.asarray(data["trajectory_tubes"]),
            X_nominal.shape[0],
            X_nominal.shape[1],
        )

        if "node_times" in data.files:
            times = np.asarray(
                data["node_times"],
                dtype=float,
            ).reshape(-1)
        else:
            times = np.arange(
                X_nominal.shape[0],
                dtype=float,
            )

        phase_end_steps = (
            np.asarray(
                data["phase_end_steps"],
                dtype=int,
            ).reshape(-1)
            if "phase_end_steps" in data.files
            else None
        )

    if X_nominal.ndim != 2:
        raise ValueError(
            f"X must be rank 2; got {X_nominal.shape}"
        )

    if X_rollouts.ndim != 3:
        raise ValueError(
            "X_random_feedback_rollouts must have shape "
            f"(R,N+1,nx); got {X_rollouts.shape}"
        )

    if X_rollouts.shape[1:] != X_nominal.shape:
        raise ValueError(
            "Rollout and nominal state shapes do not agree: "
            f"rollouts={X_rollouts.shape}, X={X_nominal.shape}"
        )

    if len(times) != X_nominal.shape[0]:
        raise ValueError(
            f"node_times has {len(times)} entries, "
            f"but X has {X_nominal.shape[0]} nodes."
        )

    return (
        X_nominal,
        X_rollouts,
        tubes,
        times,
        phase_end_steps,
    )


def finite_rollout_prefix(
    rollout: np.ndarray,
) -> int:
    """
    Return number of finite nodes from the beginning of a rollout.

    If a rollout becomes non-finite, only its finite prefix is drawn.
    """

    finite = np.all(
        np.isfinite(rollout),
        axis=1,
    )

    if np.all(finite):
        return len(rollout)

    first_bad = int(
        np.argmax(~finite)
    )

    return first_bad


def main():
    (
        X_nominal,
        X_rollouts,
        tubes,
        times,
        phase_end_steps,
    ) = load_data()

    num_rollouts = X_rollouts.shape[0]
    num_nodes = X_nominal.shape[0]
    num_states = X_nominal.shape[1]

    if MAX_ROLLOUTS_TO_PLOT is None:
        rollout_indices = np.arange(num_rollouts)
    else:
        rollout_indices = np.arange(
            min(
                num_rollouts,
                MAX_ROLLOUTS_TO_PLOT,
            )
        )

    names = state_dimension_names(
        num_states
    )

    lower = X_nominal - tubes
    upper = X_nominal + tubes

    n_rows = int(
        np.ceil(num_states / N_COLS)
    )

    fig, axes = plt.subplots(
        n_rows,
        N_COLS,
        figsize=(
            4.8 * N_COLS,
            3.0 * n_rows,
        ),
        sharex=True,
        squeeze=False,
    )

    # -------------------------------------------------------------
    # Plot every state coordinate.
    # -------------------------------------------------------------
    for state_idx, ax in enumerate(axes.flat):

        if state_idx >= num_states:
            ax.axis("off")
            continue

        # Tube region around nominal trajectory.
        ax.fill_between(
            times,
            lower[:, state_idx],
            upper[:, state_idx],
            alpha=0.20,
            label="SLS tube",
        )

        # Tube boundaries.
        ax.plot(
            times,
            lower[:, state_idx],
            "--",
            linewidth=0.8,
            alpha=0.65,
        )
        ax.plot(
            times,
            upper[:, state_idx],
            "--",
            linewidth=0.8,
            alpha=0.65,
        )

        # Monte Carlo closed-loop rollouts.
        first_rollout = True

        for rollout_idx in rollout_indices:
            rollout = X_rollouts[
                rollout_idx
            ]

            valid_count = finite_rollout_prefix(
                rollout
            )

            if valid_count < 1:
                continue

            ax.plot(
                times[:valid_count],
                rollout[
                    :valid_count,
                    state_idx,
                ],
                linewidth=ROLLOUT_LINEWIDTH,
                alpha=ROLLOUT_ALPHA,
                label=(
                    "feedback rollouts"
                    if first_rollout
                    else None
                ),
            )

            first_rollout = False

        # Nominal trajectory on top.
        ax.plot(
            times,
            X_nominal[:, state_idx],
            linewidth=2.0,
            label="nominal",
            zorder=5,
        )

        # Phase boundaries.
        if phase_end_steps is not None:
            for step in phase_end_steps:
                if 0 <= step < num_nodes:
                    ax.axvline(
                        times[step],
                        linewidth=0.6,
                        alpha=0.25,
                    )

        ax.set_title(
            f"{state_idx}: {names[state_idx]}",
            fontsize=8,
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        if state_idx % N_COLS == 0:
            ax.set_ylabel("state value")

        if state_idx >= (
            n_rows - 1
        ) * N_COLS:
            ax.set_xlabel(
                "time [s]"
            )

    # Collect a clean shared legend.
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li and li not in labels:
                handles.append(hi)
                labels.append(li)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    fig.suptitle(
        f"Nominal trajectory, SLS tubes, and "
        f"{len(rollout_indices)} random-disturbance feedback rollouts",
        y=0.999,
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.992)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    physical_error = np.abs(
        X_rollouts - X_nominal[None, :, :]
    )

    valid_tube = tubes > 1.0e-12

    ratios = np.full_like(
        physical_error,
        np.nan,
        dtype=float,
    )

    ratios[:, valid_tube] = (
        physical_error[:, valid_tube]
        / tubes[valid_tube][None, :]
    )

    if np.any(np.isfinite(ratios)):
        flat = int(
            np.nanargmax(ratios)
        )

        (
            worst_rollout,
            worst_node,
            worst_state,
        ) = np.unravel_index(
            flat,
            ratios.shape,
        )

        print(
            "Worst rollout/tube ratio: "
            f"{ratios[worst_rollout, worst_node, worst_state]:.6f}"
        )
        print(
            "Worst rollout/node/state: "
            f"{worst_rollout} / {worst_node} / "
            f"{worst_state} ({names[worst_state]})"
        )

    print(f"Loaded NPZ: {NPZ_PATH}")
    print(f"Nominal shape: {X_nominal.shape}")
    print(f"Rollout shape: {X_rollouts.shape}")
    print(f"Tube shape:    {tubes.shape}")
    print(f"Saved plot:    {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
