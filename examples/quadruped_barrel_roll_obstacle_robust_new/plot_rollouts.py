#!/usr/bin/env python3
"""
Plot the saved state-history-feedback rollout against the SLS trajectory tubes.

Expected NPZ keys:
    X
    X_state_feedback_rollout
    trajectory_tubes

Optional keys:
    node_times
    phase_names
    phase_end_steps

Example:
    python3 plot_feedback_rollout_vs_tubes.py \
        quadruped_barrel_roll_obstacle_min_time.npz

Optional:
    python3 plot_feedback_rollout_vs_tubes.py result.npz \
        --output feedback_rollout_vs_tubes.png \
        --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def state_dimension_names(n_states: int) -> list[str]:
    """Names for the 67-state Go2 minimum-time barrel-roll state."""
    n_joints = 12
    n_contact = 4
    n_phases = 6

    names = [
        "base_x",
        "base_y",
        "base_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]

    names += [f"q_{i}" for i in range(n_joints)]

    names += [
        "v_x",
        "v_y",
        "v_z",
        "omega_x",
        "omega_y",
        "omega_z",
    ]

    names += [f"dq_{i}" for i in range(n_joints)]

    names += [
        f"foot_{leg}_{axis}"
        for leg in range(n_contact)
        for axis in ("x", "y", "z")
    ]

    names += [
        f"grf_{leg}_{axis}"
        for leg in range(n_contact)
        for axis in ("x", "y", "z")
    ]

    default_phase_names = [
        "stance",
        "lateral_launch",
        "rolling_flight",
        "post_roll_flight",
        "terminal_posture",
        "posture_hold",
    ]
    names += [f"duration_{name}" for name in default_phase_names[:n_phases]]

    if len(names) != n_states:
        return [f"state_{i}" for i in range(n_states)]

    return names


def align_tubes(tubes: np.ndarray, num_state_nodes: int) -> np.ndarray:
    """
    Make tube array line up with x_0, ..., x_N.

    Handles either:
        tubes.shape[0] == N + 1
    or:
        tubes.shape[0] == N
    """
    if tubes.shape[0] == num_state_nodes:
        return tubes

    if tubes.shape[0] == num_state_nodes - 1:
        return np.concatenate(
            [
                np.zeros((1, tubes.shape[1]), dtype=tubes.dtype),
                tubes,
            ],
            axis=0,
        )

    raise ValueError(
        "Cannot align trajectory_tubes with the state rollout: "
        f"tube nodes={tubes.shape[0]}, "
        f"state nodes={num_state_nodes}."
    )


def make_times(data: np.lib.npyio.NpzFile, num_nodes: int) -> np.ndarray:
    """Use saved physical node times when available."""
    if "node_times" in data.files:
        times = np.asarray(data["node_times"]).reshape(-1)
        if len(times) == num_nodes:
            return times

    return np.arange(num_nodes, dtype=float)


def plot_feedback_rollout_vs_tubes(
    npz_path: Path,
    output_path: Path,
    show: bool = False,
) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        required = [
            "X",
            "X_state_feedback_rollout",
            "trajectory_tubes",
        ]

        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(
                f"Missing required NPZ keys: {missing}\n"
                f"Available keys: {data.files}"
            )

        X_nominal = np.asarray(data["X"])
        X_feedback = np.asarray(data["X_state_feedback_rollout"])
        tubes = np.asarray(data["trajectory_tubes"])

        if X_nominal.shape != X_feedback.shape:
            raise ValueError(
                "Nominal and feedback trajectories have different shapes: "
                f"X={X_nominal.shape}, "
                f"X_state_feedback_rollout={X_feedback.shape}"
            )

        tubes = align_tubes(tubes, X_nominal.shape[0])

        if tubes.shape != X_nominal.shape:
            raise ValueError(
                "Tube shape does not match trajectory shape after alignment: "
                f"tubes={tubes.shape}, X={X_nominal.shape}"
            )

        times = make_times(data, X_nominal.shape[0])

        phase_end_steps = None
        phase_names = None

        if "phase_end_steps" in data.files:
            phase_end_steps = np.asarray(
                data["phase_end_steps"], dtype=int
            ).reshape(-1)

        if "phase_names" in data.files:
            phase_names = [
                str(name) for name in np.asarray(data["phase_names"]).reshape(-1)
            ]

    error = X_feedback - X_nominal
    abs_error = np.abs(error)

    names = state_dimension_names(X_nominal.shape[1])

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    denominator = np.maximum(tubes, 1.0e-12)
    ratio = abs_error / denominator

    # Ignore exact zero-radius entries when computing meaningful ratio.
    valid_tube = tubes > 1.0e-12
    valid_ratio = np.where(valid_tube, ratio, np.nan)

    max_abs_error = float(np.max(abs_error))
    max_tube = float(np.max(tubes))

    if np.any(valid_tube):
        max_ratio = float(np.nanmax(valid_ratio))
        worst_flat = int(np.nanargmax(valid_ratio))
        worst_node, worst_state = np.unravel_index(
            worst_flat, valid_ratio.shape
        )
    else:
        max_ratio = float("nan")
        worst_node = -1
        worst_state = -1

    exceeded = (abs_error > tubes) & valid_tube
    num_exceeded = int(np.sum(exceeded))
    total_checked = int(np.sum(valid_tube))

    print(f"Loaded: {npz_path}")
    print(f"Nominal X shape:          {X_nominal.shape}")
    print(f"Feedback rollout shape:   {X_feedback.shape}")
    print(f"Trajectory tubes shape:   {tubes.shape}")
    print(f"Maximum |feedback - X|:   {max_abs_error:.6e}")
    print(f"Maximum tube radius:      {max_tube:.6e}")
    print(
        f"Tube exceedances:         {num_exceeded}/{total_checked}"
    )

    if worst_node >= 0:
        print(
            "Worst error/tube ratio:    "
            f"{max_ratio:.6e} at node {worst_node}, "
            f"state {worst_state} ({names[worst_state]})"
        )

    # ------------------------------------------------------------
    # Per-state figure
    # ------------------------------------------------------------
    n_states = X_nominal.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_states / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 2.9 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        if i >= n_states:
            ax.axis("off")
            continue

        ax.plot(
            times,
            abs_error[:, i],
            label=r"$|x^{fb}-x^{nom}|$",
            linewidth=1.5,
        )
        ax.plot(
            times,
            tubes[:, i],
            "--",
            label="SLS tube radius",
            linewidth=1.3,
        )

        # Highlight nodes outside the predicted tube.
        outside = (abs_error[:, i] > tubes[:, i]) & (
            tubes[:, i] > 1.0e-12
        )
        if np.any(outside):
            ax.scatter(
                times[outside],
                abs_error[outside, i],
                s=12,
                zorder=3,
                label="outside tube",
            )

        # Phase boundaries.
        if phase_end_steps is not None:
            for step in phase_end_steps:
                if 0 <= step < len(times):
                    ax.axvline(
                        times[step],
                        linewidth=0.6,
                        alpha=0.25,
                    )

        ax.set_title(f"{i}: {names[i]}", fontsize=8)
        ax.grid(True, alpha=0.25)

        if i % n_cols == 0:
            ax.set_ylabel("magnitude")

        if i >= (n_rows - 1) * n_cols:
            if "node_times" in data.files:
                ax.set_xlabel("time [s]")
            else:
                ax.set_xlabel("node")

    # Use legend entries from the first subplot.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(labels)),
    )

    fig.suptitle(
        "State-history-feedback rollout error vs. SLS trajectory tubes",
        y=0.999,
        fontsize=14,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.992))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


NPZ_PATH = Path(
    "/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/gpu_sls/examples/quadruped_barrel_roll_obstacle_robust_new/quadruped_barrel_roll_obstacle_min_time.npz"
)

OUTPUT_PATH = NPZ_PATH.parent / "quadruped_barrel_roll_feedback_rollout_vs_tubes.png"


def main():
    plot_feedback_rollout_vs_tubes(
        NPZ_PATH,
        OUTPUT_PATH,
        show=False,
    )


if __name__ == "__main__":
    main()
