from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


# ============================================================
# Animate FATROP MPC tracking result
#
# Expected NPZ keys from fatrop_drone_racing_mpc_tracker.py:
#   X_closed_loop
#   U_closed_loop
#   X_plan
#   U_plan
#   dt_plan
#   waypoint_centers
#   waypoint_half_widths
#   pos_errors
#   mpc_horizon
#   disturbance_mag
#
# Optional:
#   X_mpc_predictions
#
# If X_mpc_predictions is not present, the animation shows the
# moving OFFLINE reference window corresponding to the MPC horizon.
# ============================================================


class DisturbanceFieldParams:
    x_min = -2.0
    x_max = 1.0
    y_min = -8.0
    y_max = 0.0
    kx = 1.0
    ky = 1.0


DISTURBANCE_FIELD = DisturbanceFieldParams()


def disturbance_spatial_scale(px, py, params=DISTURBANCE_FIELD):
    x_window = (
        0.5 * (1.0 + np.tanh(params.kx * (px - params.x_min)))
        * 0.5 * (1.0 - np.tanh(params.kx * (px - params.x_max)))
    )

    y_window = (
        0.5 * (1.0 + np.tanh(params.ky * (py - params.y_min)))
        * 0.5 * (1.0 - np.tanh(params.ky * (py - params.y_max)))
    )

    return x_window * y_window


def load_result(filename: Path):
    filename = filename.expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(filename)

    with np.load(filename, allow_pickle=False) as data:
        required = [
            "X_closed_loop",
            "X_plan",
            "dt_plan",
        ]

        for key in required:
            if key not in data:
                raise ValueError(
                    f"{filename} is missing required key '{key}'"
                )

        result = {
            key: np.asarray(data[key])
            for key in data.files
        }

    return result


def build_node_times(dt_plan):
    dt_plan = np.asarray(dt_plan, dtype=float).reshape(-1)

    return np.concatenate([
        np.array([0.0]),
        np.cumsum(dt_plan),
    ])


def make_animation(
    result,
    output="fatrop_mpc_tracking.mp4",
    fps=20,
    frame_stride=1,
):
    X = np.asarray(
        result["X_closed_loop"],
        dtype=float,
    )

    X_plan = np.asarray(
        result["X_plan"],
        dtype=float,
    )

    dt_plan = np.asarray(
        result["dt_plan"],
        dtype=float,
    ).reshape(-1)

    N_closed = X.shape[0] - 1
    N_plan = X_plan.shape[0] - 1

    horizon = int(
        np.asarray(
            result.get("mpc_horizon", 15)
        ).reshape(())
    )

    disturbance_mag = float(
        np.asarray(
            result.get("disturbance_mag", 0.0)
        ).reshape(())
    )

    waypoint_centers = np.asarray(
        result.get(
            "waypoint_centers",
            np.zeros((0, 3)),
        ),
        dtype=float,
    )

    waypoint_half_widths = np.asarray(
        result.get(
            "waypoint_half_widths",
            np.zeros((0, 3)),
        ),
        dtype=float,
    )

    if "progress_idx_history" in result:
        progress_idx_history = np.asarray(
            result["progress_idx_history"],
            dtype=int,
        ).reshape(-1)

        if len(progress_idx_history) < len(X):
            fill = (
                progress_idx_history[-1]
                if len(progress_idx_history) > 0
                else 0
            )
            progress_idx_history = np.pad(
                progress_idx_history,
                (0, len(X) - len(progress_idx_history)),
                constant_values=fill,
            )
        elif len(progress_idx_history) > len(X):
            progress_idx_history = progress_idx_history[:len(X)]

        progress_idx_history = np.clip(
            progress_idx_history,
            0,
            N_plan,
        )
    else:
        progress_idx_history = np.minimum(
            np.arange(len(X), dtype=int),
            N_plan,
        )

    if "pos_errors" in result:
        pos_errors = np.asarray(
            result["pos_errors"],
            dtype=float,
        ).reshape(-1)
    else:
        X_progress_reference = X_plan[
            progress_idx_history
        ]

        pos_errors = np.linalg.norm(
            X[:, :3]
            - X_progress_reference[:, :3],
            axis=1,
        )

    if len(pos_errors) < len(X):
        fill = (
            pos_errors[-1]
            if len(pos_errors) > 0
            else 0.0
        )
        pos_errors = np.pad(
            pos_errors,
            (0, len(X) - len(pos_errors)),
            constant_values=fill,
        )
    elif len(pos_errors) > len(X):
        pos_errors = pos_errors[:len(X)]

    if "dt_applied_history" in result:
        dt_applied = np.asarray(
            result["dt_applied_history"],
            dtype=float,
        ).reshape(-1)

        dt_applied = dt_applied[:N_closed]

        if len(dt_applied) < N_closed:
            fill_dt = (
                dt_applied[-1]
                if len(dt_applied) > 0
                else dt_plan[-1]
            )
            dt_applied = np.pad(
                dt_applied,
                (0, N_closed - len(dt_applied)),
                constant_values=fill_dt,
            )

        node_times = build_node_times(
            dt_applied
        )
    else:
        if N_closed <= len(dt_plan):
            dt_used = dt_plan[:N_closed]
        else:
            dt_used = np.concatenate([
                dt_plan,
                np.full(
                    N_closed - len(dt_plan),
                    dt_plan[-1],
                ),
            ])

        node_times = build_node_times(
            dt_used
        )

    X_mpc_predictions = result.get(
        "X_mpc_predictions",
        None,
    )

    if X_mpc_predictions is not None:
        X_mpc_predictions = np.asarray(
            X_mpc_predictions,
            dtype=float,
        )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    x_min = -6.0
    x_max = 5.0
    y_min = -7.5
    y_max = 8.0

    display_mag = (
        disturbance_mag
        if disturbance_mag > 0.0
        else 2.5
    )

    xs = np.linspace(
        x_min,
        x_max,
        300,
    )

    ys = np.linspace(
        y_min,
        y_max,
        300,
    )

    XX, YY = np.meshgrid(
        xs,
        ys,
    )

    field_values = (
        display_mag
        * disturbance_spatial_scale(
            XX,
            YY,
        )
    )

    magma = plt.get_cmap("magma")

    cmap = LinearSegmentedColormap.from_list(
        "white_to_magma",
        [
            (0.00, "white"),
            (0.08, magma(0.18)),
            (0.35, magma(0.40)),
            (0.65, magma(0.68)),
            (1.00, magma(1.00)),
        ],
    )

    field = ax.contourf(
        XX,
        YY,
        field_values,
        levels=np.linspace(
            0.0,
            display_mag,
            100,
        ),
        cmap=cmap,
        vmin=0.0,
        vmax=display_mag,
        alpha=0.80,
        zorder=0,
    )

    cbar = fig.colorbar(
        field,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r"Disturbance magnitude on $v_y$"
    )

    if len(waypoint_centers) > 0:
        ax.scatter(
            waypoint_centers[:, 0],
            waypoint_centers[:, 1],
            marker="s",
            s=80,
            label="Waypoints",
            zorder=7,
        )

        for i, waypoint in enumerate(
            waypoint_centers
        ):
            ax.annotate(
                str(i + 1),
                (
                    waypoint[0],
                    waypoint[1],
                ),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                zorder=8,
            )

    if (
        len(waypoint_centers) > 0
        and waypoint_half_widths.shape[0]
        == waypoint_centers.shape[0]
    ):
        for center, hw in zip(
            waypoint_centers,
            waypoint_half_widths,
        ):
            ax.add_patch(
                Rectangle(
                    (
                        center[0] - hw[0],
                        center[1] - hw[1],
                    ),
                    2.0 * hw[0],
                    2.0 * hw[1],
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    zorder=6,
                )
            )

    ax.plot(
        X_plan[:, 0],
        X_plan[:, 1],
        "--",
        linewidth=1.5,
        label="Offline optimized path",
        zorder=3,
    )

    actual_trace, = ax.plot(
        [],
        [],
        linewidth=2.5,
        label="Closed-loop MPC",
        zorder=5,
    )

    horizon_line, = ax.plot(
        [],
        [],
        "-o",
        markersize=3,
        linewidth=2.0,
        label="Current MPC horizon",
        zorder=6,
    )

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=9,
        linestyle="None",
        label="Current state",
        zorder=9,
    )

    reference_marker, = ax.plot(
        [],
        [],
        marker="x",
        markersize=9,
        linestyle="None",
        label="Progress-matched reference",
        zorder=9,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="upper right",
    )

    title = ax.set_title("")

    frame_indices = list(
        range(
            0,
            N_closed + 1,
            max(1, frame_stride),
        )
    )

    if frame_indices[-1] != N_closed:
        frame_indices.append(
            N_closed
        )

    def update(frame_number):
        k = frame_indices[
            frame_number
        ]

        progress_idx = int(
            progress_idx_history[k]
        )

        actual_trace.set_data(
            X[:k + 1, 0],
            X[:k + 1, 1],
        )

        current_marker.set_data(
            [X[k, 0]],
            [X[k, 1]],
        )

        reference_marker.set_data(
            [X_plan[progress_idx, 0]],
            [X_plan[progress_idx, 1]],
        )

        if (
            X_mpc_predictions is not None
            and k < X_mpc_predictions.shape[0]
        ):
            pred = X_mpc_predictions[k]

            finite_rows = np.all(
                np.isfinite(pred[:, :2]),
                axis=1,
            )

            pred = pred[
                finite_rows
            ]

            horizon_line.set_data(
                pred[:, 0],
                pred[:, 1],
            )
        else:
            horizon_end = min(
                progress_idx + horizon,
                N_plan,
            )

            horizon_line.set_data(
                X_plan[
                    progress_idx:horizon_end + 1,
                    0,
                ],
                X_plan[
                    progress_idx:horizon_end + 1,
                    1,
                ],
            )

        error = float(
            pos_errors[
                min(
                    k,
                    len(pos_errors) - 1,
                )
            ]
        )

        current_time = float(
            node_times[
                min(
                    k,
                    len(node_times) - 1,
                )
            ]
        )

        title.set_text(
            "Online FATROP MPC Tracking\n"
            f"controller step {k:03d}/{N_closed:03d}   "
            f"progress {progress_idx:03d}/{N_plan:03d}\n"
            f"time {current_time:.3f} s   "
            f"position error {error:.3f} m   "
            f"horizon {horizon} steps"
        )

        return (
            actual_trace,
            horizon_line,
            current_marker,
            reference_marker,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    output_path = Path(
        output
    ).expanduser().resolve()

    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = PillowWriter(
            fps=fps
        )
    else:
        if suffix != ".mp4":
            output_path = output_path.with_suffix(
                ".mp4"
            )

        writer = FFMpegWriter(
            fps=fps,
            bitrate=4000,
        )

    animation.save(
        output_path,
        writer=writer,
        dpi=160,
    )

    plt.close(fig)

    print(
        f"Saved animation to: {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Animate the offline FATROP trajectory and "
            "online FATROP MPC rollout."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="multiphase_trajectory_tracking.npz",
        help=(
            "Tracking NPZ from fatrop_drone_racing_mpc_tracker.py"
        ),
    )

    parser.add_argument(
        "--output",
        default="fatrop_mpc_tracking.mp4",
        help=(
            "Output .mp4 or .gif "
            "(default: fatrop_mpc_tracking.mp4)"
        ),
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Animate every Nth trajectory node "
            "(default: 1)"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = load_result(
        Path(args.input)
    )

    make_animation(
        result=result,
        output=args.output,
        fps=args.fps,
        frame_stride=args.stride,
    )


if __name__ == "__main__":
    main()
