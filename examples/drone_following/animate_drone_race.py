from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Polygon
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
#   waypoint_axis_u
#   waypoint_normals
#   gate_centers
#   gate_normals
#   gate_axis_u
#   gate_opening_half_widths
#   gate_bar_thickness
#   gate_depth
#
# If explicit gate geometry is absent, it is reconstructed from consecutive
# [pre_gate, post_gate] waypoint pairs using the planner's gate convention.
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

    # --------------------------------------------------
    # Oriented waypoint/gate geometry
    # --------------------------------------------------
    # The planning script saves these arrays directly. If the tracking NPZ
    # only copied waypoint centers/half-widths, reconstruct the same gate
    # frames from each [pre_i, post_i] waypoint pair.
    waypoint_axis_u = result.get("waypoint_axis_u", None)
    waypoint_normals = result.get("waypoint_normals", None)

    gate_centers = result.get("gate_centers", None)
    gate_normals = result.get("gate_normals", None)
    gate_axis_u = result.get("gate_axis_u", None)
    gate_opening_half_widths = result.get(
        "gate_opening_half_widths",
        None,
    )

    gate_bar_thickness = float(
        np.asarray(
            result.get("gate_bar_thickness", 0.10)
        ).reshape(())
    )
    gate_depth = float(
        np.asarray(
            result.get("gate_depth", 0.10)
        ).reshape(())
    )

    if waypoint_axis_u is not None:
        waypoint_axis_u = np.asarray(waypoint_axis_u, dtype=float)
    if waypoint_normals is not None:
        waypoint_normals = np.asarray(waypoint_normals, dtype=float)

    if gate_centers is not None:
        gate_centers = np.asarray(gate_centers, dtype=float)
    if gate_normals is not None:
        gate_normals = np.asarray(gate_normals, dtype=float)
    if gate_axis_u is not None:
        gate_axis_u = np.asarray(gate_axis_u, dtype=float)
    if gate_opening_half_widths is not None:
        gate_opening_half_widths = np.asarray(
            gate_opening_half_widths,
            dtype=float,
        )

    # Reconstruct the five physical gates from the ten transition waypoints
    # when the explicit gate arrays are not present. The planner constructs
    # waypoint pairs as pre_i = c_i - offset*n_i and
    # post_i = c_i + offset*n_i, so their midpoint/difference recovers c_i,n_i.
    has_pre_post_pairs = (
        waypoint_centers.ndim == 2
        and waypoint_centers.shape[1] >= 3
        and waypoint_centers.shape[0] >= 2
        and waypoint_centers.shape[0] % 2 == 0
    )

    if has_pre_post_pairs:
        pre = waypoint_centers[0::2, :3]
        post = waypoint_centers[1::2, :3]

        inferred_gate_centers = 0.5 * (pre + post)
        inferred_normals = post - pre
        inferred_normals[:, 2] = 0.0
        inferred_norm = np.linalg.norm(
            inferred_normals[:, :2],
            axis=1,
            keepdims=True,
        )
        inferred_normals = inferred_normals / np.maximum(
            inferred_norm,
            1e-12,
        )

        inferred_axis_u = np.column_stack([
            -inferred_normals[:, 1],
            inferred_normals[:, 0],
            np.zeros(len(inferred_normals)),
        ])

        if gate_centers is None:
            gate_centers = inferred_gate_centers
        if gate_normals is None:
            gate_normals = inferred_normals
        if gate_axis_u is None:
            gate_axis_u = inferred_axis_u

        if waypoint_normals is None:
            waypoint_normals = np.repeat(
                inferred_normals,
                2,
                axis=0,
            )
        if waypoint_axis_u is None:
            waypoint_axis_u = np.repeat(
                inferred_axis_u,
                2,
                axis=0,
            )

        if gate_opening_half_widths is None:
            # Same physical opening used in the planning script: 1 m x 1 m.
            gate_opening_half_widths = np.tile(
                np.array([[0.50, 0.50]], dtype=float),
                (len(inferred_gate_centers), 1),
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

    # --------------------------------------------------
    # Physical closed-loop timestamps
    # --------------------------------------------------
    # Prefer an explicitly saved time vector. Otherwise reconstruct it from
    # the dt values actually applied by the simulated plant.
    if "t_closed_loop" in result:
        node_times = np.asarray(
            result["t_closed_loop"],
            dtype=float,
        ).reshape(-1)

        if len(node_times) != len(X):
            raise ValueError(
                "t_closed_loop must have one timestamp per closed-loop state "
                f"({len(X)} expected, got {len(node_times)})."
            )

    elif "dt_applied_history" in result:
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
        # Backward-compatible fallback for older tracking files.
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

    if len(node_times) != len(X):
        raise ValueError(
            "Closed-loop timestamp vector must match X_closed_loop."
        )

    if not np.all(np.isfinite(node_times)):
        raise ValueError(
            "Closed-loop timestamps contain NaN or Inf."
        )

    if np.any(np.diff(node_times) < 0.0):
        raise ValueError(
            "Closed-loop timestamps must be monotonically nondecreasing."
        )

    total_execution_time = float(node_times[-1])

    if total_execution_time <= 0.0:
        raise ValueError(
            "Closed-loop execution time must be positive."
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

    # ==================================================
    # Physical gates -- same top-down rendering as the planning script
    # ==================================================
    gate_handle = None

    if (
        gate_centers is not None
        and gate_normals is not None
        and gate_axis_u is not None
        and gate_opening_half_widths is not None
        and len(gate_centers) > 0
    ):
        for i, center in enumerate(gate_centers):
            u_xy = np.asarray(gate_axis_u[i, :2], dtype=float)
            n_xy = np.asarray(gate_normals[i, :2], dtype=float)

            u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
            n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

            hu_open = float(gate_opening_half_widths[i, 0])
            half_outer_width = hu_open + gate_bar_thickness
            half_depth = 0.5 * gate_depth

            c = np.asarray(center[:2], dtype=float)
            corners = np.array([
                c - half_outer_width * u_xy - half_depth * n_xy,
                c + half_outer_width * u_xy - half_depth * n_xy,
                c + half_outer_width * u_xy + half_depth * n_xy,
                c - half_outer_width * u_xy + half_depth * n_xy,
            ])

            gate_patch = Polygon(
                corners,
                closed=True,
                facecolor="none",
                edgecolor="black",
                linewidth=2.5,
                zorder=7,
            )
            ax.add_patch(gate_patch)

            if gate_handle is None:
                gate_handle = gate_patch

            ax.scatter(
                center[0],
                center[1],
                marker="x",
                s=45,
                color="black",
                zorder=8,
            )

            ax.annotate(
                f"G{i + 1}",
                (center[0], center[1]),
                xytext=(5, -14),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=9,
            )

    # ==================================================
    # Transition waypoints -- same markers as the planning script
    # ==================================================
    if len(waypoint_centers) > 0:
        ax.scatter(
            waypoint_centers[:, 0],
            waypoint_centers[:, 1],
            marker="s",
            s=90,
            label="Waypoints",
            zorder=7,
        )

        for i, waypoint in enumerate(waypoint_centers):
            ax.annotate(
                str(i + 1),
                (waypoint[0], waypoint[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                zorder=8,
            )

    # ==================================================
    # Oriented waypoint boxes -- local [u, v, n] frame
    # ==================================================
    if (
        len(waypoint_centers) > 0
        and waypoint_half_widths.shape[0] == waypoint_centers.shape[0]
    ):
        if (
            waypoint_axis_u is not None
            and waypoint_normals is not None
            and waypoint_axis_u.shape[0] == waypoint_centers.shape[0]
            and waypoint_normals.shape[0] == waypoint_centers.shape[0]
        ):
            for center, hw, u_axis, n_axis in zip(
                waypoint_centers,
                waypoint_half_widths,
                waypoint_axis_u,
                waypoint_normals,
            ):
                c = np.asarray(center[:2], dtype=float)
                u_xy = np.asarray(u_axis[:2], dtype=float)
                n_xy = np.asarray(n_axis[:2], dtype=float)

                u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
                n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

                hu = float(hw[0])
                hn = float(hw[2])

                corners = np.array([
                    c - hu * u_xy - hn * n_xy,
                    c + hu * u_xy - hn * n_xy,
                    c + hu * u_xy + hn * n_xy,
                    c - hu * u_xy + hn * n_xy,
                ])

                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        fill=False,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=6,
                    )
                )
        else:
            # Backward-compatible fallback for older NPZ files.
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
                        linewidth=1.5,
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

    handles, labels = ax.get_legend_handles_labels()
    if gate_handle is not None:
        handles.append(gate_handle)
        labels.append("Physical gate")

    ax.legend(
        handles,
        labels,
        loc="upper right",
    )

    title = ax.set_title("")

    # ==================================================
    # REAL-TIME animation clock
    # ==================================================
    # The old renderer advanced one controller node per video frame, so movie
    # duration depended on the number of MPC iterations. This renderer instead
    # advances according to the plant's accumulated physical execution time.
    #
    # At 1x playback:
    #
    #     visual/video time == simulated execution time
    #
    # The state marker is interpolated between neighboring closed-loop states.
    # The MPC horizon remains piecewise constant and uses the most recently
    # completed controller update.
    stride = max(1, int(frame_stride))
    requested_fps = float(fps)

    if requested_fps <= 0.0:
        raise ValueError(
            "fps must be positive."
        )

    render_fps_target = requested_fps / stride

    num_frames = max(
        2,
        int(round(total_execution_time * render_fps_target)),
    )

    # Choose the actual encoded FPS so video duration is exactly the simulated
    # execution duration. This adjustment is normally tiny.
    writer_fps = (
        num_frames / total_execution_time
    )

    frame_times = (
        np.arange(num_frames, dtype=float)
        / writer_fps
    )

    # Make the final frame explicitly show the terminal closed-loop state.
    frame_times[-1] = total_execution_time

    print(
        "\n================ ANIMATION TIMING ==================="
    )
    print(
        f"Closed-loop execution time: {total_execution_time:.6f} s"
    )
    print(
        f"Requested FPS:             {requested_fps:.3f}"
    )
    print(
        f"Frame stride:              {stride}"
    )
    print(
        f"Rendered frames:           {num_frames}"
    )
    print(
        f"Encoded FPS:               {writer_fps:.6f}"
    )
    print(
        f"Encoded video duration:    "
        f"{num_frames / writer_fps:.6f} s"
    )
    print(
        "Playback speed:            1.000x simulated time"
    )
    print(
        "====================================================="
    )

    def state_at_time(current_time):
        """
        Interpolate X_closed_loop at a physical simulation time.
        """
        if current_time <= node_times[0]:
            return X[0].copy(), 0, 0, 0.0

        if current_time >= node_times[-1]:
            return X[-1].copy(), N_closed, N_closed, 0.0

        k1 = int(
            np.searchsorted(
                node_times,
                current_time,
                side="right",
            )
        )

        k1 = int(
            np.clip(
                k1,
                1,
                N_closed,
            )
        )

        k0 = k1 - 1

        t0 = float(node_times[k0])
        t1 = float(node_times[k1])

        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = float(
                np.clip(
                    (current_time - t0) / (t1 - t0),
                    0.0,
                    1.0,
                )
            )

        x_vis = (
            (1.0 - alpha) * X[k0]
            + alpha * X[k1]
        )

        return (
            x_vis,
            k0,
            k1,
            alpha,
        )

    def update(frame_number):
        current_time = float(
            frame_times[frame_number]
        )

        (
            x_vis,
            k0,
            k1,
            alpha,
        ) = state_at_time(
            current_time
        )

        # Horizon/reference information changes only when an MPC solve occurs.
        k_controller = k0

        progress_idx = int(
            progress_idx_history[
                min(
                    k_controller,
                    len(progress_idx_history) - 1,
                )
            ]
        )

        # Plot all completed nodes plus the interpolated current position.
        if (
            k1 > k0
            and alpha > 0.0
        ):
            trace_x = np.concatenate([
                X[:k0 + 1, 0],
                np.array([x_vis[0]]),
            ])

            trace_y = np.concatenate([
                X[:k0 + 1, 1],
                np.array([x_vis[1]]),
            ])
        else:
            trace_end = min(
                k0 + 1,
                len(X),
            )

            trace_x = X[
                :trace_end,
                0,
            ]

            trace_y = X[
                :trace_end,
                1,
            ]

        actual_trace.set_data(
            trace_x,
            trace_y,
        )

        current_marker.set_data(
            [x_vis[0]],
            [x_vis[1]],
        )

        reference_marker.set_data(
            [X_plan[progress_idx, 0]],
            [X_plan[progress_idx, 1]],
        )

        if (
            X_mpc_predictions is not None
            and k_controller < X_mpc_predictions.shape[0]
        ):
            pred = X_mpc_predictions[
                k_controller
            ]

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

        error0 = float(
            pos_errors[
                min(
                    k0,
                    len(pos_errors) - 1,
                )
            ]
        )

        error1 = float(
            pos_errors[
                min(
                    k1,
                    len(pos_errors) - 1,
                )
            ]
        )

        error = (
            (1.0 - alpha) * error0
            + alpha * error1
        )

        title.set_text(
            "Online FATROP MPC Tracking\n"
            f"controller step {k_controller:03d}/{N_closed:03d}   "
            f"progress {progress_idx:03d}/{N_plan:03d}\n"
            f"time {current_time:.3f}/{total_execution_time:.3f} s   "
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
        frames=num_frames,
        interval=1000.0 / writer_fps,
        blit=False,
        repeat=False,
    )


    output_path = Path(
        output
    ).expanduser().resolve()

    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = PillowWriter(
            fps=writer_fps
        )
    else:
        if suffix != ".mp4":
            output_path = output_path.with_suffix(
                ".mp4"
            )

        writer = FFMpegWriter(
            fps=writer_fps,
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
            "Reduce rendered frame rate by this factor while preserving "
            "1x simulated-time playback (default: 1)"
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