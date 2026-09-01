from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


# ============================================================
# Render two drone tracking rollouts simultaneously.
#
# Each input NPZ must contain:
#   X_closed_loop : (N+1, nx)
#   X_plan        : (M+1, nx)
#   dt_plan       : (M,)
#
# Optional:
#   dt_applied_history  : per-step closed-loop dt values
#   disturbance_mag     : scalar used for the background field
#   waypoint_centers    : (W, 3) waypoint centers
#   waypoint_half_widths: (W, 3) waypoint half-widths
#
# If waypoint_centers is absent, waypoints fall back to drone_sim.GATES
# plus the 1 m exit point used by the RTI racing script.
#
# The MPC planning horizon is intentionally NOT rendered.
# Only the two predetermined/offline racing lines, the closed-loop
# traces, and the current drone positions are shown.
# ============================================================


class DisturbanceFieldParams:
    x_min = -2.0
    x_max = 1.0
    y_min = -8.0
    y_max = 0.0
    kx = 1.0
    ky = 1.0


DISTURBANCE_FIELD = DisturbanceFieldParams()


def load_course_waypoints_from_drone_sim(candidate_dirs=()):
    """Load gate centers from the actual ``drone_sim.py`` used by the race.

    The generated renderer may live outside the project directory, so a plain
    ``from drone_sim import GATES`` is not reliable.  Search beside the input
    NPZ files and upward through their parent directories as well.
    """
    search_dirs = []

    def add_dir(path):
        path = Path(path).expanduser().resolve()
        if path.is_file():
            path = path.parent
        if path not in search_dirs:
            search_dirs.append(path)

    add_dir(Path.cwd())
    add_dir(Path(__file__).resolve().parent)

    for candidate in candidate_dirs:
        base = Path(candidate).expanduser().resolve()
        if base.is_file():
            base = base.parent
        for d in [base, *list(base.parents)[:6]]:
            add_dir(d)

    # First try ordinary imports after temporarily exposing candidate folders.
    for d in search_dirs:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))

    try:
        module = importlib.import_module("drone_sim")
        if hasattr(module, "GATES"):
            GATES = module.GATES
        else:
            raise AttributeError("drone_sim has no GATES")
    except Exception as first_exc:
        module = None
        GATES = None
        # Then explicitly search for a drone_sim.py file.  This handles the
        # common case where this renderer is copied to another directory.
        for d in search_dirs:
            candidates = [d / "drone_sim.py"]
            # The examples are often nested one or two directories below a
            # repository root, so check a few likely descendants without an
            # expensive recursive walk of the whole checkout.
            candidates += list(d.glob("*/drone_sim.py"))
            candidates += list(d.glob("*/*/drone_sim.py"))

            for source in candidates:
                if not source.is_file():
                    continue
                try:
                    if str(source.parent) not in sys.path:
                        sys.path.insert(0, str(source.parent))
                    spec = importlib.util.spec_from_file_location(
                        "_two_drone_race_drone_sim", source
                    )
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    GATES = module.GATES
                    print(f"Loaded waypoint course from: {source}")
                    break
                except Exception as exc:
                    print(f"Tried {source}, but loading failed: {exc}")
            if GATES is not None:
                break

        if GATES is None:
            searched = "\n  ".join(str(d) for d in search_dirs)
            raise RuntimeError(
                "Neither NPZ contains waypoint centers, and the renderer could "
                "not load drone_sim.GATES.\n"
                f"Initial import error: {first_exc}\n"
                "Searched these directories:\n  " + searched
            )

    gate_centers = np.asarray(
        [gate.position for gate in GATES],
        dtype=float,
    )

    if gate_centers.ndim != 2 or gate_centers.shape[1] < 3 or len(gate_centers) == 0:
        raise RuntimeError(
            f"drone_sim.GATES did not provide valid 3-D gate centers: "
            f"shape={gate_centers.shape}"
        )

    # Same exit point used by the supplied RTI race script.
    exit_point = gate_centers[-1] + np.array([1.0, 0.0, 0.0])
    waypoints = np.vstack((gate_centers, exit_point))

    print(f"Using {len(gate_centers)} gates + 1 exit waypoint:")
    for i, wp in enumerate(waypoints):
        name = f"Gate {i + 1}" if i < len(gate_centers) else "Exit"
        print(f"  {name:>7s}: x={wp[0]: .3f}, y={wp[1]: .3f}, z={wp[2]: .3f}")

    return waypoints


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
        required = ["X_closed_loop", "X_plan", "dt_plan"]

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


def build_node_times(dt_values):
    dt_values = np.asarray(dt_values, dtype=float).reshape(-1)
    return np.concatenate([
        np.array([0.0]),
        np.cumsum(dt_values),
    ])


def closed_loop_times(result):
    """Build one timestamp for every row of X_closed_loop."""
    X = np.asarray(result["X_closed_loop"], dtype=float)
    dt_plan = np.asarray(result["dt_plan"], dtype=float).reshape(-1)
    n_steps = X.shape[0] - 1

    if n_steps < 0:
        raise ValueError("X_closed_loop must contain at least one state.")

    if n_steps == 0:
        return np.array([0.0])

    if len(dt_plan) == 0:
        raise ValueError("dt_plan must contain at least one time step.")

    if "dt_applied_history" in result:
        dt = np.asarray(
            result["dt_applied_history"],
            dtype=float,
        ).reshape(-1)
        dt = dt[:n_steps]

        if len(dt) < n_steps:
            fill_dt = dt[-1] if len(dt) > 0 else dt_plan[-1]
            dt = np.pad(
                dt,
                (0, n_steps - len(dt)),
                constant_values=fill_dt,
            )
    else:
        if n_steps <= len(dt_plan):
            dt = dt_plan[:n_steps]
        else:
            dt = np.concatenate([
                dt_plan,
                np.full(n_steps - len(dt_plan), dt_plan[-1]),
            ])

    if not np.all(np.isfinite(dt)):
        raise ValueError("Closed-loop time steps contain NaN/Inf values.")
    if np.any(dt <= 0.0):
        raise ValueError("Closed-loop time steps must all be positive.")

    return build_node_times(dt)


def prepare_drone(result, label):
    X = np.asarray(result["X_closed_loop"], dtype=float)
    X_plan = np.asarray(result["X_plan"], dtype=float)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(
            f"{label}: X_closed_loop must have shape (N, >=2); got {X.shape}."
        )

    if X_plan.ndim != 2 or X_plan.shape[1] < 2:
        raise ValueError(
            f"{label}: X_plan must have shape (N, >=2); got {X_plan.shape}."
        )

    times = closed_loop_times(result)

    if len(times) != len(X):
        raise RuntimeError(
            f"{label}: got {len(times)} timestamps for {len(X)} states."
        )

    finite_xy = np.all(np.isfinite(X[:, :2]), axis=1)
    if not np.any(finite_xy):
        raise ValueError(f"{label}: no finite x/y states in X_closed_loop.")

    # Truncate only at the first non-finite x/y row, if one exists.
    bad = np.flatnonzero(~finite_xy)
    if len(bad) > 0:
        stop = int(bad[0])
        if stop == 0:
            raise ValueError(f"{label}: first closed-loop state is non-finite.")
        X = X[:stop]
        times = times[:stop]
        print(f"{label}: truncated at first non-finite state, index {stop}.")

    finite_plan = np.all(np.isfinite(X_plan[:, :2]), axis=1)
    X_plan = X_plan[finite_plan]

    if len(X_plan) == 0:
        raise ValueError(f"{label}: no finite x/y states in X_plan.")

    disturbance_mag = float(
        np.asarray(result.get("disturbance_mag", 0.0)).reshape(())
    )

    waypoint_centers = np.zeros((0, 3), dtype=float)
    waypoint_key = None
    for key in ("waypoint_centers", "gate_centers", "waypoints", "waypoint_positions"):
        if key in result:
            candidate = np.asarray(result[key], dtype=float)
            if candidate.size > 0:
                waypoint_centers = candidate
                waypoint_key = key
                break

    waypoint_half_widths = np.asarray(
        result.get("waypoint_half_widths", np.zeros((0, 3))),
        dtype=float,
    )

    if waypoint_key is not None:
        print(f"{label}: loaded {len(waypoint_centers)} waypoints from NPZ key '{waypoint_key}'.")
    else:
        print(f"{label}: NPZ contains no waypoint/gate-center key.")

    if waypoint_centers.ndim != 2 or (
        waypoint_centers.shape[0] > 0 and waypoint_centers.shape[1] < 2
    ):
        raise ValueError(
            f"{label}: waypoint_centers must have shape (W, >=2); "
            f"got {waypoint_centers.shape}."
        )

    return {
        "label": label,
        "X": X,
        "X_plan": X_plan,
        "times": times,
        "duration": float(times[-1]),
        "disturbance_mag": disturbance_mag,
        "waypoint_centers": waypoint_centers,
        "waypoint_half_widths": waypoint_half_widths,
    }


def resolve_course_waypoints(drone_a, drone_b, candidate_dirs=()):
    """Resolve one course waypoint set before the figure bounds are chosen."""
    wp_a = drone_a["waypoint_centers"]
    wp_b = drone_b["waypoint_centers"]
    hw_a = drone_a["waypoint_half_widths"]
    hw_b = drone_b["waypoint_half_widths"]

    if len(wp_a) == 0 and len(wp_b) == 0:
        wp = load_course_waypoints_from_drone_sim(candidate_dirs)
        # Use the gate tolerance from the supplied RTI race script for a clear
        # top-down gate target box.  The final exit point gets the same box.
        hw = np.tile(np.array([0.55, 0.55, 0.0]), (len(wp), 1))
        return wp, hw, wp.copy(), hw.copy()

    if len(wp_a) == 0:
        print(f"{drone_a['label']}: reusing {drone_b['label']}'s NPZ waypoints.")
        wp_a = wp_b.copy()
        hw_a = hw_b.copy()

    if len(wp_b) == 0:
        print(f"{drone_b['label']}: reusing {drone_a['label']}'s NPZ waypoints.")
        wp_b = wp_a.copy()
        hw_b = hw_a.copy()

    for label, wp in ((drone_a['label'], wp_a), (drone_b['label'], wp_b)):
        print(f"{label} waypoint coordinates:")
        for i, p in enumerate(wp):
            z = p[2] if p.shape[0] > 2 else float('nan')
            print(f"  WP {i + 1:02d}: x={p[0]: .3f}, y={p[1]: .3f}, z={z: .3f}")

    return wp_a, hw_a, wp_b, hw_b


def position_at_time(drone, t):
    """Linearly interpolate x/y at simulated time t; hold final pose afterward."""
    times = drone["times"]
    xy = drone["X"][:, :2]

    if t <= times[0]:
        return xy[0].copy()
    if t >= times[-1]:
        return xy[-1].copy()

    idx_hi = int(np.searchsorted(times, t, side="right"))
    idx_lo = idx_hi - 1

    t0 = times[idx_lo]
    t1 = times[idx_hi]
    alpha = (t - t0) / (t1 - t0)

    return (1.0 - alpha) * xy[idx_lo] + alpha * xy[idx_hi]


def trace_through_time(drone, t):
    """Return completed closed-loop nodes plus the interpolated current point."""
    times = drone["times"]
    xy = drone["X"][:, :2]

    if t >= times[-1]:
        return xy

    idx = int(np.searchsorted(times, t, side="right"))
    current = position_at_time(drone, t)

    if idx <= 0:
        return current[None, :]

    return np.vstack([xy[:idx], current])


def make_animation(
    result_a,
    result_b,
    output="two_drone_race.mp4",
    fps=20,
    playback_speed=1.0,
    name_a="Drone 1",
    name_b="Drone 2",
    show_disturbance=True,
    waypoint_search_dirs=(),
):
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive.")

    drone_a = prepare_drone(result_a, name_a)
    drone_b = prepare_drone(result_b, name_b)

    wp_a, hw_a, wp_b, hw_b = resolve_course_waypoints(
        drone_a, drone_b, candidate_dirs=waypoint_search_dirs
    )

    total_duration = max(drone_a["duration"], drone_b["duration"])

    # 1.0x means one simulated second is one video second.
    video_duration = total_duration / playback_speed
    n_frames = max(2, int(np.ceil(video_duration * fps)) + 1)
    frame_times = np.linspace(0.0, total_duration, n_frames)

    print(f"{name_a}: {len(drone_a['X'])} states, {drone_a['duration']:.3f} s")
    print(f"{name_b}: {len(drone_b['X'])} states, {drone_b['duration']:.3f} s")
    print(f"Combined simulated duration: {total_duration:.3f} s")
    print(f"Video duration at {playback_speed:.3f}x: {video_duration:.3f} s")
    print(f"Frames: {n_frames} at {fps} fps")

    fig, ax = plt.subplots(figsize=(9, 8))

    xy_parts = [
        drone_a["X"][:, :2],
        drone_b["X"][:, :2],
        drone_a["X_plan"][:, :2],
        drone_b["X_plan"][:, :2],
    ]
    if len(wp_a) > 0:
        xy_parts.append(wp_a[:, :2])
    if len(wp_b) > 0:
        xy_parts.append(wp_b[:, :2])
    all_xy = np.vstack(xy_parts)

    data_x_min = float(np.min(all_xy[:, 0]))
    data_x_max = float(np.max(all_xy[:, 0]))
    data_y_min = float(np.min(all_xy[:, 1]))
    data_y_max = float(np.max(all_xy[:, 1]))

    # Preserve the original scripts' field of view, but expand if either race
    # extends outside it.
    x_min = min(-6.0, data_x_min - 0.75)
    x_max = max(5.0, data_x_max + 0.75)
    y_min = min(-7.5, data_y_min - 0.75)
    y_max = max(8.0, data_y_max + 0.75)

    if show_disturbance:
        display_mag = max(
            drone_a["disturbance_mag"],
            drone_b["disturbance_mag"],
            2.5,
        )

        xs = np.linspace(x_min, x_max, 300)
        ys = np.linspace(y_min, y_max, 300)
        XX, YY = np.meshgrid(xs, ys)

        field_values = (
            display_mag
            * disturbance_spatial_scale(XX, YY)
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
            levels=np.linspace(0.0, display_mag, 100),
            cmap=cmap,
            vmin=0.0,
            vmax=display_mag,
            alpha=0.80,
            zorder=0,
        )

        cbar = fig.colorbar(field, ax=ax, pad=0.02)
        cbar.set_label(r"Disturbance magnitude on $v_y$")

    # Static waypoints / waypoint tolerance boxes.
    # Match the supplied single-drone renderer exactly: one square marker at
    # each saved waypoint center, a numeric annotation, and the saved
    # waypoint-half-width rectangle.  Prefer the NPZ data directly.
    if len(wp_a) > 0:
        waypoint_centers = wp_a
        waypoint_half_widths = hw_a
        waypoint_source = name_a
    elif len(wp_b) > 0:
        waypoint_centers = wp_b
        waypoint_half_widths = hw_b
        waypoint_source = name_b
    else:
        waypoint_centers = np.zeros((0, 3), dtype=float)
        waypoint_half_widths = np.zeros((0, 3), dtype=float)
        waypoint_source = "none"

    print(f"Waypoint source used for rendering: {waypoint_source}")
    print(f"waypoint_centers shape: {waypoint_centers.shape}")
    print(f"waypoint_half_widths shape: {waypoint_half_widths.shape}")

    if len(waypoint_centers) > 0:
        print("Rendered waypoint coordinates:")
        for i, waypoint in enumerate(waypoint_centers):
            z = waypoint[2] if waypoint.shape[0] > 2 else float("nan")
            print(
                f"  {i + 1:02d}: x={waypoint[0]: .3f}, "
                f"y={waypoint[1]: .3f}, z={z: .3f}"
            )

        ax.scatter(
            waypoint_centers[:, 0],
            waypoint_centers[:, 1],
            marker="s",
            s=80,
            label="Waypoints",
            zorder=7,
        )

        for i, waypoint in enumerate(waypoint_centers):
            ax.annotate(
                str(i + 1),
                (waypoint[0], waypoint[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                zorder=8,
            )

    if (
        len(waypoint_centers) > 0
        and waypoint_half_widths.ndim == 2
        and waypoint_half_widths.shape[0] == waypoint_centers.shape[0]
        and waypoint_half_widths.shape[1] >= 2
    ):
        for center, hw in zip(waypoint_centers, waypoint_half_widths):
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

    # Static predetermined racing lines.
    ax.plot(
        drone_a["X_plan"][:, 0],
        drone_a["X_plan"][:, 1],
        "--",
        linewidth=2.0,
        color="tab:blue",
        label=f"{name_a} racing line",
        zorder=3,
    )

    ax.plot(
        drone_b["X_plan"][:, 0],
        drone_b["X_plan"][:, 1],
        "--",
        linewidth=2.0,
        color="tab:green",
        label=f"{name_b} racing line",
        zorder=3,
    )

    # Animated closed-loop traces.
    trace_a, = ax.plot(
        [],
        [],
        linewidth=2.8,
        color="tab:blue",
        label=f"{name_a} actual",
        zorder=5,
    )

    trace_b, = ax.plot(
        [],
        [],
        linewidth=2.8,
        color="tab:green",
        label=f"{name_b} actual",
        zorder=5,
    )

    marker_a, = ax.plot(
        [],
        [],
        marker="o",
        markersize=10,
        linestyle="None",
        color="tab:blue",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=name_a,
        zorder=9,
    )

    marker_b, = ax.plot(
        [],
        [],
        marker="o",
        markersize=10,
        linestyle="None",
        color="tab:green",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=name_b,
        zorder=9,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    title = ax.set_title("")

    def update(frame_number):
        t = float(frame_times[frame_number])

        xy_a = position_at_time(drone_a, t)
        xy_b = position_at_time(drone_b, t)

        hist_a = trace_through_time(drone_a, t)
        hist_b = trace_through_time(drone_b, t)

        trace_a.set_data(hist_a[:, 0], hist_a[:, 1])
        trace_b.set_data(hist_b[:, 0], hist_b[:, 1])

        marker_a.set_data([xy_a[0]], [xy_a[1]])
        marker_b.set_data([xy_b[0]], [xy_b[1]])

        status_a = (
            "finished"
            if t >= drone_a["duration"]
            else "racing"
        )
        status_b = (
            "finished"
            if t >= drone_b["duration"]
            else "racing"
        )

        title.set_text(
            "Two-Drone Racing Comparison\n"
            f"sim time {t:.3f} s   "
            f"{name_a}: {status_a}   "
            f"{name_b}: {status_b}"
        )

        return (
            trace_a,
            trace_b,
            marker_a,
            marker_b,
            title,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    output_path = Path(output).expanduser().resolve()
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        if suffix != ".mp4":
            output_path = output_path.with_suffix(".mp4")

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
    print(f"Saved animation to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render two drone tracking rollouts simultaneously with their "
            "predetermined/offline racing lines. MPC horizons are not shown."
        )
    )

    parser.add_argument(
        "input_a",
        nargs="?",
        default="fatrop_mpc_tracking.npz",
        help="First tracking NPZ.",
    )

    parser.add_argument(
        "input_b",
        nargs="?",
        default="multiphase_trajectory_tracking.npz",
        help="Second tracking NPZ.",
    )

    parser.add_argument(
        "--output",
        default="two_drone_race.mp4",
        help="Output .mp4 or .gif (default: two_drone_race.mp4).",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed. 1.0 means one simulated second per video second; "
            "2.0 is twice as fast."
        ),
    )

    parser.add_argument(
        "--name-a",
        default="Drone 1",
        help="Legend/title name for the first drone.",
    )

    parser.add_argument(
        "--name-b",
        default="Drone 2",
        help="Legend/title name for the second drone.",
    )

    parser.add_argument(
        "--no-field",
        action="store_true",
        help="Disable the disturbance-field background.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_a = Path(args.input_a).expanduser().resolve()
    input_b = Path(args.input_b).expanduser().resolve()
    result_a = load_result(input_a)
    result_b = load_result(input_b)

    make_animation(
        result_a=result_a,
        result_b=result_b,
        output=args.output,
        fps=args.fps,
        playback_speed=args.speed,
        name_a=args.name_a,
        name_b=args.name_b,
        show_disturbance=not args.no_field,
        waypoint_search_dirs=(input_a.parent, input_b.parent),
    )


if __name__ == "__main__":
    main()