import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Polygon


def _normalize_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n < eps):
        raise ValueError("Encountered a zero-length gate axis.")
    return x / n


def _build_gate_axes_from_path(start_position, gate_centers):
    """
    Reconstruct the same vertical gate frames used by the optimizer:
      n = incoming XY direction
      v = world +z
      u = v x n
    """
    start_position = np.asarray(start_position, dtype=float)
    gate_centers = np.asarray(gate_centers, dtype=float)

    previous = np.concatenate(
        [start_position[None, :], gate_centers[:-1]],
        axis=0,
    )

    incoming = gate_centers - previous
    normals = incoming.copy()
    normals[:, 2] = 0.0
    normals = _normalize_rows(normals)

    axis_v = np.broadcast_to(
        np.array([0.0, 0.0, 1.0]),
        normals.shape,
    ).copy()

    axis_u = np.cross(axis_v, normals)
    axis_u = _normalize_rows(axis_u)

    return normals, axis_u, axis_v


def _gate_bar_geometry_xy(
    center,
    normal,
    axis_u,
    opening_half_width,
    opening_half_height,
    bar_thickness,
    gate_depth,
    inflation=0.0,
):
    """
    Return the four gate bars as top-down oriented rectangles.

    In top view, top and bottom bars overlap in XY because their separation is
    purely vertical. This is correct for a top-down projection.
    """
    c = np.asarray(center[:2], dtype=float)
    u = np.asarray(axis_u[:2], dtype=float)
    n = np.asarray(normal[:2], dtype=float)

    u = u / max(np.linalg.norm(u), 1e-12)
    n = n / max(np.linalg.norm(n), 1e-12)

    hu = float(opening_half_width)
    _ = float(opening_half_height)
    t = float(bar_thickness)
    half_depth = 0.5 * float(gate_depth)
    inflation = float(inflation)

    tb_half_u = hu + t + inflation
    tb_half_n = half_depth + inflation

    lr_half_u = 0.5 * t + inflation
    lr_half_n = half_depth + inflation

    top_center = c.copy()
    bottom_center = c.copy()
    left_center = c - (hu + 0.5 * t) * u
    right_center = c + (hu + 0.5 * t) * u

    return {
        "top": (top_center, tb_half_u, tb_half_n),
        "bottom": (bottom_center, tb_half_u, tb_half_n),
        "left": (left_center, lr_half_u, lr_half_n),
        "right": (right_center, lr_half_u, lr_half_n),
    }


def _oriented_rectangle(center, axis_u, axis_n, half_u, half_n):
    center = np.asarray(center, dtype=float)
    u = np.asarray(axis_u[:2], dtype=float)
    n = np.asarray(axis_n[:2], dtype=float)

    u = u / max(np.linalg.norm(u), 1e-12)
    n = n / max(np.linalg.norm(n), 1e-12)

    return np.array([
        center - half_u * u - half_n * n,
        center + half_u * u - half_n * n,
        center + half_u * u + half_n * n,
        center - half_u * u + half_n * n,
    ])


def animate_nominal_topdown(
    npz_path="multiphase_trajectory.npz",
    save_path="nominal_topdown.mp4",
    fps=30,
    trail=True,
    show_reference=True,
    show_transition_waypoints=True,
    show_collision_inflation=False,
    show=False,
):
    data = np.load(npz_path)

    X = np.asarray(data["X"])
    pos = X[:, :3]

    reference = np.asarray(data["reference"]) if "reference" in data else None

    if "gate_centers" not in data:
        raise KeyError(
            "Expected 'gate_centers' in the NPZ. Run the two-waypoint-per-gate "
            "solver first."
        )
    gate_centers = np.asarray(data["gate_centers"])

    waypoint_centers = (
        np.asarray(data["waypoint_centers"])
        if "waypoint_centers" in data
        else None
    )
    waypoint_steps = (
        np.asarray(data["waypoint_steps"], dtype=int)
        if "waypoint_steps" in data
        else None
    )

    fallback_normals, fallback_u, _ = _build_gate_axes_from_path(
        start_position=pos[0],
        gate_centers=gate_centers,
    )

    gate_normals = (
        np.asarray(data["gate_normals"])
        if "gate_normals" in data
        else fallback_normals
    )
    gate_axis_u = (
        np.asarray(data["gate_axis_u"])
        if "gate_axis_u" in data
        else fallback_u
    )

    gate_opening_half_widths = (
        np.asarray(data["gate_opening_half_widths"])
        if "gate_opening_half_widths" in data
        else np.tile(np.array([[0.50, 0.50]]), (len(gate_centers), 1))
    )

    gate_bar_thickness = float(
        np.asarray(data["gate_bar_thickness"])
        if "gate_bar_thickness" in data
        else 0.10
    )
    gate_depth = float(
        np.asarray(data["gate_depth"])
        if "gate_depth" in data
        else 0.10
    )
    drone_radius = float(
        np.asarray(data["drone_radius"])
        if "drone_radius" in data
        else 0.10
    )

    min_time = (
        float(np.asarray(data["min_time"]))
        if "min_time" in data
        else None
    )
    phase_times = (
        np.asarray(data["phase_times"], dtype=float)
        if "phase_times" in data
        else None
    )

    # Build physical timestamps from optimized phase times.
    if phase_times is not None and waypoint_steps is not None:
        segment_lengths = np.concatenate([
            waypoint_steps[:1],
            waypoint_steps[1:] - waypoint_steps[:-1],
        ])

        timestamps = np.zeros(len(X), dtype=float)
        start_k = 0
        elapsed = 0.0

        for phase_idx, end_k in enumerate(waypoint_steps):
            end_k = int(end_k)
            Ni = int(segment_lengths[phase_idx])
            Ti = float(phase_times[phase_idx])

            if Ni <= 0:
                continue

            local_times = np.linspace(0.0, Ti, Ni + 1)

            if phase_idx == 0:
                timestamps[start_k:end_k + 1] = elapsed + local_times
            else:
                timestamps[start_k + 1:end_k + 1] = elapsed + local_times[1:]

            elapsed += Ti
            start_k = end_k
    else:
        timestamps = np.arange(len(X), dtype=float)

    fig, ax = plt.subplots(figsize=(9, 9))

    if show_reference and reference is not None:
        ax.plot(
            reference[:, 0],
            reference[:, 1],
            "--",
            linewidth=1.3,
            alpha=0.65,
            label="Reference",
            zorder=1,
        )

    ax.plot(
        pos[:, 0],
        pos[:, 1],
        linewidth=1.2,
        alpha=0.35,
        label="Nominal path",
        zorder=2,
    )

    # Physical gates.
    for i, center in enumerate(gate_centers):
        n = gate_normals[i]
        u = gate_axis_u[i]
        hu = gate_opening_half_widths[i, 0]
        hv = gate_opening_half_widths[i, 1]

        bars = _gate_bar_geometry_xy(
            center=center,
            normal=n,
            axis_u=u,
            opening_half_width=hu,
            opening_half_height=hv,
            bar_thickness=gate_bar_thickness,
            gate_depth=gate_depth,
            inflation=0.0,
        )

        for name in ("top", "bottom", "left", "right"):
            bar_center, half_u, half_n = bars[name]
            corners = _oriented_rectangle(bar_center, u, n, half_u, half_n)
            ax.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    fill=True,
                    alpha=0.25,
                    linewidth=1.2,
                    zorder=3,
                )
            )

        if show_collision_inflation:
            inflated = _gate_bar_geometry_xy(
                center=center,
                normal=n,
                axis_u=u,
                opening_half_width=hu,
                opening_half_height=hv,
                bar_thickness=gate_bar_thickness,
                gate_depth=gate_depth,
                inflation=drone_radius,
            )

            for name in ("top", "bottom", "left", "right"):
                bar_center, half_u, half_n = inflated[name]
                corners = _oriented_rectangle(bar_center, u, n, half_u, half_n)
                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        fill=False,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=3,
                    )
                )

        ax.scatter(center[0], center[1], marker="x", s=55, zorder=5)

        arrow_len = 0.55
        ax.arrow(
            center[0] - 0.5 * arrow_len * n[0],
            center[1] - 0.5 * arrow_len * n[1],
            arrow_len * n[0],
            arrow_len * n[1],
            head_width=0.12,
            head_length=0.16,
            length_includes_head=True,
            alpha=0.8,
            zorder=5,
        )

        ax.text(
            center[0] + 0.08,
            center[1] + 0.08,
            f"G{i + 1}",
            fontsize=9,
            zorder=6,
        )

    # Pre/post transition waypoints.
    if show_transition_waypoints and waypoint_centers is not None:
        for i, wp in enumerate(waypoint_centers):
            marker = "o" if (i % 2 == 0) else "s"
            label = None
            if i == 0:
                label = "Pre-gate waypoint"
            elif i == 1:
                label = "Post-gate waypoint"

            ax.scatter(
                wp[0],
                wp[1],
                marker=marker,
                s=35,
                alpha=0.7,
                label=label,
                zorder=5,
            )

    ax.scatter(
        pos[0, 0], pos[0, 1],
        s=90, marker="o", label="Start", zorder=7,
    )
    ax.scatter(
        pos[-1, 0], pos[-1, 1],
        s=120, marker="*", label="End", zorder=7,
    )

    trail_line, = ax.plot([], [], linewidth=2.5, zorder=8, label="Nominal rollout")
    drone_dot, = ax.plot([], [], marker="o", markersize=9, linestyle="None", zorder=10)
    velocity_line, = ax.plot([], [], linewidth=2.0, zorder=9)

    time_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )

    xy_sets = [pos[:, :2], gate_centers[:, :2]]
    if reference is not None:
        xy_sets.append(reference[:, :2])
    if waypoint_centers is not None:
        xy_sets.append(waypoint_centers[:, :2])

    all_xy = np.concatenate(xy_sets, axis=0)
    margin = 1.0
    xmin = np.min(all_xy[:, 0]) - margin
    xmax = np.max(all_xy[:, 0]) + margin
    ymin = np.min(all_xy[:, 1]) - margin
    ymax = np.max(all_xy[:, 1]) + margin

    span = max(xmax - xmin, ymax - ymin)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    ax.set_xlim(xmid - 0.5 * span, xmid + 0.5 * span)
    ax.set_ylim(ymid - 0.5 * span, ymid + 0.5 * span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    if min_time is None:
        ax.set_title("Nominal Minimum-Time Trajectory — Top View")
    else:
        ax.set_title(
            "Nominal Minimum-Time Trajectory — Top View\n"
            f"Total optimized time: {min_time:.3f} s"
        )

    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = h
    ax.legend(unique.values(), unique.keys(), loc="best")

    def init():
        trail_line.set_data([], [])
        drone_dot.set_data([], [])
        velocity_line.set_data([], [])
        time_text.set_text("")
        return trail_line, drone_dot, velocity_line, time_text

    def update(k):
        if trail:
            trail_line.set_data(pos[:k + 1, 0], pos[:k + 1, 1])
        else:
            trail_line.set_data([], [])

        drone_dot.set_data([pos[k, 0]], [pos[k, 1]])

        if X.shape[1] >= 9:
            vx = float(X[k, 6])
            vy = float(X[k, 7])
            speed = np.hypot(vx, vy)

            if speed > 1e-8:
                direction = np.array([vx, vy]) / speed
                heading_len = 0.35
                velocity_line.set_data(
                    [pos[k, 0], pos[k, 0] + heading_len * direction[0]],
                    [pos[k, 1], pos[k, 1] + heading_len * direction[1]],
                )
            else:
                velocity_line.set_data([], [])

        time_text.set_text(
            f"k = {k}/{len(X) - 1}\n"
            f"t = {timestamps[k]:.3f} s"
        )

        return trail_line, drone_dot, velocity_line, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(X),
        init_func=init,
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )

    if save_path:
        if save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
        else:
            writer = FFMpegWriter(fps=fps, bitrate=4000)

        animation.save(save_path, writer=writer, dpi=180)
        print(f"Saved animation to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Animate the nominal trajectory from a top-down view."
    )
    parser.add_argument(
        "--input",
        default="multiphase_trajectory.npz",
        help="NPZ trajectory produced by the optimizer.",
    )
    parser.add_argument(
        "--output",
        default="nominal_topdown.mp4",
        help="Output .mp4 or .gif path.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-waypoints", action="store_true")
    parser.add_argument("--show-collision-inflation", action="store_true")
    parser.add_argument("--no-trail", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    animate_nominal_topdown(
        npz_path=args.input,
        save_path=args.output,
        fps=args.fps,
        trail=not args.no_trail,
        show_reference=not args.no_reference,
        show_transition_waypoints=not args.no_waypoints,
        show_collision_inflation=args.show_collision_inflation,
        show=args.show,
    )


if __name__ == "__main__":
    main()
