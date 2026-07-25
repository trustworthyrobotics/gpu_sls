import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle


# --- Styling palette (muted, readable) ---
PALETTE = {
    "plan":      "#1f77b4",   # blue
    "random":    "#ff7f0e",   # orange
    "adversary": "#d62728",   # red
    "tube_face": "#2ca02c",   # green
    "tube_edge": "#1b7f1b",   # darker green edge
    "obs_face":  "#7f7f7f",   # gray
    "obs_edge":  "#4d4d4d",   # dark gray edge
    "goal_face": "#9467bd",   # purple
    "goal_edge": "#5e3c99",   # darker purple edge
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
    ],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


PROJECTION_AXES = {
    "xy": (0, 1, "x", "y"),
    "xz": (0, 2, "x", "z"),
    "yz": (1, 2, "y", "z"),
}


def _normalize_plan_array(arr, name: str):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{name} has shape {arr.shape}. Expected (n_steps, N+1, 2).")
    return arr


def _box_faces(lower, upper):
    """Return the six faces of an axis-aligned 3D box."""
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    return [
        vertices[[0, 1, 2, 3]], vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]], vertices[[2, 3, 7, 6]],
        vertices[[1, 2, 6, 5]], vertices[[3, 0, 4, 7]],
    ]


def _set_axes_equal_3d(ax, points, margin):
    """Set equal physical scale on all three axes."""
    points = np.asarray(points)
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        finite = np.array([[0.0, 0.0, 0.0]])
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)) + margin, margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_quadrotor_3d(
    xs,
    plan,
    lower=None,
    upper=None,
    centers=None,
    radii=None,
    goal_center=None,
    goal_half_width=None,
    tube_stride: int = 1,
    tube_alpha: float = 0.08,
    rollout_alpha: float = 0.45,
    obstacle_height: tuple[float, float] | None = None,  # deprecated; ignored for spheres
    margin: float = 0.2,
    filename: str | None = "quadrotor_3d_rollouts_tube.png",
    dpi: int = 300,
    title: str = "Minimum-Time Quadrotor: 3D Rollouts + Robust Tube",
):
    """Plot quadrotor trajectories, robust boxes, spherical obstacles, and goal in 3D."""
    xs = np.asarray(xs)
    plan = np.asarray(plan)
    if xs.ndim == 2:
        xs = xs[None, ...]
    if xs.ndim != 3 or xs.shape[-1] < 3:
        raise ValueError(f"xs has shape {xs.shape}. Expected (n_rollouts, T, n>=3).")
    if plan.ndim != 2 or plan.shape[-1] < 3:
        raise ValueError(f"plan has shape {plan.shape}. Expected (T, n>=3).")

    lower = None if lower is None else np.asarray(lower)
    upper = None if upper is None else np.asarray(upper)
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be provided together.")
    if lower is not None and (lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] < 3):
        raise ValueError("lower and upper must have matching shape (T, n>=3).")

    centers = None if centers is None else np.atleast_2d(np.asarray(centers))
    radii = None if radii is None else np.asarray(radii).reshape(-1)
    if centers is not None and (
        centers.shape[1] != 3
        or radii is None
        or len(radii) != len(centers)
    ):
        raise ValueError("centers and radii must have shapes (K, 3) and (K,).")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    if lower is not None:
        for k in range(0, len(lower), max(1, int(tube_stride))):
            lo, up = lower[k, :3], upper[k, :3]
            if not np.all(np.isfinite([lo, up])) or np.any(up < lo):
                continue
            ax.add_collection3d(Poly3DCollection(
                _box_faces(lo, up),
                facecolor=PALETTE["tube_face"],
                edgecolor=PALETTE["tube_edge"],
                linewidth=0.35,
                alpha=tube_alpha,
            ))
        ax.plot([], [], [], color=PALETTE["tube_edge"], alpha=0.5, label="Robust tube")

    if goal_center is not None and goal_half_width is not None:
        center = np.asarray(goal_center).reshape(3)
        half_width = np.asarray(goal_half_width).reshape(3)
        ax.add_collection3d(Poly3DCollection(
            _box_faces(center - half_width, center + half_width),
            facecolor=PALETTE["goal_face"],
            edgecolor=PALETTE["goal_edge"],
            linewidth=1.2,
            alpha=0.3,
        ))
        ax.plot([], [], [], color=PALETTE["goal_edge"], linewidth=3, label="Goal terminal set")

    extent_points = [plan[:, :3]]
    finite_positions = xs[..., :3].reshape(-1, 3)
    extent_points.append(finite_positions)
    if lower is not None:
        extent_points.extend([lower[:, :3], upper[:, :3]])

    if centers is not None:
        # Spherical coordinates:
        #   x = cx + r sin(phi) cos(theta)
        #   y = cy + r sin(phi) sin(theta)
        #   z = cz + r cos(phi)
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        phi = np.linspace(0.0, np.pi, 32)
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        for index, (center, radius) in enumerate(zip(centers, radii)):
            cx, cy, cz = center
            radius = float(radius)

            sphere_x = cx + radius * np.sin(phi_grid) * np.cos(theta_grid)
            sphere_y = cy + radius * np.sin(phi_grid) * np.sin(theta_grid)
            sphere_z = cz + radius * np.cos(phi_grid)

            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=PALETTE["obs_face"],
                edgecolor=PALETTE["obs_edge"],
                linewidth=0.15,
                alpha=0.32,
                shade=True,
                antialiased=True,
            )

            # Include the full sphere bounds when computing equal axis limits.
            extent_points.append(np.array([
                [cx - radius, cy, cz],
                [cx + radius, cy, cz],
                [cx, cy - radius, cz],
                [cx, cy + radius, cz],
                [cx, cy, cz - radius],
                [cx, cy, cz + radius],
            ]))

            if index == 0:
                ax.plot(
                    [], [], [],
                    color=PALETTE["obs_edge"],
                    linewidth=4,
                    alpha=0.5,
                    label="Spherical obstacle",
                )

    ax.plot(
        plan[:, 0], plan[:, 1], plan[:, 2],
        linestyle="--", linewidth=2.5, color=PALETTE["plan"], label="Planned trajectory",
    )
    valid_rollouts = 0
    for rollout in xs:
        valid = np.all(np.isfinite(rollout[:, :3]), axis=1)
        if np.any(valid):
            ax.plot(
                rollout[valid, 0], rollout[valid, 1], rollout[valid, 2],
                color=PALETTE["random"], alpha=rollout_alpha, linewidth=1.2,
            )
            valid_rollouts += 1
    if valid_rollouts:
        ax.plot([], [], [], color=PALETTE["random"], alpha=rollout_alpha,
                label=f"Rollouts (n={valid_rollouts})")

    ax.scatter(*plan[0, :3], color=PALETTE["plan"], marker="o", s=45, label="Start")
    _set_axes_equal_3d(ax, np.concatenate(extent_points), margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_rollouts_tubes_centers(
    xs,
    centers=None,
    radii=None,
    plans_xy=None,
    lowers_xy=None,
    uppers_xy=None,
    goal_center=None,
    goal_half_width=None,
    step_idx: int | None = 0,
    tube_stride: int = 2,
    tube_alpha: float = 0.15,
    rollout_alpha: float = 0.35,
    show_plan: bool = True,
    margin: float = 0.5,
    filename: str | None = "quadrotor_3d_rollouts_projection.png",
    dpi: int = 300,
    projection: str = "xy",
    x_label: str | None = None,
    y_label: str | None = None,
    title: str = "3D Quadrotor: Rollouts + Robust Tube + Obstacle Centers",
):
    """
    Static projected plot with obstacle centers, tube rectangles, and rollout trajectories.

    Expected shapes:
      xs:        (n_rollouts, T, n) OR (T, n), with n >= 3 for 3D quadrotor
      plans_xy:  (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      lowers_xy: (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      uppers_xy: (n_steps, N+1, 2) OR (N+1, 2)  (optional)
      centers:   (K, 2)                         (optional)
      radii:     (K,)                           (optional)
      goal_center:     (3,)                      (optional)
      goal_half_width: (3,)                      (optional)

    Notes:
      - For the 3D quadrotor state:
          0: px, 1: py, 2: pz, 3: phi, 4: theta, 5: psi,
          6: vx, 7: vy, 8: vz, 9: p, 10: q, 11: r
      - xs is projected using `projection` unless you already pass preprojected plans/tubes.
      - plans_xy / lowers_xy / uppers_xy are assumed already projected into 2D.
    """
    xs = np.asarray(xs)

    if projection not in PROJECTION_AXES:
        raise ValueError(f"projection must be one of {list(PROJECTION_AXES.keys())}, got {projection!r}")

    ax_i, ax_j, default_x_label, default_y_label = PROJECTION_AXES[projection]
    if x_label is None:
        x_label = default_x_label
    if y_label is None:
        y_label = default_y_label

    # Normalize xs to (n_rollouts, T, n)
    if xs.ndim == 2:
        if xs.shape[1] < max(ax_i, ax_j) + 1:
            raise ValueError(f"xs has shape {xs.shape}. Expected last dim large enough for projection {projection}.")
        xs = xs[None, :, :]
    elif xs.ndim == 3:
        if xs.shape[2] < max(ax_i, ax_j) + 1:
            raise ValueError(f"xs has shape {xs.shape}. Expected xs[..., :] large enough for projection {projection}.")
    else:
        raise ValueError(f"xs has shape {xs.shape}. Expected 2D or 3D array.")

    n_rollouts, T, nx = xs.shape

    plans_xy = _normalize_plan_array(plans_xy, "plans_xy")
    lowers_xy = _normalize_plan_array(lowers_xy, "lowers_xy")
    uppers_xy = _normalize_plan_array(uppers_xy, "uppers_xy")

    if centers is not None:
        centers = np.asarray(centers)
        if centers.ndim == 1:
            centers = centers[None, :]
        if centers.shape[-1] != 2:
            raise ValueError(f"centers has shape {centers.shape}. Expected (K, 2).")

    if radii is not None:
        radii = np.asarray(radii).reshape(-1)

    if goal_center is not None:
        goal_center = np.asarray(goal_center).reshape(3)
    if goal_half_width is not None:
        goal_half_width = np.asarray(goal_half_width).reshape(3)
        if np.any(goal_half_width < 0.0):
            raise ValueError("goal_half_width must be nonnegative.")
    if (goal_center is None) != (goal_half_width is None):
        raise ValueError("goal_center and goal_half_width must be provided together.")

    # pick tube/plan frame
    if lowers_xy is not None and uppers_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, lowers_xy.shape[0] - 1))
        lo = lowers_xy[step_idx]
        up = uppers_xy[step_idx]
    else:
        lo = up = None

    # axis limits (nan-aware)
    all_x = [xs[:, :, ax_i].ravel()]
    all_y = [xs[:, :, ax_j].ravel()]

    if plans_xy is not None:
        all_x.append(plans_xy[:, :, 0].ravel())
        all_y.append(plans_xy[:, :, 1].ravel())
    if lo is not None and up is not None:
        all_x.append(lo[:, 0].ravel())
        all_x.append(up[:, 0].ravel())
        all_y.append(lo[:, 1].ravel())
        all_y.append(up[:, 1].ravel())
    if centers is not None and centers.size:
        all_x.append(centers[:, 0].ravel())
        all_y.append(centers[:, 1].ravel())
    if goal_center is not None:
        all_x.extend([
            np.array([goal_center[ax_i] - goal_half_width[ax_i]]),
            np.array([goal_center[ax_i] + goal_half_width[ax_i]]),
        ])
        all_y.extend([
            np.array([goal_center[ax_j] - goal_half_width[ax_j]]),
            np.array([goal_center[ax_j] + goal_half_width[ax_j]]),
        ])

    all_x = np.concatenate(all_x) if len(all_x) else np.array([0.0])
    all_y = np.concatenate(all_y) if len(all_y) else np.array([0.0])

    xmin, xmax = float(np.nanmin(all_x) - margin), float(np.nanmax(all_x) + margin)
    ymin, ymax = float(np.nanmin(all_y) - margin), float(np.nanmax(all_y) + margin)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # obstacles
    if centers is not None and centers.size and radii is not None and radii.size == centers.shape[0]:
        for c, r in zip(centers, radii):
            circ = plt.Circle(
                (float(c[0]), float(c[1])),
                float(r),
                alpha=0.5,
                color="tab:red",
            )
            ax.add_patch(circ)

    # projected terminal goal set: |position - goal_center| <= goal_half_width
    if goal_center is not None:
        goal_lower = (
            goal_center[ax_i] - goal_half_width[ax_i],
            goal_center[ax_j] - goal_half_width[ax_j],
        )
        ax.add_patch(Rectangle(
            goal_lower,
            2.0 * goal_half_width[ax_i],
            2.0 * goal_half_width[ax_j],
            facecolor=PALETTE["goal_face"],
            edgecolor=PALETTE["goal_edge"],
            linewidth=2.0,
            alpha=0.25,
            label="Goal terminal set",
            zorder=2,
        ))

    # tubes
    if lo is not None and up is not None:
        for k in range(0, lo.shape[0], max(1, int(tube_stride))):
            w = up[k, 0] - lo[k, 0]
            h = up[k, 1] - lo[k, 1]
            if not np.isfinite(w) or not np.isfinite(h) or w < 0.0 or h < 0.0:
                continue
            rect = Rectangle(
                (lo[k, 0], lo[k, 1]),
                w,
                h,
                facecolor=PALETTE["tube_face"],
                edgecolor=PALETTE["tube_edge"],
                alpha=tube_alpha,
            )
            ax.add_patch(rect)
        ax.plot([], [], color=PALETTE["tube_face"], alpha=tube_alpha, label=f"Tube boxes (step {step_idx})")

    # plan
    if show_plan and plans_xy is not None:
        step_idx = int(step_idx if step_idx is not None else 0)
        step_idx = max(0, min(step_idx, plans_xy.shape[0] - 1))
        ax.plot(
            plans_xy[step_idx, :, 0],
            plans_xy[step_idx, :, 1],
            linestyle="--",
            linewidth=2,
            color=PALETTE["plan"],
            label="Planned (open-loop)",
        )

    # rollouts
    for i in range(n_rollouts):
        ax.plot(
            xs[i, :, ax_i],
            xs[i, :, ax_j],
            alpha=rollout_alpha,
            color=PALETTE["random"],
        )
    ax.plot([], [], alpha=rollout_alpha, color=PALETTE["random"], label=f"Rollouts (n={n_rollouts})")

    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tube_graph_quadrotor(
    disturbed,
    tube,
    dt,
    filename: str = "disturbance_vs_tube_size_quadrotor_3d.png",
    state_labels: list | None = None,
):
    """
    Plot deviation vs tube size for quadrotor states.

    disturbed: (n_rollouts, T, n_states)
    tube:      (T+1, n_states)
    """
    disturbed = np.asarray(disturbed)
    tube = np.asarray(tube)

    n_states = disturbed.shape[2]

    if disturbed.ndim != 3:
        raise ValueError(
            f"disturbed has shape {disturbed.shape}. Expected (n_rollouts, T, n_states)."
        )
    if tube.ndim != 2 or tube.shape[1] != n_states:
        raise ValueError(
            f"tube has shape {tube.shape}. Expected (T+1, {n_states})."
        )

    default_labels_6 = [
        ("px", "meters"), ("pz", "meters"), ("theta", "radians"),
        ("vx", "m/s"),    ("vz", "m/s"),    ("omega", "rad/s"),
    ]
    default_labels_12 = [
        ("px", "meters"), ("py", "meters"), ("pz", "meters"),
        ("phi", "radians"), ("theta", "radians"), ("psi", "radians"),
        ("vx", "m/s"), ("vy", "m/s"), ("vz", "m/s"),
        ("p", "rad/s"), ("q", "rad/s"), ("r", "rad/s"),
    ]

    if state_labels is None:
        if n_states == 6:
            state_labels = default_labels_6
        elif n_states == 12:
            state_labels = default_labels_12
        else:
            state_labels = [(f"x{i}", "units") for i in range(n_states)]

    T = disturbed.shape[1]
    tube_trim = tube[1:, :]
    if tube_trim.shape[0] != T:
        raise ValueError(
            f"tube[1:] has length {tube_trim.shape[0]}, but disturbed time dimension is {T}."
        )

    t = np.arange(T) * dt

    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2 * n_states + 2), sharex=True)
    if n_states == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        tube_i = tube_trim[:, idx]
        dev_all = disturbed[:, :, idx]

        ax.plot(t, tube_i, label=f"tube size ({state_labels[idx][0]})", linewidth=3)

        for r_idx, dev in enumerate(dev_all):
            m = np.isfinite(dev)
            ax.plot(
                t[m],
                dev[m],
                label=f"|{state_labels[idx][0]} - nominal|" if r_idx == 0 else None,
                alpha=0.8,
            )

        ax.set_ylabel(state_labels[idx][1])
        ax.set_title(f"{state_labels[idx][0]}: Deviation vs Tube Size")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)