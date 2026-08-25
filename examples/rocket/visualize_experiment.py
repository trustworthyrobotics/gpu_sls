from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _as_np(x):
    return np.asarray(x, dtype=float)


def _save(fig, filename: str) -> None:
    path = Path(filename)
    if path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                          cth * cphi],
    ])


def _set_axes_equal_3d(ax) -> None:
    xlim = np.array(ax.get_xlim3d(), dtype=float)
    ylim = np.array(ax.get_ylim3d(), dtype=float)
    zlim = np.array(ax.get_zlim3d(), dtype=float)
    centers = np.array([xlim.mean(), ylim.mean(), zlim.mean()])
    radius = 0.5 * max(np.ptp(xlim), np.ptp(ylim), np.ptp(zlim))
    if radius <= 0.0 or not np.isfinite(radius):
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _draw_box_3d(ax, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.08) -> None:
    lo = _as_np(lo)[:3]
    hi = _as_np(hi)[:3]
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, edgecolor='none')
    ax.add_collection3d(poly)


def _draw_sphere(ax, center: np.ndarray, radius: float, alpha: float = 0.22) -> None:
    center = _as_np(center)[:3]
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0.0, alpha=alpha)


def _draw_rocket_3d(
    ax,
    state: np.ndarray,
    length: float = 0.42,
    radius: float = 0.07,
    alpha: float = 0.9,
) -> None:
    """Draw a small oriented rocket whose body +z axis is its longitudinal axis."""
    state = _as_np(state)
    if state.size < 6 or not np.all(np.isfinite(state[:6])):
        return

    pos = state[:3]
    R = _rotation_matrix(*state[3:6])

    # Body frame geometry: cylindrical body from -L/2 to +L/4 and cone to +L/2.
    n_ring = 14
    a = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
    ring_xy = np.c_[radius * np.cos(a), radius * np.sin(a)]
    z_tail = -0.50 * length
    z_shoulder = 0.22 * length
    z_tip = 0.50 * length

    tail_B = np.c_[ring_xy, np.full(n_ring, z_tail)]
    shoulder_B = np.c_[ring_xy, np.full(n_ring, z_shoulder)]
    tip_B = np.array([0.0, 0.0, z_tip])

    tail = pos + tail_B @ R.T
    shoulder = pos + shoulder_B @ R.T
    tip = pos + R @ tip_B

    faces = []
    for i in range(n_ring):
        j = (i + 1) % n_ring
        faces.append([tail[i], tail[j], shoulder[j], shoulder[i]])
        faces.append([shoulder[i], shoulder[j], tip])

    body = Poly3DCollection(faces, alpha=alpha, edgecolor='none')
    ax.add_collection3d(body)

    # Longitudinal axis makes orientation easy to read.
    axis_B = np.array([0.0, 0.0, 0.72 * length])
    axis_end = pos + R @ axis_B
    ax.plot(
        [pos[0], axis_end[0]],
        [pos[1], axis_end[1]],
        [pos[2], axis_end[2]],
        linewidth=1.3,
    )


def plot_rocket_3d(
    *,
    xs,
    plan,
    lower,
    upper,
    centers,
    radii,
    goal_center,
    goal_half_width,
    tube_stride: int = 1,
    rocket_stride: int = 5,
    filename: str = 'rocket_3d_rollouts_tube.png',
    tube_alpha: float = 0.08,
    margin: float = 0.2,
    rollout_alpha: float = 0.5,
    title: str = 'Minimum-Time Rocket',
) -> None:
    plan = _as_np(plan)
    lower = _as_np(lower)
    upper = _as_np(upper)
    xs = _as_np(xs)
    centers = _as_np(centers)
    radii = _as_np(radii).reshape(-1)
    goal_center = _as_np(goal_center)[:3]
    goal_half_width = _as_np(goal_half_width)[:3]

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(plan[:, 0], plan[:, 1], plan[:, 2], linewidth=2.2, label='Nominal plan')
    ax.scatter(plan[0, 0], plan[0, 1], plan[0, 2], s=45, marker='o', label='Start')
    ax.scatter(plan[-1, 0], plan[-1, 1], plan[-1, 2], s=55, marker='*', label='Terminal')

    if xs.ndim == 3:
        for rollout in xs:
            finite = np.all(np.isfinite(rollout[:, :3]), axis=1)
            if np.any(finite):
                rr = rollout[finite]
                ax.plot(rr[:, 0], rr[:, 1], rr[:, 2], alpha=rollout_alpha, linewidth=0.9)

    stride = max(1, int(tube_stride))
    for k in range(0, min(len(lower), len(upper)), stride):
        if np.all(np.isfinite(lower[k, :3])) and np.all(np.isfinite(upper[k, :3])):
            _draw_box_3d(ax, lower[k, :3], upper[k, :3], alpha=tube_alpha)

    for c, r in zip(centers.reshape(-1, 3), radii):
        _draw_sphere(ax, c, float(r))

    goal_lo = goal_center - goal_half_width
    goal_hi = goal_center + goal_half_width
    _draw_box_3d(ax, goal_lo, goal_hi, alpha=0.16)

    rstride = max(1, int(rocket_stride))
    for k in range(0, len(plan), rstride):
        _draw_rocket_3d(ax, plan[k])
    if (len(plan) - 1) % rstride != 0:
        _draw_rocket_3d(ax, plan[-1])

    all_xyz = [plan[:, :3], lower[:, :3], upper[:, :3], goal_lo[None, :], goal_hi[None, :]]
    if centers.size:
        all_xyz.append(centers.reshape(-1, 3))
    xyz = np.concatenate(all_xyz, axis=0)
    finite = np.all(np.isfinite(xyz), axis=1)
    if np.any(finite):
        xyz = xyz[finite]
        ax.set_xlim(xyz[:, 0].min() - margin, xyz[:, 0].max() + margin)
        ax.set_ylim(xyz[:, 1].min() - margin, xyz[:, 1].max() + margin)
        ax.set_zlim(xyz[:, 2].min() - margin, xyz[:, 2].max() + margin)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title)
    _set_axes_equal_3d(ax)
    ax.legend(loc='best')
    _save(fig, filename)


def plot_tube_graph_rocket(
    *,
    disturbed,
    tube,
    dt: float,
    filename: str = 'rocket_disturbance_vs_tube_size_pose.png',
) -> None:
    disturbed = _as_np(disturbed)
    tube = _as_np(tube)

    n_plot = min(6, tube.shape[1])
    labels = ['x [m]', 'y [m]', 'z [m]', 'roll [rad]', 'pitch [rad]', 'yaw [rad]'][:n_plot]
    t_tube = np.arange(tube.shape[0]) * float(dt)

    fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.25 * n_plot), sharex=True)
    axes = np.atleast_1d(axes)

    for j, ax in enumerate(axes):
        ax.plot(t_tube, np.abs(tube[:, j]), linewidth=2.0, label='SLS tube bound')

        if disturbed.ndim == 3 and disturbed.shape[2] > j:
            max_err = np.nanmax(np.abs(disturbed[:, :, j]), axis=0)
            if np.any(np.isfinite(max_err)):
                t_err = np.arange(max_err.shape[0]) * float(dt)
                ax.plot(t_err, max_err, linewidth=1.4, label='Max rollout error')

        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.25)
        if j == 0:
            ax.legend(loc='best')

    axes[-1].set_xlabel('time [s]')
    fig.suptitle('Rocket disturbance error versus SLS tube size')
    _save(fig, filename)


def plot_rocket_controls(
    *,
    controls,
    dt: float,
    u_min=None,
    u_max=None,
    filename: str = 'rocket_controls.png',
) -> None:
    controls = _as_np(controls)
    if controls.ndim != 2 or controls.shape[1] != 4:
        raise ValueError('Rocket controls must have shape (N, 4).')

    u_min = None if u_min is None else _as_np(u_min).reshape(-1)
    u_max = None if u_max is None else _as_np(u_max).reshape(-1)
    t = np.arange(controls.shape[0]) * float(dt)

    # Show gimbal commands in degrees for readability.
    values = controls.copy()
    values[:, 1:3] = np.rad2deg(values[:, 1:3])
    lo = None if u_min is None else u_min.copy()
    hi = None if u_max is None else u_max.copy()
    if lo is not None:
        lo[1:3] = np.rad2deg(lo[1:3])
    if hi is not None:
        hi[1:3] = np.rad2deg(hi[1:3])

    labels = [
        'thrust [N]',
        'gimbal x [deg]',
        'gimbal y [deg]',
        'roll torque [N m]',
    ]

    fig, axes = plt.subplots(4, 1, figsize=(9, 8.5), sharex=True)
    for j, ax in enumerate(axes):
        ax.step(t, values[:, j], where='post', linewidth=1.7)
        if lo is not None and lo.size > j:
            ax.axhline(lo[j], linestyle='--', linewidth=1.0, label='limits' if j == 0 else None)
        if hi is not None and hi.size > j:
            ax.axhline(hi[j], linestyle='--', linewidth=1.0)
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel('time [s]')
    if lo is not None or hi is not None:
        axes[0].legend(loc='best')
    fig.suptitle('Rocket control trajectory')
    _save(fig, filename)


def plot_rocket_top_down_tubes_goal_obstacles(
    *,
    plan,
    lower,
    upper,
    centers,
    radii,
    goal_center,
    goal_half_width,
    tube_stride: int = 1,
    filename: str = 'rocket_top_down_tubes.png',
) -> None:
    plan = _as_np(plan)
    lower = _as_np(lower)
    upper = _as_np(upper)
    centers = _as_np(centers).reshape(-1, 3)
    radii = _as_np(radii).reshape(-1)
    goal_center = _as_np(goal_center)
    goal_half_width = _as_np(goal_half_width)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(plan[:, 0], plan[:, 1], linewidth=2.2, label='Nominal plan')
    ax.scatter(plan[0, 0], plan[0, 1], s=42, label='Start')
    ax.scatter(plan[-1, 0], plan[-1, 1], s=55, marker='*', label='Terminal')

    stride = max(1, int(tube_stride))
    for k in range(0, min(len(lower), len(upper)), stride):
        w = upper[k, 0] - lower[k, 0]
        h = upper[k, 1] - lower[k, 1]
        if np.isfinite([lower[k, 0], lower[k, 1], w, h]).all():
            ax.add_patch(Rectangle((lower[k, 0], lower[k, 1]), w, h, alpha=0.06, linewidth=0.0))

    for c, r in zip(centers, radii):
        ax.add_patch(Circle((c[0], c[1]), float(r), alpha=0.22))

    goal_lo = goal_center[:2] - goal_half_width[:2]
    goal_size = 2.0 * goal_half_width[:2]
    ax.add_patch(Rectangle(goal_lo, goal_size[0], goal_size[1], alpha=0.16, label='Goal set'))

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Rocket trajectory and SLS tubes — top view')
    ax.axis('equal')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    _save(fig, filename)


def plot_rocket_side_view_tubes_goal_obstacles(
    *,
    plan,
    lower,
    upper,
    centers,
    radii,
    goal_center,
    goal_half_width,
    tube_stride: int = 1,
    ground_height: float = 0.0,
    filename: str = 'rocket_side_view_tubes.png',
    title: str = 'Rocket trajectory and SLS tubes — side view',
    terminal_visible_fraction: float = 0.35,
) -> None:
    plan = _as_np(plan)
    lower = _as_np(lower)
    upper = _as_np(upper)
    centers = _as_np(centers).reshape(-1, 3)
    radii = _as_np(radii).reshape(-1)
    goal_center = _as_np(goal_center)
    goal_half_width = _as_np(goal_half_width)

    fig, ax = plt.subplots(figsize=(9, 6.2))

    # ---------------------------------------------------------
    # Nominal trajectory
    # ---------------------------------------------------------
    ax.plot(
        plan[:, 0],
        plan[:, 2],
        linewidth=2.2,
        label='Nominal plan',
    )

    ax.scatter(
        plan[0, 0],
        plan[0, 2],
        s=42,
        label='Start',
    )

    ax.scatter(
        plan[-1, 0],
        plan[-1, 2],
        s=55,
        marker='*',
        label='Terminal',
    )

    # ---------------------------------------------------------
    # SLS tubes
    # ---------------------------------------------------------
    stride = max(1, int(tube_stride))

    for k in range(0, min(len(lower), len(upper)), stride):
        w = upper[k, 0] - lower[k, 0]
        h = upper[k, 2] - lower[k, 2]

        if np.isfinite(
            [lower[k, 0], lower[k, 2], w, h]
        ).all():
            ax.add_patch(
                Rectangle(
                    (lower[k, 0], lower[k, 2]),
                    w,
                    h,
                    alpha=0.06,
                    linewidth=0.0,
                )
            )

    # ---------------------------------------------------------
    # Obstacles
    # ---------------------------------------------------------
    for c, r in zip(centers, radii):
        ax.add_patch(
            Circle(
                (c[0], c[2]),
                float(r),
                alpha=0.22,
            )
        )

    # ---------------------------------------------------------
    # Terminal set
    # ---------------------------------------------------------
    goal_lo = np.array([
        goal_center[0] - goal_half_width[0],
        goal_center[2] - goal_half_width[2],
    ])

    goal_size = np.array([
        2.0 * goal_half_width[0],
        2.0 * goal_half_width[2],
    ])

    ax.add_patch(
        Rectangle(
            goal_lo,
            goal_size[0],
            goal_size[1],
            alpha=0.16,
            label='Goal set',
        )
    )

    # ---------------------------------------------------------
    # Ground
    # ---------------------------------------------------------
    ax.axhline(
        float(ground_height),
        linewidth=1.2,
        linestyle='--',
        label='Ground',
    )

    # ---------------------------------------------------------
    # Crop plot around trajectory.
    #
    # Only show part of the terminal set on the right-hand side.
    # Matplotlib automatically clips the rectangle at the axis.
    # ---------------------------------------------------------
    x_data_min = min(
        np.min(lower[:, 0]),
        np.min(plan[:, 0]),
    )

    x_data_max = max(
        np.max(upper[:, 0]),
        np.max(plan[:, 0]),
    )

    z_data_min = min(
        np.min(lower[:, 2]),
        np.min(plan[:, 2]),
        ground_height,
    )

    z_data_max = max(
        np.max(upper[:, 2]),
        np.max(plan[:, 2]),
    )

    x_span = max(x_data_max - x_data_min, 1.0)
    z_span = max(z_data_max - z_data_min, 1.0)

    x_margin = 0.05 * x_span
    z_margin = 0.08 * z_span

    # Near/left edge of terminal set
    goal_x_min = goal_center[0] - goal_half_width[0]

    # Only expose this fraction of the terminal set.
    goal_visible_x = (
        goal_x_min
        + terminal_visible_fraction * 2.0 * goal_half_width[0]
    )

    x_right = max(
        x_data_max + x_margin,
        goal_visible_x,
    )

    ax.set_xlim(
        x_data_min - x_margin,
        x_right,
    )

    ax.set_ylim(
        z_data_min - z_margin,
        z_data_max + z_margin,
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title(title)

    # Do NOT use ax.axis('equal') here, because it can expand
    # the limits and reveal the entire terminal set again.
    ax.set_aspect('equal', adjustable='box')

    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    _save(fig, filename)