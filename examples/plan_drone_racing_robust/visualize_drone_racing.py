import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations


def visualize_trajectory_and_gates(
    X,
    waypoint_centers,
    waypoint_half_widths,
    waypoint_steps=None,
    reference=None,
    title="Quadrotor Trajectory Through Gates",
    save_path=None,
    show=False,
):
    """
    Visualize a 3D quadrotor trajectory and axis-aligned waypoint/gate boxes.

    Parameters
    ----------
    X : array-like, shape (N+1, nx)
        State trajectory. Position must be X[:, :3].

    waypoint_centers : array-like, shape (num_gates, 3)
        Center [x, y, z] of each gate.

    waypoint_half_widths : array-like, shape (num_gates, 3)
        Half-width [hx, hy, hz] of each gate/waypoint constraint box.

    waypoint_steps : array-like, optional
        Horizon index associated with each gate.

    reference : array-like, optional, shape (N+1, nx)
        Reference trajectory to show as a dashed line.

    title : str
        Plot title.

    save_path : str, optional
        If provided, save the figure here.

    show : bool
        Whether to call plt.show().
    """

    X = np.asarray(X)
    centers = np.asarray(waypoint_centers)
    half_widths = np.asarray(waypoint_half_widths)

    if waypoint_steps is not None:
        waypoint_steps = np.asarray(waypoint_steps, dtype=int)

    if reference is not None:
        reference = np.asarray(reference)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # --------------------------------------------------
    # Planned / executed trajectory
    # --------------------------------------------------
    pos = X[:, :3]

    ax.plot(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        linewidth=2.5,
        label="Trajectory",
    )

    # Start
    ax.scatter(
        pos[0, 0],
        pos[0, 1],
        pos[0, 2],
        s=100,
        marker="o",
        label="Start",
    )

    # End
    ax.scatter(
        pos[-1, 0],
        pos[-1, 1],
        pos[-1, 2],
        s=100,
        marker="*",
        label="End",
    )

    # --------------------------------------------------
    # Optional reference trajectory
    # --------------------------------------------------
    if reference is not None:
        ref_pos = reference[:, :3]

        ax.plot(
            ref_pos[:, 0],
            ref_pos[:, 1],
            ref_pos[:, 2],
            "--",
            linewidth=1.5,
            alpha=0.7,
            label="Reference",
        )

    # --------------------------------------------------
    # Gates / waypoint boxes
    # --------------------------------------------------
    for i, (center, half_width) in enumerate(zip(centers, half_widths)):
        _draw_box(
            ax,
            center=center,
            half_width=half_width,
            alpha=0.35,
        )

        # Gate center
        ax.scatter(
            center[0],
            center[1],
            center[2],
            s=60,
            marker="x",
        )

        # Label
        ax.text(
            center[0],
            center[1],
            center[2],
            f"  Gate {i + 1}",
            fontsize=10,
        )

        # Mark the optimized trajectory position at that gate step
        if waypoint_steps is not None:
            k = waypoint_steps[i]

            if 0 <= k < len(X):
                gate_pos = pos[k]

                ax.scatter(
                    gate_pos[0],
                    gate_pos[1],
                    gate_pos[2],
                    s=80,
                    marker="o",
                )

                # Line from desired gate center to actual position
                ax.plot(
                    [center[0], gate_pos[0]],
                    [center[1], gate_pos[1]],
                    [center[2], gate_pos[2]],
                    ":",
                    linewidth=1.0,
                    alpha=0.8,
                )

    # --------------------------------------------------
    # Axis configuration
    # --------------------------------------------------
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)

    ax.grid(True)
    ax.legend()

    _set_axes_equal(ax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def _draw_box(ax, center, half_width, alpha=0.3):
    """
    Draw an axis-aligned 3D box corresponding to:

        |x - center| <= half_width
    """
    center = np.asarray(center)
    half_width = np.asarray(half_width)

    lower = center - half_width
    upper = center + half_width

    # 8 box corners
    corners = np.array(
        list(
            product(
                [lower[0], upper[0]],
                [lower[1], upper[1]],
                [lower[2], upper[2]],
            )
        )
    )

    # Draw edges. Two corners share an edge if exactly
    # two coordinates are equal.
    for p1, p2 in combinations(corners, 2):
        if np.sum(np.isclose(p1, p2)) == 2:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                linewidth=1.5,
                alpha=alpha,
            )


def _set_axes_equal(ax):
    """
    Make x/y/z use approximately equal spatial scale.
    """
    x_limits = np.array(ax.get_xlim3d())
    y_limits = np.array(ax.get_ylim3d())
    z_limits = np.array(ax.get_zlim3d())

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    radius = 0.5 * max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    ax.set_xlim3d(x_mid - radius, x_mid + radius)
    ax.set_ylim3d(y_mid - radius, y_mid + radius)
    ax.set_zlim3d(z_mid - radius, z_mid + radius)