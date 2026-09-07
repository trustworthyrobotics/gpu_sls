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
    gate_normals=None,
    gate_axis_u=None,
    gate_axis_v=None,
    gate_opening_half_widths=None,
    gate_bar_thickness=0.10,
    gate_depth=0.10,
    drone_radius=0.10,
    show_collision_boxes=True,
    show_waypoint_boxes=False,
):
    """
    Visualize a 3D quadrotor trajectory and the same four-bar gates used by
    the optimizer.

    Each physical gate consists of four oriented hyperboxes:
        top, bottom, left, right.

    If ``show_collision_boxes=True``, the plot also draws the obstacle boxes
    after inflation by ``drone_radius``. These inflated boxes correspond to
    the geometry used by ``make_gate_frame_constraints()``.

    Gate orientation
    ----------------
    If gate_normals / gate_axis_u / gate_axis_v are not supplied, they are
    reconstructed exactly as in the optimizer:

        normal_i = XY direction from previous gate/start -> current gate
        v_i      = world +z
        u_i      = v_i x normal_i

    Parameters
    ----------
    X : array-like, shape (N+1, nx)
        State trajectory. Position must be X[:, :3].

    waypoint_centers : array-like, shape (num_gates, 3)
        Center [x, y, z] of each gate.

    waypoint_half_widths : array-like, shape (num_gates, 3)
        Original waypoint constraint half-widths. These are only drawn when
        ``show_waypoint_boxes=True``.

    waypoint_steps : array-like, optional
        Horizon index associated with each gate.

    reference : array-like, optional, shape (N+1, nx)
        Reference trajectory to show as a dashed line.

    gate_normals : array-like, optional, shape (num_gates, 3)
        Desired traversal direction through each gate.

    gate_axis_u : array-like, optional, shape (num_gates, 3)
        Horizontal in-plane gate axis.

    gate_axis_v : array-like, optional, shape (num_gates, 3)
        Vertical in-plane gate axis.

    gate_opening_half_widths : array-like, optional
        Shape (num_gates, 2), where:
            [:, 0] = horizontal half-width along gate_axis_u
            [:, 1] = vertical half-height along gate_axis_v

        Defaults to a 1.0 m x 1.0 m opening:
            [0.50, 0.50] for every gate.

    gate_bar_thickness : float
        Physical bar thickness in the gate plane.

    gate_depth : float
        Physical gate depth along the gate normal.

    drone_radius : float
        Radius used to inflate every physical gate bar for collision
        constraints.

    show_collision_boxes : bool
        Draw the inflated obstacle hyperboxes used by the optimizer.

    show_waypoint_boxes : bool
        Also draw the original waypoint constraint boxes.


    title : str
        Plot title.

    save_path : str, optional
        If provided, save the figure here.

    show : bool
        Whether to call plt.show().
    """

    X = np.asarray(X)
    centers = np.asarray(waypoint_centers, dtype=float)
    half_widths = np.asarray(waypoint_half_widths, dtype=float)

    if X.ndim != 2 or X.shape[1] < 3:
        raise ValueError(f"X must have shape (N+1, nx>=3); got {X.shape}")

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(
            "waypoint_centers must have shape (num_gates, 3); "
            f"got {centers.shape}"
        )

    num_gates = centers.shape[0]

    if half_widths.shape != (num_gates, 3):
        raise ValueError(
            "waypoint_half_widths must have shape "
            f"({num_gates}, 3); got {half_widths.shape}"
        )

    if waypoint_steps is not None:
        waypoint_steps = np.asarray(waypoint_steps, dtype=int)

    if reference is not None:
        reference = np.asarray(reference)

    opening_half_widths = _normalize_gate_openings(
        gate_opening_half_widths,
        num_gates,
    )

    # --------------------------------------------------
    # Gate frames
    # --------------------------------------------------
    if gate_normals is None:
        normals, axis_u, axis_v = _build_gate_axes_from_path(
            start_position=X[0, :3],
            gate_centers=centers,
        )
    else:
        normals = _normalize_rows(np.asarray(gate_normals, dtype=float))

        if gate_axis_v is None:
            axis_v = np.broadcast_to(
                np.array([0.0, 0.0, 1.0]),
                normals.shape,
            ).copy()
        else:
            axis_v = _normalize_rows(
                np.asarray(gate_axis_v, dtype=float)
            )

        if gate_axis_u is None:
            axis_u = np.cross(axis_v, normals)
            axis_u = _normalize_rows(axis_u)
        else:
            axis_u = _normalize_rows(
                np.asarray(gate_axis_u, dtype=float)
            )

    if normals.shape != (num_gates, 3):
        raise ValueError(
            f"gate_normals must have shape ({num_gates}, 3); "
            f"got {normals.shape}"
        )

    if axis_u.shape != (num_gates, 3):
        raise ValueError(
            f"gate_axis_u must have shape ({num_gates}, 3); "
            f"got {axis_u.shape}"
        )

    if axis_v.shape != (num_gates, 3):
        raise ValueError(
            f"gate_axis_v must have shape ({num_gates}, 3); "
            f"got {axis_v.shape}"
        )

    fig = plt.figure(figsize=(11, 9))
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
        zorder=8,
    )

    # Start
    ax.scatter(
        pos[0, 0],
        pos[0, 1],
        pos[0, 2],
        s=100,
        marker="o",
        label="Start",
        zorder=9,
    )

    # End
    ax.scatter(
        pos[-1, 0],
        pos[-1, 1],
        pos[-1, 2],
        s=130,
        marker="*",
        label="End",
        zorder=9,
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
            zorder=5,
        )

    # --------------------------------------------------
    # Gates
    # --------------------------------------------------
    for i in range(num_gates):
        center = centers[i]
        n = normals[i]
        u_axis = axis_u[i]
        v_axis = axis_v[i]

        hu = opening_half_widths[i, 0]
        hv = opening_half_widths[i, 1]

        physical_bars = _gate_bar_geometry(
            gate_center=center,
            gate_normal=n,
            gate_axis_u=u_axis,
            gate_axis_v=v_axis,
            opening_half_width=hu,
            opening_half_height=hv,
            bar_thickness=gate_bar_thickness,
            gate_depth=gate_depth,
            inflation=0.0,
        )

        inflated_bars = _gate_bar_geometry(
            gate_center=center,
            gate_normal=n,
            gate_axis_u=u_axis,
            gate_axis_v=v_axis,
            opening_half_width=hu,
            opening_half_height=hv,
            bar_thickness=gate_bar_thickness,
            gate_depth=gate_depth,
            inflation=drone_radius,
        )

        # ----------------------------------------------
        # Physical four-bar frame
        # ----------------------------------------------
        for bar_name in ("top", "bottom", "left", "right"):
            bar_center, bar_half_sizes = physical_bars[bar_name]

            _draw_oriented_box(
                ax,
                center=bar_center,
                half_sizes=bar_half_sizes,
                axis_u=u_axis,
                axis_v=v_axis,
                axis_n=n,
                linewidth=2.0,
                alpha=0.95,
                linestyle="-",
            )

        # ----------------------------------------------
        # Inflated collision boxes
        #
        # These are the actual outside-hyperbox regions
        # used by make_gate_frame_constraints().
        # ----------------------------------------------
        if show_collision_boxes:
            for bar_name in ("top", "bottom", "left", "right"):
                bar_center, bar_half_sizes = inflated_bars[bar_name]

                _draw_oriented_box(
                    ax,
                    center=bar_center,
                    half_sizes=bar_half_sizes,
                    axis_u=u_axis,
                    axis_v=v_axis,
                    axis_n=n,
                    linewidth=1.0,
                    alpha=0.40,
                    linestyle="--",
                )

        # ----------------------------------------------
        # Opening outline
        # ----------------------------------------------
        _draw_gate_opening(
            ax,
            center=center,
            axis_u=u_axis,
            axis_v=v_axis,
            half_width=hu,
            half_height=hv,
            linewidth=1.2,
            alpha=0.8,
        )

        # Gate center
        ax.scatter(
            center[0],
            center[1],
            center[2],
            s=55,
            marker="x",
            zorder=10,
        )

        # Label
        label_position = center + 0.12 * v_axis
        ax.text(
            label_position[0],
            label_position[1],
            label_position[2],
            f"  Gate {i + 1}",
            fontsize=10,
        )

        # ----------------------------------------------
        # Optional original waypoint constraint box
        # ----------------------------------------------
        if show_waypoint_boxes:
            _draw_box(
                ax,
                center=center,
                half_width=half_widths[i],
                alpha=0.30,
                linestyle=":",
            )

        # ----------------------------------------------
        # Actual trajectory state at gate timestep
        # ----------------------------------------------
        if waypoint_steps is not None:
            k = waypoint_steps[i]

            if 0 <= k < len(X):
                gate_pos = pos[k]

                ax.scatter(
                    gate_pos[0],
                    gate_pos[1],
                    gate_pos[2],
                    s=75,
                    marker="o",
                    zorder=10,
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


def _build_gate_axes_from_path(start_position, gate_centers):
    """
    NumPy equivalent of build_gate_axes_from_path() in the optimizer.

    For gate i:
        normal = XY direction from previous gate/start to current gate
        v      = world +z
        u      = v x normal
    """
    start_position = np.asarray(start_position, dtype=float)
    gate_centers = np.asarray(gate_centers, dtype=float)

    previous = np.concatenate(
        [
            start_position[None, :],
            gate_centers[:-1],
        ],
        axis=0,
    )

    incoming = gate_centers - previous

    # Optimizer keeps the gates vertical.
    normals = incoming.copy()
    normals[:, 2] = 0.0
    normals = _normalize_rows(normals)

    axis_v = np.broadcast_to(
        np.array([0.0, 0.0, 1.0]),
        normals.shape,
    ).copy()

    # Same convention as optimizer:
    # u x v = n  =>  u = v x n
    axis_u = np.cross(axis_v, normals)
    axis_u = _normalize_rows(axis_u)

    return normals, axis_u, axis_v


def _gate_bar_geometry(
    gate_center,
    gate_normal,
    gate_axis_u,
    gate_axis_v,
    opening_half_width,
    opening_half_height,
    bar_thickness,
    gate_depth,
    inflation=0.0,
):
    """
    Reproduce the exact four-hyperbox geometry used by the optimizer.

    Returns
    -------
    dict:
        {
            "top":    (center_world, half_sizes_local),
            "bottom": (...),
            "left":   (...),
            "right":  (...),
        }

    half_sizes_local is ordered [u, v, n].
    """
    c = np.asarray(gate_center, dtype=float)
    n = np.asarray(gate_normal, dtype=float)
    u = np.asarray(gate_axis_u, dtype=float)
    v = np.asarray(gate_axis_v, dtype=float)

    hu = float(opening_half_width)
    hv = float(opening_half_height)
    t = float(bar_thickness)
    half_depth = 0.5 * float(gate_depth)
    inflation = float(inflation)

    # Same centers as make_gate_frame_constraints().
    top_center = c + (hv + 0.5 * t) * v
    bottom_center = c - (hv + 0.5 * t) * v
    left_center = c - (hu + 0.5 * t) * u
    right_center = c + (hu + 0.5 * t) * u

    # Same physical half-sizes as optimizer, in [u, v, n].
    top_bottom_half_sizes = np.array(
        [
            hu + t,
            0.5 * t,
            half_depth,
        ],
        dtype=float,
    )

    left_right_half_sizes = np.array(
        [
            0.5 * t,
            hv,
            half_depth,
        ],
        dtype=float,
    )

    # Optimizer inflates every local half-size by drone_radius.
    top_bottom_half_sizes += inflation
    left_right_half_sizes += inflation

    return {
        "top": (
            top_center,
            top_bottom_half_sizes.copy(),
        ),
        "bottom": (
            bottom_center,
            top_bottom_half_sizes.copy(),
        ),
        "left": (
            left_center,
            left_right_half_sizes.copy(),
        ),
        "right": (
            right_center,
            left_right_half_sizes.copy(),
        ),
    }


def _draw_gate_opening(
    ax,
    center,
    axis_u,
    axis_v,
    half_width,
    half_height,
    linewidth=1.2,
    alpha=0.8,
):
    """
    Draw the rectangular free opening in the gate plane.
    """
    c = np.asarray(center)
    u = np.asarray(axis_u)
    v = np.asarray(axis_v)

    corners = np.array(
        [
            c - half_width * u - half_height * v,
            c + half_width * u - half_height * v,
            c + half_width * u + half_height * v,
            c - half_width * u + half_height * v,
        ]
    )

    closed = np.vstack([corners, corners[0]])

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        closed[:, 2],
        ":",
        linewidth=linewidth,
        alpha=alpha,
    )


def _draw_oriented_box(
    ax,
    center,
    half_sizes,
    axis_u,
    axis_v,
    axis_n,
    linewidth=1.5,
    alpha=0.5,
    linestyle="-",
):
    """
    Draw an oriented 3D hyperbox.

    Parameters
    ----------
    center : shape (3,)
        World-space center.

    half_sizes : shape (3,)
        Half-sizes in local [u, v, n] coordinates.

    axis_u, axis_v, axis_n : shape (3,)
        Orthonormal world-space box axes.
    """
    center = np.asarray(center, dtype=float)
    half_sizes = np.asarray(half_sizes, dtype=float)

    basis = np.column_stack(
        [
            np.asarray(axis_u, dtype=float),
            np.asarray(axis_v, dtype=float),
            np.asarray(axis_n, dtype=float),
        ]
    )

    # Local corners in [u, v, n].
    local_corners = np.array(
        list(
            product(
                [-half_sizes[0], half_sizes[0]],
                [-half_sizes[1], half_sizes[1]],
                [-half_sizes[2], half_sizes[2]],
            )
        ),
        dtype=float,
    )

    # local row-vector q -> world = center + basis @ q
    corners = center[None, :] + local_corners @ basis.T

    # Two corners form an edge if they differ in exactly one local sign.
    for idx1, idx2 in combinations(range(8), 2):
        q1 = local_corners[idx1]
        q2 = local_corners[idx2]

        if np.sum(np.isclose(q1, q2)) == 2:
            p1 = corners[idx1]
            p2 = corners[idx2]

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
            )


def _draw_box(
    ax,
    center,
    half_width,
    alpha=0.3,
    linestyle="-",
):
    """
    Draw an axis-aligned 3D box corresponding to:

        |x - center| <= half_width

    This is retained for optionally showing the original waypoint constraint
    box; it is NOT the physical gate geometry.
    """
    center = np.asarray(center)
    half_width = np.asarray(half_width)

    lower = center - half_width
    upper = center + half_width

    corners = np.array(
        list(
            product(
                [lower[0], upper[0]],
                [lower[1], upper[1]],
                [lower[2], upper[2]],
            )
        )
    )

    for p1, p2 in combinations(corners, 2):
        if np.sum(np.isclose(p1, p2)) == 2:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                linewidth=1.2,
                alpha=alpha,
                linestyle=linestyle,
            )


def _normalize_gate_openings(gate_opening_half_widths, num_gates):
    """
    Normalize gate opening specification to shape (num_gates, 2).
    """
    if gate_opening_half_widths is None:
        return np.tile(
            np.array([[0.50, 0.50]], dtype=float),
            (num_gates, 1),
        )

    openings = np.asarray(gate_opening_half_widths, dtype=float)

    if openings.ndim == 0:
        return np.full(
            (num_gates, 2),
            float(openings),
            dtype=float,
        )

    if openings.shape == (2,):
        return np.tile(
            openings[None, :],
            (num_gates, 1),
        )

    if openings.shape != (num_gates, 2):
        raise ValueError(
            "gate_opening_half_widths must be scalar, shape (2,), "
            f"or shape ({num_gates}, 2); got {openings.shape}"
        )

    return openings


def _normalize_rows(array, eps=1e-12):
    """
    Normalize every row of a shape (N, 3) array.
    """
    array = np.asarray(array, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)

    if np.any(norms < eps):
        raise ValueError(
            "Cannot normalize a zero-length gate direction/axis."
        )

    return array / norms


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