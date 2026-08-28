from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_control_box_constraints, make_state_box_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_quadrotor_3d, plot_tube_graph_quadrotor
from visualize_drone_racing import visualize_trajectory_and_gates

config.update("jax_enable_x64", False)

DTYPE = jnp.float64 if config.x64_enabled else jnp.float32
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# -----------------------------
# 3D quadrotor parameters
# x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r, T]
# u = [T, tau_phi, tau_theta, tau_psi]
# -----------------------------
MASS = 1.0
GRAVITY = 9.81
E_MAG = 2.5
JX = 0.02
JY = 0.02
JZ = 0.04
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=DTYPE))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=DTYPE))

NUM_RANDOM = 5
NUM_ADV = 26


# -----------------------------
# Shared disturbance-field parameters
# -----------------------------
# These values are used by BOTH the optimization disturbance model and
# the top-down visualization, so changing them here changes both.
@dataclass(frozen=True)
class DisturbanceFieldParams:
    x_min: float = -2.0
    x_max: float = 1.0
    y_min: float = -8.0
    y_max: float = 0.0
    kx: float = 1.0
    ky: float = 1.0


DISTURBANCE_FIELD = DisturbanceFieldParams()


def disturbance_spatial_scale(px, py, params: DisturbanceFieldParams, xp):
    """Spatial window shared by the disturbance model and visualization."""
    x_window = (
        0.5 * (1.0 + xp.tanh(params.kx * (px - params.x_min)))
        *
        0.5 * (1.0 - xp.tanh(params.kx * (px - params.x_max)))
    )

    y_window = (
        0.5 * (1.0 + xp.tanh(params.ky * (py - params.y_min)))
        *
        0.5 * (1.0 - xp.tanh(params.ky * (py - params.y_max)))
    )

    return x_window * y_window


def rotation_matrix(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)

    # R = Rz(psi) Ry(theta) Rx(phi)
    return jnp.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                          cth * cphi],
    ], dtype=DTYPE)

def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta)
    cth = jnp.cos(theta)

    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=DTYPE)

def dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: Any,
) -> jnp.ndarray:
    """
    Multi-phase dynamics.

    State:
        x = [physical_state (12), T1, T2, ..., Tn]

    parameter should contain:
        waypoint_steps
        segment_lengths
    """

    waypoint_steps, segment_lengths = parameter

    num_phases = waypoint_steps.shape[0]

    # Determine which phase the current transition belongs to.
    #
    # t = 0, ..., N1-1       -> phase 0
    # t = N1, ..., N2-1      -> phase 1
    # t = N2, ..., N3-1      -> phase 2
    phase_idx = jnp.searchsorted(
        waypoint_steps,
        t,
        side="right",
    )

    # Safety in case dynamics is evaluated at the final state.
    phase_idx = jnp.clip(phase_idx, 0, num_phases - 1)

    # Duration corresponding to the active phase
    T_i = x[12 + phase_idx]

    # Number of discrete intervals in this phase
    N_i = segment_lengths[phase_idx]

    # Physical timestep for this phase
    dt = T_i / N_i

    return rigid_body_3d_step(x, u, dt)

def rigid_body_3d_step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:

    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x[:12]

    T, tau_phi, tau_theta, tau_psi = u

    v = jnp.array([vx, vy, vz], dtype=x.dtype)
    omega = jnp.array([p, q, r], dtype=x.dtype)
    tau = jnp.array(
        [tau_phi, tau_theta, tau_psi],
        dtype=x.dtype,
    )

    R = rotation_matrix(phi, theta, psi)
    E = euler_angle_rates_matrix(phi, theta)

    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    # Translational dynamics
    pos_dot = v
    v_dot = (R @ (T * e3)) / MASS - GRAVITY * e3

    # Rotational dynamics
    euler_dot = E @ omega
    omega_dot = J_INV @ (
        tau - jnp.cross(omega, J @ omega)
    )

    x_dot = jnp.concatenate([
        pos_dot,
        euler_dot,
        v_dot,
        omega_dot,
    ])

    physical_next = x[:12] + dt * x_dot

    # All duration variables are constant states:
    # T_i,k+1 = T_i,k
    return jnp.concatenate([
        physical_next,
        x[12:],
    ])


def cost(W, reference, x, u, t):
    """
    W =
    [wpx, wpy, wpz,
     wphi, wtheta, wpsi,
     wvx, wvy, wvz,
     wp, wq, wr,
     wT, wtau_phi, wtau_theta, wtau_psi, wtime]
    """
    (
        wpx, wpy, wpz,
        wphi, wtheta, wpsi,
        wvx, wvy, wvz,
        wp, wq, wr,
        wT, wtau_phi, wtau_theta, wtau_psi
    ) = W[:16]

    wtime = W[16:]

    xref = reference[t]

    dpos = x[:3] - xref[:3]
    dang = x[3:6] - xref[3:6]
    dvel = x[6:9] - xref[6:9]
    drates = x[9:12] - xref[9:12]

    T_hover = MASS * GRAVITY
    du = jnp.array([
        u[0] - T_hover,
        u[1],
        u[2],
        u[3],
    ], dtype=x.dtype)

    angle_cost = (
        wphi * (1.0 - jnp.cos(dang[0]))
        + wtheta * (1.0 - jnp.cos(dang[1]))
        + wpsi * (1.0 - jnp.cos(dang[2]))
    )

    durations = x[12:]
    time_cost = jnp.dot(wtime, durations)

    return (
        wpx * dpos[0] ** 2
        + wpy * dpos[1] ** 2
        + wpz * dpos[2] ** 2
        + angle_cost
        + wvx * dvel[0] ** 2
        + wvy * dvel[1] ** 2
        + wvz * dvel[2] ** 2
        + wp * drates[0] ** 2
        + wq * drates[1] ** 2
        + wr * drates[2] ** 2
        + wT * du[0] ** 2
        + wtau_phi * du[1] ** 2
        + wtau_theta * du[2] ** 2
        + wtau_psi * du[3] ** 2
        + time_cost
    )

def build_piecewise_reference(
    x0: jnp.ndarray,
    waypoint_centers: jnp.ndarray,
    waypoint_steps: jnp.ndarray,
    initial_durations: jnp.ndarray,
    N: int,
) -> jnp.ndarray:
    """
    Build a piecewise straight-line reference trajectory:

        x0 -> waypoint_1 -> waypoint_2 -> ... -> waypoint_n

    Each segment ends at its corresponding waypoint_steps[i].

    State ordering:
        [px, py, pz,
         phi, theta, psi,
         vx, vy, vz,
         p, q, r,
         T1, ..., Tn]
    """

    num_phases = waypoint_centers.shape[0]
    nx = 12 + num_phases

    X_ref = jnp.zeros((N + 1, nx), dtype=DTYPE)

    # --------------------------------------------------
    # Duration states are constant reference values
    # --------------------------------------------------
    X_ref = X_ref.at[:, 12:].set(
        jnp.broadcast_to(initial_durations, (N + 1, num_phases))
    )

    # --------------------------------------------------
    # Piecewise position / velocity reference
    # --------------------------------------------------
    start_pos = x0[:3]
    start_step = 0

    for i in range(num_phases):
        end_pos = waypoint_centers[i]
        end_step = int(waypoint_steps[i])

        segment_length = end_step - start_step

        # Includes both endpoints
        alpha = jnp.linspace(
            0.0,
            1.0,
            segment_length + 1
        )

        # Straight line between current point and waypoint
        pos_segment = (
            (1.0 - alpha[:, None]) * start_pos
            + alpha[:, None] * end_pos
        )

        X_ref = X_ref.at[start_step:end_step + 1, :3].set(
            pos_segment
        )

        # Constant velocity reference for this phase
        vel_segment = (
            end_pos - start_pos
        ) / initial_durations[i]

        X_ref = X_ref.at[start_step:end_step + 1, 6:9].set(
            vel_segment
        )

        # Next phase starts here
        start_pos = end_pos
        start_step = end_step

    return X_ref

def make_waypoint_constraints(
    waypoint_steps: jnp.ndarray,
    waypoint_centers: jnp.ndarray,
    waypoint_half_widths: jnp.ndarray,
):
    """
    Enforce

        |x[:3] - waypoint_centers[i]| <= waypoint_half_widths[i]

    at t == waypoint_steps[i] for every waypoint i.

    Each waypoint contributes 6 inequality constraints:
        x[:3] - center - half_width <= 0
        center - x[:3] - half_width <= 0

    At all other time steps, that waypoint's constraints are inactive
    and return -1.
    """

    waypoint_steps = jnp.asarray(waypoint_steps)
    waypoint_centers = jnp.asarray(waypoint_centers)
    waypoint_half_widths = jnp.asarray(waypoint_half_widths)

    def constraints(x, u, t):
        pos = x[:3]

        # Shape: (NUM_PHASES, 3)
        upper = (
            pos[None, :]
            - waypoint_centers
            - waypoint_half_widths
        )

        lower = (
            waypoint_centers
            - pos[None, :]
            - waypoint_half_widths
        )

        # Shape: (NUM_PHASES, 6)
        waypoint_constraints = jnp.concatenate(
            [upper, lower],
            axis=1,
        )

        # Shape: (NUM_PHASES,)
        active = (t == waypoint_steps)

        # Only activate waypoint i when t == waypoint_steps[i]
        waypoint_constraints = jnp.where(
            active[:, None],
            waypoint_constraints,
            -jnp.ones_like(waypoint_constraints),
        )

        # Return shape: (6 * NUM_PHASES,)
        return waypoint_constraints.reshape(-1)

    return constraints

def make_min_time_disturbance(
    n: int,
    E_mag,
    NUM_PHASES,
    WAYPOINT_STEPS,
    SEGMENT_LENGTHS,
    field_params: DisturbanceFieldParams = DISTURBANCE_FIELD,
):
    WAYPOINT_STEPS = jnp.asarray(WAYPOINT_STEPS)
    SEGMENT_LENGTHS = jnp.asarray(SEGMENT_LENGTHS)

    def disturbance_at_state(
        x_k: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        phase_idx = jnp.searchsorted(
            WAYPOINT_STEPS,
            t,
            side="right",
        )

        phase_idx = jnp.clip(
            phase_idx,
            0,
            NUM_PHASES - 1,
        )

        T_i = x_k[12 + phase_idx]
        N_i = SEGMENT_LENGTHS[phase_idx]
        dt = T_i / N_i

        px = x_k[0]
        py = x_k[1]

        # -------------------------------------------
        # Spatial disturbance field
        # -------------------------------------------
        # Uses the same helper/parameters as the visualization.
        spatial_scale = disturbance_spatial_scale(
            px,
            py,
            field_params,
            xp=jnp,
        )

        # Disturbance acts only on vy
        E_k = jnp.zeros((n, n), dtype=x_k.dtype)

        E_k = E_k.at[7, 7].set(
            dt * E_mag * spatial_scale
        )

        return E_k

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        ts = jnp.arange(X.shape[0])

        return jax.vmap(
            disturbance_at_state,
            in_axes=(0, 0),
        )(X, ts)

    disturbance.at_state = disturbance_at_state

    return disturbance

def build_controller(
    N,
    X_IN,
    WAYPOINT_STEPS,
    WAYPOINT_CENTERS,
    WAYPOINT_HALF_WIDTHS,
    SEGMENT_LENGTHS,
    disturbance_field: DisturbanceFieldParams = DISTURBANCE_FIELD,
):
    # -----------------------------
    # Dimensions
    # -----------------------------
    NUM_PHASES = len(WAYPOINT_STEPS)
    n = 12 + NUM_PHASES
    nu = 4
    TIME_WEIGHT = 5.0

    # -----------------------------
    # Cost weights
    # -----------------------------
    W = jnp.concatenate([jnp.array([
            0.001, 0.001, 0.001,        # position
            0.01, 0.01, 0.01,        # roll, pitch, yaw
            0.01, 0.01, 0.01,        # velocities
            0.01, 0.01, 0.01,        # body rates
            0.01, 0.01, 0.01, 0.01,  # control
        ], dtype=DTYPE),
        TIME_WEIGHT * jnp.ones(NUM_PHASES),  # time
    ])

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=DTYPE),
    )

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 2.0 * T_hover
    tau_max = 10.0

    u_min = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=DTYPE)
    u_max = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=DTYPE)
    constraints_u = make_control_box_constraints(u_min, u_max)

    MIN_TIME = 0.005
    MAX_TIME = 20.0

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.concatenate(
        [jnp.array([
            15.0, 15.0, 15.0,       # px, py, pz
            jnp.pi / 2.0,           # phi
            jnp.pi / 2.0,           # theta
            10.0 * jnp.pi,          # psi
            5.0, 5.0, 5.0,          # vx, vy, vz
            8.0, 8.0, 8.0,          # p, q, r
        ], dtype=DTYPE),
        MAX_TIME * jnp.ones(NUM_PHASES)
    ])
    x_min = -x_max
    x_min = x_min.at[2].set(-1.0)
    x_min = x_min.at[-NUM_PHASES:].set(MIN_TIME)

    constraints_x = make_state_box_constraints(x_min, x_max)
    waypoint_constraint = make_waypoint_constraints(
        WAYPOINT_STEPS,
        WAYPOINT_CENTERS,
        WAYPOINT_HALF_WIDTHS,
    )
    constraints_all = combine_constraints(
        constraints_x,
        constraints_u,
        waypoint_constraint
    )

    obstacles = jnp.zeros((0, 3))

    disturbance = make_min_time_disturbance(
        n=n,
        E_mag=E_MAG,
        NUM_PHASES=NUM_PHASES,
        WAYPOINT_STEPS=WAYPOINT_STEPS,
        SEGMENT_LENGTHS=SEGMENT_LENGTHS,
        field_params=disturbance_field,
    )

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e6,
        max_iterations=3000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        num_phases=NUM_PHASES,
    )

    # Q_bar = jnp.broadcast_to(jnp.eye(n), (N +1, n, n)).at[..., 1, 1].set(1000).at[..., 7, 7].set(0.001)
    # Q_bar = Q_bar + 1e-6 * jnp.eye(n)[None, :, :]
    # R_bar = jnp.broadcast_to(jnp.eye(nu) * 0.5, (N, nu, nu))
    Q_bar = jnp.broadcast_to(jnp.eye(n), (N +1, n, n))
    R_bar = jnp.broadcast_to(jnp.eye(nu), (N, nu, nu))

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=True,
        rti=False,
        gradient_window=10,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=10,
        line_search=True,
        lm_regularization=1e-2,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        disturbance=disturbance,
        shift=1,
        X_in=X_IN,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=DTYPE).at[:, 0].set(T_hover),
        Q_bar=Q_bar,
        R_bar=R_bar,
    )
    return controller

def plot_top_down_trajectory(
    X,
    waypoint_centers,
    waypoint_half_widths=None,
    reference=None,
    Phi_x=None,
    E_mag=0.05,
    disturbance_field: DisturbanceFieldParams = DISTURBANCE_FIELD,
    tube_stride=1,
    min_time=None,
    save_path="trajectory_topdown.png",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.colors import LinearSegmentedColormap

    X = np.asarray(X)
    waypoints = np.asarray(waypoint_centers)

    if reference is not None:
        ref = np.asarray(reference)
    else:
        ref = None

    # ==================================================
    # Plot bounds
    # ==================================================
    xy_data = [X[:, :2], waypoints[:, :2]]

    if ref is not None:
        xy_data.append(ref[:, :2])

    all_xy = np.concatenate(xy_data, axis=0)

    margin = 0.75

    # Bounds required by trajectory / waypoints / reference
    data_x_min = np.min(all_xy[:, 0]) - margin
    data_x_max = np.max(all_xy[:, 0]) + margin
    data_y_min = np.min(all_xy[:, 1]) - margin
    data_y_max = np.max(all_xy[:, 1]) + margin

    # Bounds required to show the disturbance-field tails
    tail_width = 4.0

    field_x_min = (
        disturbance_field.x_min
        - tail_width / disturbance_field.kx
    )
    field_x_max = (
        disturbance_field.x_max
        + tail_width / disturbance_field.kx
    )

    field_y_min = (
        disturbance_field.y_min
        - tail_width / disturbance_field.ky
    )
    field_y_max = (
        disturbance_field.y_max
        + tail_width / disturbance_field.ky
    )

    # Final plot bounds include both
    # x_min = min(data_x_min, field_x_min)
    # x_max = max(data_x_max, field_x_max)

    # y_min = min(data_y_min, field_y_min)
    # y_max = max(data_y_max, field_y_max)
    x_min = -6
    x_max = 5
    y_min = -7.5
    y_max = 8

    fig, ax = plt.subplots(figsize=(9, 8))

    # ==================================================
    # Disturbance field
    #
    # Uses the exact same helper and parameters as
    # make_min_time_disturbance().
    #
    # disturbance acts only in vy
    # ==================================================
    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)

    XX, YY = np.meshgrid(xs, ys)

    spatial_scale = disturbance_spatial_scale(
        XX,
        YY,
        disturbance_field,
        xp=np,
    )

    disturbance_field_values = E_mag * spatial_scale

    # White at zero, then smoothly transition into magma as the
    # disturbance magnitude increases.  The color scale is pinned
    # to [0, E_mag], so a true zero always renders as pure white.
    magma = plt.get_cmap("magma")
    disturbance_cmap = LinearSegmentedColormap.from_list(
        "white_to_magma",
        [
            (0.00, "white"),
            (0.08, magma(0.18)),
            (0.35, magma(0.40)),
            (0.65, magma(0.68)),
            (1.00, magma(1.00)),
        ],
    )

    # Explicit levels starting at zero guarantee that the bottom of
    # the colorbar/field corresponds to white.  No contour-line
    # overlay is drawn.
    field_levels = np.linspace(0.0, float(E_mag), 100)

    field = ax.contourf(
        XX,
        YY,
        disturbance_field_values,
        levels=field_levels,
        cmap=disturbance_cmap,
        vmin=0.0,
        vmax=float(E_mag),
        alpha=0.85,
        zorder=0,
    )

    cbar = fig.colorbar(field, ax=ax, pad=0.02)
    cbar.set_label("Disturbance magnitude on $v_y$")

    # ==================================================
    # SLS tubes
    #
    # EXACT tube calculation:
    #
    # get_trajectory_tubes(Phi_x)
    # =
    # norm(Phi_x, axis=-1).sum(axis=1)
    # ==================================================
    tube_handle = None

    if Phi_x is not None:

        tube = np.asarray(
            get_trajectory_tubes(Phi_x)
        )

        # print(tube)

        print("X shape:", X.shape)
        print("Phi_x shape:", np.asarray(Phi_x).shape)
        print("tube shape:", tube.shape)

        # Usually both are N+1.
        # Handle N tube nodes just in case x0 is omitted.
        if tube.shape[0] == X.shape[0] - 1:
            tube = np.concatenate(
                [
                    np.zeros(
                        (1, tube.shape[1]),
                        dtype=tube.dtype,
                    ),
                    tube,
                ],
                axis=0,
            )

        if tube.shape[0] != X.shape[0]:
            raise ValueError(
                f"Tube/X mismatch: "
                f"tube={tube.shape}, X={X.shape}"
            )

        # ----------------------------------------------
        # Draw XY projection of the box tube
        #
        # center:
        #   (X[k,0], X[k,1])
        #
        # half widths:
        #   tube[k,0], tube[k,1]
        # ----------------------------------------------
        for k in range(
            0,
            X.shape[0],
            max(1, tube_stride),
        ):

            cx = X[k, 0]
            cy = X[k, 1]

            rx = tube[k, 0]
            ry = tube[k, 1]

            if not np.all(
                np.isfinite([cx, cy, rx, ry])
            ):
                continue

            rect = Rectangle(
                (cx - rx, cy - ry),
                2.0 * rx,
                2.0 * ry,
                facecolor="cyan",
                edgecolor="blue",
                linewidth=0.6,
                alpha=0.18,
                zorder=2,
            )

            ax.add_patch(rect)

        tube_handle = Patch(
            facecolor="cyan",
            edgecolor="blue",
            alpha=0.18,
            label="SLS tube",
        )

    # ==================================================
    # Reference trajectory
    # ==================================================
    if ref is not None:
        ax.plot(
            ref[:, 0],
            ref[:, 1],
            "--",
            linewidth=1.5,
            label="Reference",
            zorder=4,
        )

    # ==================================================
    # Optimized trajectory
    # ==================================================
    ax.plot(
        X[:, 0],
        X[:, 1],
        "-o",
        markersize=2,
        linewidth=2,
        label="Optimized trajectory",
        zorder=5,
    )

    # ==================================================
    # Start
    # ==================================================
    ax.scatter(
        X[0, 0],
        X[0, 1],
        s=100,
        marker="x",
        linewidths=3,
        label="Start",
        zorder=7,
    )

    # ==================================================
    # Waypoints
    # ==================================================
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=90,
        marker="s",
        label="Waypoints",
        zorder=7,
    )

    for i, waypoint in enumerate(waypoints):
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
    # Waypoint boxes
    # ==================================================
    if waypoint_half_widths is not None:

        half_widths = np.asarray(
            waypoint_half_widths
        )

        for center, hw in zip(
            waypoints,
            half_widths,
        ):

            rect = Rectangle(
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

            ax.add_patch(rect)

    # ==================================================
    # Formatting
    # ==================================================
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    if min_time is not None:
        ax.set_title(
            f"Total Min Time: {float(min_time):.3f} s\n"
            "Top-Down Trajectory, Disturbance Field, and SLS Tube"
        )
    else:
        ax.set_title(
            "Top-Down Trajectory, Disturbance Field, and SLS Tube"
        )

    # ax.axis("equal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()

    if tube_handle is not None:
        handles.append(tube_handle)
        labels.append("SLS tube")

    # ax.legend(handles, labels)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved top-down trajectory to: {save_path}"
    )

def main():
    # Five waypoint / five phase problem.
    #
    # Approximate XY layout:
    #
    #       3
    #                         2
    #
    #          4              1
    #
    #                         x
    #          5
    #
    # The trajectory is constrained in the order:
    #     x -> 1 -> 2 -> 3 -> 4 -> 5
    #
    # Keep z constant so the geometry is essentially planar.
    N = 100
    nu = 4

    WAYPOINT_STEPS = jnp.array([
        20, 40, 60, 80, 100,
    ], dtype=jnp.int32)

    WAYPOINT_CENTERS = jnp.array([
        [ 2.5,  -1.0, 1.0],   # 1: above the start, right side
        [ 3.0,  5.0, 1.0],   # 2: upper right
        [-2.5,  7.0, 1.0],   # 3: upper left
        [-0.75,  3.0, 1.0],   # 4: middle left
        [-3, -6.0, 1.0],   # 5: lower left
    ], dtype=DTYPE)

    WAYPOINT_HALF_WIDTHS = 0.20 * jnp.ones(
        (len(WAYPOINT_STEPS), 3),
        dtype=DTYPE,
    )

    SEGMENT_LENGTHS = jnp.concatenate([
        WAYPOINT_STEPS[:1],
        WAYPOINT_STEPS[1:] - WAYPOINT_STEPS[:-1],
    ]).astype(DTYPE)

    INITIAL_DURATIONS = 1.0 * jnp.ones(
        len(WAYPOINT_STEPS),
        dtype=DTYPE,
    )

    x0 = jnp.concatenate([
        jnp.array([
            2.5, -3.0, 1.0,   # px, py, pz: x in the sketch
            0.0,  0.0, 0.0,   # phi, theta, psi
            0.0,  0.0, 0.0,   # vx, vy, vz
            0.0,  0.0, 0.0,   # p, q, r
        ], dtype=DTYPE),
        INITIAL_DURATIONS,
])

    reference = build_piecewise_reference(
        x0=x0,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_steps=WAYPOINT_STEPS,
        initial_durations=INITIAL_DURATIONS,
        N=N,
    )
    # TODO: Here
    controller = build_controller(
        N,
        reference,
        WAYPOINT_STEPS,
        WAYPOINT_CENTERS,
        WAYPOINT_HALF_WIDTHS,
        SEGMENT_LENGTHS,
        disturbance_field=DISTURBANCE_FIELD,
    )
    parameter = (WAYPOINT_STEPS, SEGMENT_LENGTHS)
    # -----------------------------
    # Robust plan
    # -----------------------------
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    controller.reset(reference, jnp.zeros((N, nu), dtype=DTYPE).at[:, 0].set(T_hover))
    phase_times = X_pred[0, 12:]
    # import time
    # start = time.perf_counter()
    # u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
    #     x0=x0, reference=reference, parameter=parameter
    # )
    # print(Phi_x)
    # end = time.perf_counter()
    # print(end - start)
    min_time = jnp.sum(phase_times)

    print("Phase Times:", phase_times)
    print("Computed Total Min Time:", min_time)

    # -----------------------------
    # Save optimized trajectory
    # -----------------------------
    trajectory_path = "multiphase_trajectory.npz"
    np.savez(
        trajectory_path,
        X=np.asarray(X_pred),
        U=np.asarray(U_pred),
        phase_times=np.asarray(phase_times),
        min_time=np.asarray(min_time),
        waypoint_steps=np.asarray(WAYPOINT_STEPS),
        waypoint_centers=np.asarray(WAYPOINT_CENTERS),
        waypoint_half_widths=np.asarray(WAYPOINT_HALF_WIDTHS),
        segment_lengths=np.asarray(SEGMENT_LENGTHS),
        reference=np.asarray(reference),
    )
    print(f"Saved optimized trajectory to: {trajectory_path}")


    visualize_trajectory_and_gates(
        X=X_pred,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_half_widths=WAYPOINT_HALF_WIDTHS,
        waypoint_steps=WAYPOINT_STEPS,
        reference=reference,
        save_path="multiphase_trajectory.png",
    )

    plot_top_down_trajectory(
        X=X_pred,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_half_widths=WAYPOINT_HALF_WIDTHS,
        reference=reference,
        Phi_x=Phi_x,
        E_mag=E_MAG,
        disturbance_field=DISTURBANCE_FIELD,
        tube_stride=2,
        min_time=min_time,
        save_path="trajectory_topdown.png",
    )

    # -----------------------------
    # Rollout simulations
    # -----------------------------


if __name__ == "__main__":
    main()