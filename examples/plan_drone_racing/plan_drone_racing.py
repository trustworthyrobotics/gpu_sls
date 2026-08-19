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

JX = 0.02
JY = 0.02
JZ = 0.04
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=DTYPE))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=DTYPE))

NUM_RANDOM = 5
NUM_ADV = 26


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

def quadrotor_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,      # (13,)
    u: jnp.ndarray,      # (4,)
    E: jnp.ndarray,      # (13,13)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = f(x_k, u_k) + E w, with ||w||_2 <= 1
    """
    x_nom = rigid_body_3d_step(x, u, dt)

    # Random disturbance in unit ball
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (12,), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(12, dtype=x.dtype)
    a = jnp.asarray(0.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = jnp.concatenate([r * z, jnp.zeros((1,), dtype=x.dtype)])

    # Deterministic adversarial-ish directions
    start = i - NUM_RANDOM + 5
    if 5 <= start <= 16:
        idx = start - 5
        w = jnp.zeros((13,), dtype=x.dtype).at[idx].set(1.0)
    if 17 <= start <= 28:
        idx = start - 17
        w = jnp.zeros((13,), dtype=x.dtype).at[idx].set(-1.0)
    if start == 29:
        w = jnp.concatenate([jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype)), jnp.zeros((1,), dtype=x.dtype)])
    if start == 30:
        w = jnp.concatenate([-jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype)), jnp.zeros((1,), dtype=x.dtype)])

    # Account for linearization error
    w = w.at[-1].set(0.0)
    x_next = x_nom + E @ w * dt
    return key, x_next, w

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

        E_k = dt * E_mag * jnp.eye(
            n,
            dtype=x_k.dtype,
        )

        # No disturbance on duration states
        E_k = E_k.at[12:, :].set(0.0)
        E_k = E_k.at[:, 12:].set(0.0)

        return E_k

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        ts = jnp.arange(X.shape[0])

        return jax.vmap(
            disturbance_at_state,
            in_axes=(0, 0),
        )(X, ts)

    disturbance.at_state = disturbance_at_state

    return disturbance

def build_controller(N, X_IN, WAYPOINT_STEPS, WAYPOINT_CENTERS, WAYPOINT_HALF_WIDTHS, SEGMENT_LENGTHS):
    # -----------------------------
    # Dimensions
    # -----------------------------
    NUM_PHASES = len(WAYPOINT_STEPS)
    n = 12 + NUM_PHASES
    nu = 4
    TIME_WEIGHT = 0.5

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
        # constraints_x,
        constraints_u,
        waypoint_constraint
    )

    obstacles = jnp.zeros((0, 3))

    E_mag = 0.1
    disturbance = make_min_time_disturbance(n=n, E_mag=E_mag, NUM_PHASES=NUM_PHASES, WAYPOINT_STEPS=WAYPOINT_STEPS, SEGMENT_LENGTHS=SEGMENT_LENGTHS)

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-3,
        rho_max=1e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        num_phases=NUM_PHASES,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=0,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=0.1,
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
    )
    return controller

def run_rollouts(N, X_pred, U_pred, x0, min_time, Phi_x, Phi_u, disturbance_fn, waypoints):
    pass
    # N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    # n = X_pred.shape[-1]
    # nu = U_pred.shape[-1]
    # xs = np.full((N_ROLLOUTS, N, n), np.nan, dtype=np.float64)
    # disturbed = np.full((N_ROLLOUTS, N, n), np.nan, dtype=np.float64)
    # stop_steps = np.full((N_ROLLOUTS,), N, dtype=np.int32)
    # min_time = X_pred[0, -1]
    # dt = min_time / N
    # for i in range(N_ROLLOUTS):
    #     disturbance_history = [jnp.zeros((n,), dtype=DTYPE)]
    #     x = x0.at[-1].set(min_time)
    #     jax.debug.print("Rolling out iteration {}", i)

    #     for k in range(N):
    #         disturbance_feedback = jnp.zeros((nu,), dtype=DTYPE)
    #         for j in range(k + 1):
    #             disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

    #         u = U_pred[k] + disturbance_feedback
    #         key, x, w = quadrotor_step_with_disturbance(key, x, u, disturbance_fn(x), dt, i)

    #         err = np.abs(np.asarray(X_pred[k + 1] - x))

    #         disturbed[i, k, :] = err
    #         disturbance_history.append(w)
    #         xs[i, k] = np.asarray(x)

    # # -----------------------------
    # # 3D tube visualization
    # # -----------------------------
    # tube = get_trajectory_tubes(Phi_x)
    # lower = X_pred[:, :3] - tube[:, :3]
    # upper = X_pred[:, :3] + tube[:, :3]

    # obstacle_centers = jnp.array([
    #     [0.0, 0.0, 0.0],
    # ], dtype=DTYPE)

    # obstacle_radii = jnp.array([
    #     0.01,
    # ], dtype=DTYPE)

    # plot_quadrotor_3d(
    #     xs=xs,
    #     plan=np.asarray(X_pred),
    #     lower=np.asarray(lower),
    #     upper=np.asarray(upper),
    #     centers=np.asarray(obstacle_centers),
    #     radii=np.asarray(obstacle_radii),
    #     goal_center=np.asarray(terminal_center),
    #     goal_half_width=np.asarray(terminal_half_width),
    #     tube_stride=1,
    #     filename="quadrotor_3d_rollouts_tube.png",
    #     tube_alpha=0.08,
    #     margin=0.2,
    #     rollout_alpha=0.5,
    #     title="Minimum-Time Quadrotor: 3D Spherical Obstacles",
    # )

    # plot_tube_graph_quadrotor(
    #     disturbed=disturbed[:, :, :6],   # position + Euler angles only
    #     tube=tube[:, :6],
    #     dt=dt,
    #     filename="quadrotor_3d_disturbance_vs_tube_size_pose.png",
    # )

def main():
    N = 180
    nu = 4

    WAYPOINT_STEPS = jnp.array([
        15,  30,  45,  60,
        75,  90, 105, 120,
        135, 150, 165, 180,
    ], dtype=jnp.int32)

    WAYPOINT_CENTERS = jnp.array([
        [ 1.5,  0.0, 1.2],
        [ 2.8,  1.5, 1.5],
        [ 1.8,  3.0, 2.0],
        [ 0.0,  2.0, 2.5],
        [-1.8,  3.0, 1.8],
        [-3.0,  1.2, 1.1],
        [-1.5,  0.0, 0.8],
        [ 0.0,  1.5, 1.4],
        [ 1.8,  0.0, 2.2],
        [ 0.5, -2.0, 2.6],
        [-2.0, -2.5, 1.5],
        [ 0.0,  0.0, 1.2],
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
            0.0, -1.0, 1.0,   # px, py, pz
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
    controller = build_controller(N, reference, WAYPOINT_STEPS, WAYPOINT_CENTERS, WAYPOINT_HALF_WIDTHS, SEGMENT_LENGTHS)
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
    import time
    start = time.perf_counter()
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    end = time.perf_counter()
    print(end - start)
    min_time = jnp.sum(phase_times)

    print("Phase Times:", phase_times)
    print("Computed Total Min Time:", min_time)


    visualize_trajectory_and_gates(
        X=X_pred,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_half_widths=WAYPOINT_HALF_WIDTHS,
        waypoint_steps=WAYPOINT_STEPS,
        reference=reference,
        save_path="multiphase_trajectory.png",
    )

    # -----------------------------
    # Rollout simulations
    # -----------------------------


if __name__ == "__main__":
    main()