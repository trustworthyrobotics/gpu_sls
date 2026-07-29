from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

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
from visualize_experiment import plot_controls, plot_quadrotor_3d, plot_tube_graph_quadrotor, plot_top_down_tubes_goal_obstacles

#export ACADOS_SOURCE_DIR=/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/acados_baseline/acados
# export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
#
#


# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.25  # meters in xyz


def reached_goal_xyz(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dpos = x[:3] - x_goal[:3]
    return (dpos @ dpos) <= (tol * tol)

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
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=jnp.float32))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=jnp.float32))

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
    ], dtype=jnp.float32)

def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta)
    cth = jnp.cos(theta)

    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=jnp.float32)

def rigid_body_3d_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x[:12]
    T, tau_phi, tau_theta, tau_psi = u

    v = jnp.array([vx, vy, vz], dtype=x.dtype)
    omega = jnp.array([p, q, r], dtype=x.dtype)
    tau = jnp.array([tau_phi, tau_theta, tau_psi], dtype=x.dtype)

    R = rotation_matrix(phi, theta, psi)
    E = euler_angle_rates_matrix(phi, theta)

    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    # Translational dynamics
    pos_dot = v
    v_dot = (R @ (T * e3)) / MASS - GRAVITY * e3

    # Rotational dynamics
    euler_dot = E @ omega
    omega_dot = J_INV @ (tau - jnp.cross(omega, J @ omega))

    x_dot = jnp.concatenate([pos_dot, euler_dot, v_dot, omega_dot], axis=0)
    physical_next = x[:12] + dt * x_dot
    if x.shape[0] == 13:
        return jnp.concatenate([physical_next, x[12:13]])
    return physical_next

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

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dtau = parameter
    return rigid_body_3d_step(x, u, dtau * x[-1])

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
        wT, wtau_phi, wtau_theta, wtau_psi, wtime
    ) = W

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
        + wtime * x[-1]
    )

def build_piecewise_reference(x0: jnp.ndarray, x_goal: jnp.ndarray, N: int, duration: float) -> jnp.ndarray:
    """
    Build a straight-line reference trajectory from x0 to x_goal.

    Only position (px,py,pz) and yaw are interpolated.
    All other states are set to zero reference.
    """

    t = jnp.linspace(0.0, 1.0, N + 1)

    # Linear interpolation for position
    pos = (1.0 - t[:, None]) * x0[:3] + t[:, None] * x_goal[:3]

    # Shortest-path yaw interpolation
    dpsi = x_goal[5] - x0[5]
    psi = x0[5] + t * dpsi

    X_ref = jnp.zeros((N + 1, 13), dtype=jnp.float32)

    X_ref = X_ref.at[:, :3].set(pos)
    X_ref = X_ref.at[:, 5].set(psi)

    # Optionally compute velocity reference from the line
    vel = (x_goal[:3] - x0[:3]) / duration
    X_ref = X_ref.at[:, 6:9].set(vel)
    X_ref = X_ref.at[:, -1].set(duration)

    return X_ref

def make_terminal_set_constraint(center: jnp.ndarray, half_width: jnp.ndarray, N: int):
    """Enforce |x[:3] - center| <= half_width at the terminal step."""
    center = jnp.asarray(center)
    half_width = jnp.asarray(half_width)

    def constraints(x, u, t):
        terminal = jnp.concatenate([
            x[:3] - center - half_width,
            center - x[:3] - half_width,
        ])
        return jnp.where(t == N, terminal, -jnp.ones_like(terminal))

    return constraints


# def make_min_time_disturbance(n: int, E_mag: float):
#     """Scale physical-state uncertainty by the optimized integration step."""
#     def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
#         dt = X_prefix[0, -1] / (X_prefix.shape[0] - 1)
#         E0 = dt * E_mag * jnp.eye(n, dtype=X_prefix.dtype)
#         E0 = E0.at[-1, -1].set(0.0)
#         return jnp.broadcast_to(E0, (X_prefix.shape[0], n, n))

#     return disturbance

# def make_min_time_disturbance(
#     n: int,
#     E_mag: float,
#     N: int,
#     disturbance_index: int = 6,
# ):
#     """Create pointwise and trajectory disturbance maps.

#     Assumption:
#         The last component of every state x_k is the optimized final time.
#     """

#     def disturbance_at_state(x_k: jnp.ndarray) -> jnp.ndarray:
#         """
#         x_k: shape (n,)
#         returns: shape (n, n)
#         """
#         final_time = x_k[-1]  # scalar
#         dt = final_time / N

#         E_k = jnp.zeros((n, n), dtype=x_k.dtype)

#         return E_k.at[
#             disturbance_index,
#             disturbance_index,
#         ].set(dt * E_mag)

#     def disturbance(X: jnp.ndarray) -> jnp.ndarray:
#         """
#         X: shape (T + 1, n)
#         returns: shape (T + 1, n, n)
#         """
#         return jax.vmap(disturbance_at_state)(X)

#     # Expose the pointwise function for local differentiation.
#     disturbance.at_state = disturbance_at_state

#     return disturbance

def make_min_time_disturbance(
    n: int,
    N: int,
    disturbance_index: int = 6,
):
    """
    Disturbance magnitude:
        - 3.0 for z >= 0.25
        - decreases quadratically to 0.1 at z = 0
        - clipped at 0.1 below z = 0
    """

    def disturbance_at_state(x_k: jnp.ndarray) -> jnp.ndarray:
        z = x_k[2]
        final_time = x_k[-1]
        dt = final_time / N

        # Normalize altitude into [0, 1]
        s = jnp.clip(z / 0.25, 0.0, 1.0)

        # Quadratic profile
        E_mag = 0.1 + (3.0 - 0.1) * s**2

        E_k = jnp.zeros((n, n), dtype=x_k.dtype)
        E_k = E_k.at[
            disturbance_index,
            disturbance_index,
        ].set(dt * E_mag)

        return E_k

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(disturbance_at_state)(X)

    disturbance.at_state = disturbance_at_state
    return disturbance

def make_sphere_obstacle_constraint(
    center: jnp.ndarray,
    radius: float,
    clearance: float = 0.0,
):
    """
    Create a smooth 3D spherical obstacle-avoidance constraint.

    The returned constraint follows the convention:
        g(x, u, t) <= 0

    and enforces:
        ||x[:3] - center||_2 >= radius + clearance

    Args:
        center:
            Sphere center [cx, cy, cz], shape (3,).
        radius:
            Physical obstacle radius.
        clearance:
            Additional safety distance around the obstacle.

    Returns:
        constraints(x, u, t), returning shape (1,).
    """
    center = jnp.asarray(center)
    safe_radius = jnp.asarray(radius + clearance)

    def constraints(
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        del u, t

        displacement = x[:3] - center

        # <= 0 outside/on the sphere; > 0 inside the sphere.
        sphere_constraint = (
            safe_radius**2
            - jnp.dot(displacement, displacement)
        )

        return jnp.reshape(sphere_constraint, (1,))

    return constraints

def main():
    # -----------------------------
    # Dimensions
    # -----------------------------
    n = 13
    nu = 4

    # -----------------------------
    # Horizon and dt
    # -----------------------------
    N = 30
    parameter = 1.0 / N
    initial_duration = 2.0

    # -----------------------------
    # Cost weights
    # -----------------------------
    # W = jnp.array([
    #     0.0, 0.0, 0.0,     # position
    #     0.1, 0.1, 0.1,        # roll, pitch, yaw
    #     0.5, 0.5, 0.5,        # velocities
    #     0.05, 0.05, 0.05,     # body rates
    #     0.01, 0.01, 0.01, 0.01,  # control
    #     5.0                         # total time
    # ], dtype=jnp.float32)
    W = jnp.array([
        0.01, 0.01, 0.01,     # position
        0.01, 0.01, 0.01,        # roll, pitch, yaw
        0.01, 0.01, 0.01,        # velocities
        0.01, 0.01, 0.01,     # body rates
        0.01, 0.01, 0.01, 0.01,  # control
        5.0                         # total time
    ], dtype=jnp.float32)
    

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 2.0 * T_hover
    tau_max = 10.0

    u_min = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=jnp.float32)
    u_max = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=jnp.float32)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.array([
        15.0, 15.0, 15.0,       # px, py, pz
        jnp.pi / 2.0,           # phi
        jnp.pi / 2.0,           # theta
        10.0 * jnp.pi,          # psi
        5.0, 5.0, 5.0,          # vx, vy, vz
        8.0, 8.0, 8.0,          # p, q, r
        20.0                      # total time
    ], dtype=jnp.float32)
    x_min = -x_max
    x_min = x_min.at[2].set(-1.0)
    x_min = x_min.at[-1].set(0.1)

    constraints_x = make_state_box_constraints(x_min, x_max)
    terminal_center = jnp.array([1.0, 0.1, 0.5], dtype=jnp.float32)
    terminal_half_width = jnp.array([0.4, 0.4, 0.4], dtype=jnp.float32)
    terminal_constraint = make_terminal_set_constraint(
        terminal_center, terminal_half_width, N
    )
    obstacle_constraint = make_sphere_obstacle_constraint(
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32), radius=0.4
    )
    constraints_all = combine_constraints(
        constraints_x, constraints_u, terminal_constraint, obstacle_constraint
    )

    # obstacles = jnp.array([
    #     [0.0, 0.0, 0.35],
    # ], dtype=jnp.float32)

    obstacles = jnp.zeros((0, 3))

    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs + 6 + 1

    E_mag = 3.5
    # disturbance = make_min_time_disturbance(n=n, E_mag=E_mag, N=N)
    disturbance = make_min_time_disturbance(n=n, N=N)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0 = jnp.array([
        -0.75, -0.1, 0.25,    # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0,          # p, q, r
        initial_duration             # total time
    ], dtype=jnp.float32)

    x_goal = jnp.array([
        1.0, 0.1, 0.5,          # px, py, pz
        0.0, 0.0, 0.0,          # phi, theta, psi
        0.0, 0.0, 0.0,          # vx, vy, vz
        0.0, 0.0, 0.0,          # p, q, r
        initial_duration             # total time
    ], dtype=jnp.float32)

    X_ref = build_piecewise_reference(x0, x_goal, N, initial_duration)
    reference = X_ref
    T_steps = N

    key = jax.random.PRNGKey(0)
    E_sim = E_mag * jnp.eye(n, dtype=jnp.float32)
    E_sim = E_sim.at[-1, -1].set(0.0)

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=5e-4,
        rho_max=1e3,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1e-2,
        regularized_rho_update=False,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=True,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=50,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=1e-10,
        line_search=True,
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
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_ref,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float32).at[:, 0].set(T_hover),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    import time
    # start = time.perf_counter()
    # u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
    #         x0=x0, reference=reference, parameter=parameter
    #     )
    # end = time.perf_counter()
    # print(end - start)
    min_time = X_pred[0, -1]
    dt = min_time / N
    print("Computed Min Time:", min_time)

    # -----------------------------
    # Rollout simulations
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float32)
    disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float32)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

    # for i in range(N_ROLLOUTS):
    #     disturbance_history = [jnp.zeros((n,), dtype=jnp.float32)]
    #     x = x0.at[-1].set(min_time)
    #     jax.debug.print("Rolling out iteration {}", i)

    #     for k in range(T_steps):
    #         disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float32)
    #         for j in range(k + 1):
    #             disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

    #         u = U_pred[k] + disturbance_feedback
    #         u = jnp.clip(u, u_min, u_max)

    #         key, x, w = quadrotor_step_with_disturbance(key, x, u, E_sim, dt, i)

    #         err = np.abs(np.asarray(X_pred[k + 1] - x))

    #         disturbed[i, k, :] = err
    #         disturbance_history.append(w)
    #         xs[i, k] = np.asarray(x)

    # -----------------------------
    # 3D tube visualization
    # -----------------------------
    tube = get_trajectory_tubes(Phi_x)
    lower = X_pred[:, :3] - tube[:, :3]
    upper = X_pred[:, :3] + tube[:, :3]

    obstacle_centers = jnp.array([
        [0.0, 0.0, 0.0],
    ], dtype=jnp.float32)

    obstacle_radii = jnp.array([
        0.40,
    ], dtype=jnp.float32)

    plot_quadrotor_3d(
        xs=xs,
        plan=np.asarray(X_pred),
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        centers=np.asarray(obstacle_centers),
        radii=np.asarray(obstacle_radii),
        goal_center=np.asarray(terminal_center),
        goal_half_width=np.asarray(terminal_half_width),
        tube_stride=1,
        filename="quadrotor_3d_rollouts_tube.png",
        tube_alpha=0.08,
        margin=0.2,
        rollout_alpha=0.5,
        title="Minimum-Time Quadrotor: 3D Spherical Obstacles",
    )

    plot_tube_graph_quadrotor(
        disturbed=disturbed[:, :, :6],   # position + Euler angles only
        tube=tube[:, :6],
        dt=dt,
        filename="quadrotor_3d_disturbance_vs_tube_size_pose.png",
    )
    plot_controls(
        controls=np.asarray(U_pred),
        dt=dt,
        u_min=np.asarray(u_min),
        u_max=np.asarray(u_max),
        filename="quadrotor_controls.png",
    )
    plot_top_down_tubes_goal_obstacles(
        plan=X_pred,
        lower=lower,
        upper=upper,
        centers=obstacle_centers,
        radii=obstacle_radii,
        goal_center=np.asarray(terminal_center),
        goal_half_width=np.asarray(terminal_half_width),
        tube_stride=1,
        filename="quadrotor_top_down_tubes.png",
    )


if __name__ == "__main__":
    main()
