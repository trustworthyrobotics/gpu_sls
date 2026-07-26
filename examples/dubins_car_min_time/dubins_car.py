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
from gpu_sls.utils.constraint_utils import combine_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import plot_controls, plot_rollouts_tubes_centers, plot_tube_graph

config.update("jax_enable_x64", False)

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.2  # meters (XY distance)

def reached_goal_xy(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dxy = x[:2] - x_goal[:2]
    return (dxy @ dxy) <= (tol * tol)


# -----------------------------
# Angle wrapping
# -----------------------------
def wrap_to_pi(a: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles elementwise to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

# -----------------------------
# Dubins car dynamics
# x = [px, py, theta], u = [omega]
# -----------------------------
V_CONST = 0.2
NUM_RANDOM = 5
NUM_ADV = 26

def dubins_step_with_disturbance(
    key: jax.Array,          # PRNGKey
    x: jnp.ndarray,          # (3,)
    u: jnp.ndarray,          # (1,)
    E: jnp.ndarray,          # (3,3)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    Simulates: x_{k+1} = f(x_k,u_k) + E w,   with ||w||_2 <= 1
    where w is sampled from a unit-ball-ish distribution (plus some deterministic cases).

    Returns (key_next, x_next, w).
    """
    px, py, th, T = x
    v, om = u[0], u[1]

    # Nominal Dubins step
    px_next = px + dt * v * jnp.cos(th)
    py_next = py + dt * v * jnp.sin(th)
    th_next = wrap_to_pi(th + dt * om)
    x_nom = jnp.array([px_next, py_next, th_next, 0.0], dtype=x.dtype)

    # Stronger disturbance sampling
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (x.shape[0],), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(x.shape[0] - 1, dtype=x.dtype)
    a = jnp.asarray(1.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = r * z

    # Optional deterministic set of w's for "adversarial" rollouts
    # jax.debug.print("{}", w)
    start = i - NUM_RANDOM + 5
    if start == 5:
        w = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=x.dtype)
    if start == 6:
        w = jnp.array([0.0, -1.0, 0.0, 0.0], dtype=x.dtype)
    if start == 7:
        w = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=x.dtype)
    if start == 8:
        w = jnp.array([-1.0, 0.0, 0.0, 0.0], dtype=x.dtype)
    if start == 9:
        w = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=x.dtype)
    if start == 10:
        w = jnp.array([0.0, 0.0, -1.0, 0.0], dtype=x.dtype)
    if start == 11:
        w = jnp.array([0.707, 0.707, 0.0, 0.0], dtype=x.dtype)
    if start == 12:
        w = jnp.array([-0.707, 0.707, 0.0, 0.0], dtype=x.dtype)
    if start == 13:
        w = jnp.array([0.707, -0.707, 0.0, 0.0], dtype=x.dtype)
    if start == 14:
        w = jnp.array([-0.707, -0.707, 0.0, 0.0], dtype=x.dtype)
    if start == 15:
        w = jnp.array([0.707, 0.0, 0.707, 0.0], dtype=x.dtype)
    if start == 16:
        w = jnp.array([-0.707, 0.0, 0.707, 0.0], dtype=x.dtype)
    if start == 17:
        w = jnp.array([0.707, 0.0, -0.707, 0.0], dtype=x.dtype)
    if start == 18:
        w = jnp.array([-0.707, 0.0, -0.707, 0.0], dtype=x.dtype)
    if start == 19:
        w = jnp.array([0.0, 0.707, 0.707, 0.0], dtype=x.dtype)
    if start == 20:
        w = jnp.array([0.0, -0.707, 0.707, 0.0], dtype=x.dtype)
    if start == 21:
        w = jnp.array([0.0, 0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 22:
        w = jnp.array([0.0, -0.707, -0.707, 0.0], dtype=x.dtype)
    if start == 23:
        w = jnp.array([0.577, 0.577, 0.577, 0.0], dtype=x.dtype)
    if start == 24:
        w = jnp.array([-0.577, 0.577, 0.577, 0.0], dtype=x.dtype)
    if start == 25:
        w = jnp.array([0.577, -0.577, 0.577, 0.0], dtype=x.dtype)
    if start == 26:
        w = jnp.array([0.577, 0.577, -0.577, 0.0], dtype=x.dtype)
    if start == 27:
        w = jnp.array([-0.577, -0.577, 0.577, 0.0], dtype=x.dtype)
    if start == 28:
        w = jnp.array([0.577, -0.577, -0.577, 0.0], dtype=x.dtype)
    if start == 29:
        w = jnp.array([-0.577, 0.577, -0.577, 0.0], dtype=x.dtype)
    if start == 30:
        w = jnp.array([-0.577, -0.577, -0.577, 0.0], dtype=x.dtype)

    # Additive disturbance
    w = w.at[-1].set(0.0)
    x_next = x_nom + E @ w * dt
    return key, x_next, w

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    """Discrete-time dynamics required by your model evaluator."""
    dtau = parameter
    px, py, th, T = x[0], x[1], x[2], x[3]
    v, om = u
    dt = dtau * T
    px_next = px + dt * v * jnp.cos(th)
    py_next = py + dt * v * jnp.sin(th)
    th_next = th + dt * om
    return jnp.array([px_next, py_next, th_next, T], dtype=x.dtype)

def cost(W, reference, x, u, t):
    """
    W = [wx, wy, wtheta, womega, wT]
    """
    wx, wy, wtheta, wvel, womega, wT = W
    xref = reference[t]

    dx = x[0] - xref[0]
    dy = x[1] - xref[1]
    dth = x[2] - xref[2]
    theta_cost = 1 - jnp.cos(dth)
    time_cost = wT * x[3]

    v = u[0]
    om = u[1]

    return (
        wx * (dx * dx)
        + wy * (dy * dy)
        + wtheta * theta_cost
        + wvel * (v * v)
        + womega * (om * om)
        + time_cost
    )

def make_control_box_constraints(
    u_min: jnp.ndarray,
    u_max: jnp.ndarray
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for control bounds:
      u - u_max <= 0
      u_min - u <= 0
    """
    u_min = jnp.asarray(u_min)
    u_max = jnp.asarray(u_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([u - u_max, u_min - u], axis=0)

    return constraints

def make_terminal_set_constraint(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    N: int,
):
    """
    Enforces at t = N, elementwise:

        |x[:2] - alpha| <= beta

    Returns 4 scalar inequalities:
        x[:2] - alpha - beta <= 0
        alpha - x[:2] - beta <= 0
    """
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)

    def constraints(x, u, t):
        terminal_constraint = jnp.concatenate([
            x[:2] - alpha - beta,
            alpha - x[:2] - beta,
        ])

        inactive_constraint = -jnp.ones_like(terminal_constraint)

        return jnp.where(
            t == N,
            terminal_constraint,
            inactive_constraint,
        )

    return constraints

def make_state_box_constraints(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Inequality constraints g(x,u,t) <= 0 for state bounds:
      x - x_max <= 0
      x_min - x <= 0
    """
    x_min = jnp.asarray(x_min)
    x_max = jnp.asarray(x_max)

    def constraints(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x - x_max, x_min - x], axis=0)

    return constraints

def make_straight_line_reference(
    x_start: jnp.ndarray,
    x_goal: jnp.ndarray,
    N: int,
) -> jnp.ndarray:
    # Evenly spaced interpolation parameter
    alpha = jnp.linspace(0.0, 1.0, N + 1)

    # Straight-line interpolation in x-y
    xy_ref = (
        (1.0 - alpha)[:, None] * x_start[:2]
        + alpha[:, None] * x_goal[:2]
    )

    # Constant heading pointing toward the goal
    delta = x_goal[:2] - x_start[:2]
    theta_ref = jnp.arctan2(delta[1], delta[0])
    theta_vec = jnp.full((N + 1,), theta_ref)

    return jnp.column_stack((
        xy_ref[:, 0],
        xy_ref[:, 1],
        theta_vec,
        jnp.ones(N + 1) * 2.0
    ))

def make_constant_disturbance(
    n: int,
    E_mag: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Returns a constant disturbance E with shape (T, n, n),
    where E[t] = mag * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        TP1 = X_prefix.shape[0]
        MIN_T = X_prefix[0, -1]
        E0 = MIN_T / (TP1 - 1) * E_mag * jnp.eye(n, n, dtype=X_prefix.dtype)  # (n, n)
        E0 = E0.at[-1, -1].set(0.0)
        return jnp.broadcast_to(E0, (TP1, n, n))

    return disturbance

def make_circular_obstacle_constraints(
    obstacles: jnp.ndarray,
    margin: float = 0.0,
    eps: float = 1e-8,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Create circular obstacle constraints of the form

        radius_i + margin - ||x[:2] - center_i||_2 <= 0.

    Parameters
    ----------
    obstacles:
        Array with shape (num_obstacles, 3), where each row is

            [center_x, center_y, radius].

    margin:
        Additional deterministic clearance around every obstacle.

    eps:
        Small regularization inside the square root. This prevents an
        undefined derivative when the nominal state lies exactly at an
        obstacle center.

    Returns
    -------
    constraints:
        Function with signature constraints(x, u, t) returning one scalar
        inequality per obstacle.
    """
    obstacles = jnp.asarray(obstacles)

    centers = obstacles[:, :2]
    radii = obstacles[:, 2]

    def constraints(
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        del u, t

        displacement = x[:2][None, :] - centers

        # Smooth regularized distance:
        # sqrt(||p-c||^2 + eps^2)
        distance = jnp.sqrt(
            jnp.sum(displacement * displacement, axis=-1) + eps**2
        )

        return radii + margin - distance

    return constraints

# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Dimensions
    n = 4      # [px, py, theta, T]
    nu = 2     # [omega]

    # Horizon and dt
    N = 90

    # Weights: (x, y, theta, v, omega, T)
    W = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0], dtype=jnp.float32)

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.zeros((nu,), dtype=jnp.float32),
    )

    parameter = 1 / N

    v_max = 2.0
    om_max = 4.0
    u_min = jnp.array([-v_max, -om_max], dtype=jnp.float32)
    u_max = jnp.array([v_max, om_max], dtype=jnp.float32)

    constraints_u = make_control_box_constraints(u_min, u_max)

    x_min = jnp.array([-15.0, -15.0, -jnp.inf, 0.0], dtype=jnp.float32)
    x_max = jnp.array([15.0, 15.0, jnp.inf, 10.0], dtype=jnp.float32)
    constraints_x = make_state_box_constraints(x_min, x_max)
    terminal_center = jnp.array([0.5, 0.6])
    terminal_half_width = jnp.array([0.1, 0.1])
    term_constraint = make_terminal_set_constraint(
        alpha=terminal_center,
        beta=terminal_half_width,
        N=N,
    )
    obstacles = jnp.array([[0.0, 0.0, 0.3]], dtype=jnp.float32)
    constraints_obs = make_circular_obstacle_constraints(
        obstacles=obstacles,
        margin=0.0,
        eps=1e-6,
    )

    constraints_all = combine_constraints(constraints_x, constraints_u, term_constraint, constraints_obs)


    n_obs = obstacles.shape[0]
    nc = 2 * nu + 2 * n + n_obs + 4
    E_mag = 0.1
    disturbance = make_constant_disturbance(n=n, E_mag=E_mag)

    x0 = jnp.array([-0.75, -0.75, 0.0, 1.0], dtype=jnp.float32)
    x_goal = jnp.array([0.5, 0.6, 0.0, 1.0], dtype=jnp.float32)

    reference = make_straight_line_reference(
        x_start=x0,
        x_goal=x_goal,
        N=N,
    )
    X_ref = reference
    T_steps = N

    key = jax.random.PRNGKey(0)

    # -----------------------------
    # Update configs for robust run
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-3,
        rho_max=1e5,
        max_iterations=400,
        rho_update_frequency=25,
        initial_rho=1e-2,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=100,
        warm_start=True,
        rti=False,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=1e-5,
        line_search=True,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3)),
        cost=cost,
        num_constraints=nc,
        disturbance=disturbance,
        shift=1,
        X_in=X_ref,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float32),
    )
    E = jnp.identity(n) * E_mag
    E = E.at[-1, -1].set(0.0)
    # robust plan (single call in your script)
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    min_time = X_pred[0, -1]
    print("Computed Min Time:", min_time)
    # -----------------------------
    # Rollout simulations with early stopping
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float64)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

    dt = min_time / N

    for i in range(N_ROLLOUTS):
        disturbance_history = [jnp.zeros((n,), dtype=jnp.float32)]
        x = x0
        jax.debug.print(f"Rolling out iteration {i}")
        for k in range(T_steps):
            disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float32)
            for j in range(k + 1):
                disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

            u = U_pred[k] + disturbance_feedback

            key, x, w = dubins_step_with_disturbance(key, x, u, E, dt, i)

            disturbed[i, k, :2] = np.abs(np.asarray(X_pred[k + 1, :2] - x[:2]))
            disturbed[i, k, 2]  = np.abs(np.asarray(wrap_to_pi(X_pred[k + 1, 2] - x[2])))

            disturbance_history.append(w)
            xs[i, k] = np.asarray(x)

    plans_xy = []
    lowers_xy = []
    uppers_xy = []
    tube = get_trajectory_tubes(Phi_x)
    plan_xy = X_pred[:, :2]
    lower = plan_xy - tube[:, :2]
    upper = plan_xy + tube[:, :2]

    plans_xy.append(plan_xy)
    lowers_xy.append(lower)
    uppers_xy.append(upper)

    plot_rollouts_tubes_centers(
        xs=xs,
        centers=np.asarray(obstacles[:, :2]),
        radii=np.asarray(obstacles[:, 2]),
        plans_xy=np.asarray(plans_xy),
        lowers_xy=np.asarray(lowers_xy),
        uppers_xy=np.asarray(uppers_xy),
        goal_center=np.asarray(terminal_center),
        goal_half_width=np.asarray(terminal_half_width),
        step_idx=0,
        tube_stride=1,
        filename="rollouts_tubes_centers.png",
        show_plan=True,
        tube_alpha=0.1,
        margin=0.2,
        rollout_alpha=0.5,
    )
    plot_tube_graph(
        disturbed=disturbed,
        tube=tube,
        dt=X_pred[0, -1] / N
    )
    plot_controls(
        controls=np.asarray(U_pred),
        dt=dt,
        u_min=np.asarray(u_min),
        u_max=np.asarray(u_max),
        filename="dubins_controls.png",
    )


if __name__ == "__main__":
    main()