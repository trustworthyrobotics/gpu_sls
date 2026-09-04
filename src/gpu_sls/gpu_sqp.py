from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from trajax.optimizers import linearize, quadratize, vectorize
from matplotlib.patches import Rectangle
from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    parallel_filter_line_search,
)
from gpu_sls.gpu_admm import (
    ADMMConfig,
    constrained_solve,
    slack_weight_for_iteration,
)
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu, tightening_from_nominal_state


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def save_sqp_xz_step_plot(
    sqp_iteration,
    X_curr,
    dX,
    X_next,
    output_dir="sqp_step_plots",
    x_idx=0,
    z_idx=2,
):
    """
    Plot the full SQP step X_curr + dX against the line-search accepted
    trajectory X_next = X_curr + alpha * dX.

    One image is saved per SQP iteration.

    Parameters
    ----------
    sqp_iteration:
        Scalar SQP iteration index.
    X_curr:
        Current nominal trajectory, shape (N + 1, nx).
    dX:
        Full SQP state search direction, shape (N + 1, nx).
    X_next:
        State trajectory accepted by the line search, shape (N + 1, nx).
    output_dir:
        Directory in which plots are saved.
    x_idx:
        State index corresponding to physical x-position.
    z_idx:
        State index corresponding to physical z-position.
    """
    iteration = int(np.asarray(sqp_iteration))

    X_curr_np = np.asarray(X_curr)
    dX_np = np.asarray(dX)
    X_next_np = np.asarray(X_next)

    X_full_np = X_curr_np + dX_np

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Current SQP nominal trajectory
    ax.plot(
        X_curr_np[:, x_idx],
        X_curr_np[:, z_idx],
        linewidth=1.5,
        linestyle=":",
        label=r"Current $X$",
    )

    # Unscaled/full SQP step: alpha = 1
    ax.plot(
        X_full_np[:, x_idx],
        X_full_np[:, z_idx],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=r"Full step $X + dX$",
    )

    # Line-search accepted trajectory
    ax.plot(
        X_next_np[:, x_idx],
        X_next_np[:, z_idx],
        marker="x",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        label=r"Accepted $X + \alpha dX$",
    )

    # Start and terminal markers
    ax.scatter(
        X_curr_np[0, x_idx],
        X_curr_np[0, z_idx],
        marker="s",
        s=70,
        label="Start",
        zorder=5,
    )

    ax.scatter(
        X_full_np[-1, x_idx],
        X_full_np[-1, z_idx],
        marker="*",
        s=100,
        label="Full-step terminal",
        zorder=5,
    )

    ax.scatter(
        X_next_np[-1, x_idx],
        X_next_np[-1, z_idx],
        marker="D",
        s=55,
        label="Accepted terminal",
        zorder=5,
    )

    # ---------------------------------------------------------
    # Hardcoded terminal set, projected onto the x-z plane
    # Full 3D terminal set:
    #   center     = [5.0, 0.1, 1.0]
    #   half-width = [0.6, 0.6, 0.2]
    # ---------------------------------------------------------
    terminal_x_center = 5.0
    terminal_z_center = 1.0

    terminal_x_half_width = 0.6
    terminal_z_half_width = 0.2

    terminal_x_min = terminal_x_center - terminal_x_half_width
    terminal_z_min = terminal_z_center - terminal_z_half_width

    terminal_patch = Rectangle(
        (terminal_x_min, terminal_z_min),
        width=2.0 * terminal_x_half_width,
        height=2.0 * terminal_z_half_width,
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
        linestyle="-.",
        label="Terminal set",
        zorder=2,
    )

    ax.add_patch(terminal_patch)

    # Optional marker at the terminal-set center
    ax.scatter(
        terminal_x_center,
        terminal_z_center,
        marker="+",
        s=100,
        linewidths=2.0,
        label="Terminal center",
        zorder=6,
    )

    ax.set_xlabel(f"State {x_idx}: x")
    ax.set_ylabel(f"State {z_idx}: z")
    ax.set_title(f"SQP iteration {iteration}: full vs accepted step")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()

    filename = output_path / f"sqp_iteration_{iteration:05d}_xz.png"
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True
    lm_regularization: float = 0.0

    def tree_flatten(self):
        children = (
            self.max_sqp_iterations,
            self.feas_tol,
            self.step_tol,
            self.warm_start,
            self.line_search,
            self.lm_regularization,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _slack_weight_for_sqp_iteration(
    admm_config: ADMMConfig,
    sls_config: SLSConfig,
    sqp_config: SQPConfig,
    sqp_iteration,
    dtype,
):
    """Schedule slack within nominal and robust SQP phases."""
    has_nominal_phase = (
        sls_config.enable_fastsls
        and sls_config.max_initial_sqp_iterations > 0
    )
    if has_nominal_phase:
        nominal_phase = sqp_iteration < sls_config.max_initial_sqp_iterations
        phase_iteration = lax.select(
            nominal_phase,
            sqp_iteration,
            sqp_iteration - sls_config.max_initial_sqp_iterations,
        )
        phase_iterations = lax.select(
            nominal_phase,
            jnp.asarray(
                sls_config.max_initial_sqp_iterations,
                dtype=sqp_iteration.dtype,
            ),
            jnp.asarray(
                sqp_config.max_sqp_iterations,
                dtype=sqp_iteration.dtype,
            ),
        )
    else:
        phase_iteration = sqp_iteration
        phase_iterations = jnp.asarray(
            (
                sqp_config.max_sqp_iterations
                + sls_config.max_initial_sqp_iterations
            ),
            dtype=sqp_iteration.dtype,
        )

    return slack_weight_for_iteration(
        admm_config,
        phase_iteration,
        phase_iterations,
        dtype=dtype,
    )

@partial(jit, static_argnums=(0, 1, 2))
def model_evaluator_helper_min_time(
    cost,
    dynamics,
    num_phases,
    x0,
    X,
    U,
):
    T = U.shape[0]

    costs = jax.vmap(cost)(
        X,
        jnp.pad(U, [[0, 1], [0, 0]]),
        jnp.arange(T + 1),
    )
    g = jnp.sum(costs)

    residual_fn = lambda t: (
        dynamics(X[t], U[t], t) - X[t + 1]
    )

    # Enforce the complete initial condition for fixed-time problems.  For a
    # minimum-time problem, only the final `num_phases` duration states are
    # free at node zero.  Handle zero explicitly because ``-0:`` is ``0:``.
    initial_residual = x0 - X[0]
    if num_phases > 0:
        initial_residual = initial_residual.at[-num_phases:].set(0.0)

    c = jnp.vstack([
        initial_residual,
        jax.vmap(residual_fn)(jnp.arange(T)),
    ])

    return g, c


def filter_model_evaluator_factory(
    model_evaluator,
    constraints,
    disturbance_fn,
    Phi_x_I: jnp.ndarray,
    Phi_u_I: jnp.ndarray,
    C_box: jnp.ndarray,
    D_box: jnp.ndarray,
    eps_abs: float,
    backoffs,
    L,
):
    """Build a line-search evaluator with state-dependent tightenings.

    For every candidate trajectory X_trial, this evaluator computes

        E_j = E(X_trial[j]),
        Phi_x[k, j] = Phi_x_I[k, j] @ E_j,
        Phi_u[k, j] = Phi_u_I[k, j] @ E_j,

    and then recomputes the robust constraint tightening.

    Phi_x_I and Phi_u_I remain fixed throughout the line search.
    """

    def filter_model_evaluator(
        X: jnp.ndarray,
        U: jnp.ndarray,
    ):
        cost, c_dyn = model_evaluator(X, U)

        T = U.shape[0]
        U_pad = jnp.pad(U, ((0, 1), (0, 0)))
        t = jnp.arange(T + 1)

        # Original nonlinear nominal constraints.
        g_base = vectorize(constraints)(X, U_pad, t)

        if L != 0:
            h_ct_trial = tightening_from_nominal_state(disturbance_fn, X, Phi_x_I, Phi_u_I, C_box, D_box)
        else:
            h_ct_trial = backoffs

        n_tight = h_ct_trial.shape[1]
        # eps_abs = 0
        g_base_tight = (
            g_base[:, :n_tight]
            + h_ct_trial
            + eps_abs
        )

        # Any remaining constraints are evaluated without this tightening.
        g_remaining = g_base[:, n_tight:]

        g_all = jnp.concatenate(
            [g_base_tight, g_remaining],
            axis=1,
        )
        g_viol = jnp.maximum(g_all, 0.0)

        # g_viol = jnp.maximum(g_base, 0.0)

        c_filter = jnp.concatenate(
            [
                c_dyn.reshape(-1),
                g_viol.reshape(-1),
            ]
        )

        return cost, c_filter

    return filter_model_evaluator

def lagrangian(cost, dynamics, constraints, x0, obstacles, backoffs):
    def fun(x, u, t, v, v_prev, lam):
        c1 = cost(x, u, t)

        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))

        g_base = constraints(x, u, t)
        g_all = g_base + backoffs[t]

        c4 = jnp.dot(lam, g_all)

        return c1 + c2 + c3 + c4

    return fun

@jax.jit
def add_obstacle_constraints(C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                             obstacles: jnp.ndarray, x_curr: jnp.ndarray, eps=1e-5):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _,  _, nu = D.shape

    centers = obstacles[:, :2]
    radii   = obstacles[:, 2]
    pos = x_curr[:, :2]
    diff = pos[:, None, :] - centers[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) + eps
    n = diff / dist[..., None]
    coeffs = -n

    C_obstacle = jnp.zeros((Tp1, centers.shape[0], nx), dtype=C.dtype)
    D_obstacle = jnp.zeros((Tp1, centers.shape[0], nu), dtype=D.dtype)

    C_obstacle = C_obstacle.at[..., 0:2].set(coeffs)

    f_obstacle = (dist - radii[None, :]).astype(f.dtype)

    C_all = jnp.concatenate([C, C_obstacle], axis=1)
    D_all = jnp.concatenate([D, D_obstacle], axis=1)
    f_all = jnp.concatenate([f, f_obstacle], axis=1)
    
    return C_all, D_all, f_all

def merit_function_factory(rho_merit):
    def merit_fn(V, g, c):
        return g + jnp.sum(V * c) + 0.5 * rho_merit * jnp.sum(c * c)
    return merit_fn

@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def compute_search_direction(
    sls_config: SLSConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    obstacles,
    lm_regularization,
    x0, X, U, V, c,
    w, y, rho, rho_grad, 
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
    Phi_x_I_ws, Phi_u_I_ws,
    a, b,
    sqp_iteration,
    slack_weight,
    Q_bar, R_bar,
):
    T = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]
    nc = w.shape[1]
    pad = lambda A: jnp.pad(A, [[0, 1], [0, 0]])

    if hessian_approx is None:
        quadratizer = quadratize(cost)
        Q, R_pad, M_pad = quadratizer(X, pad(U), jnp.arange(T + 1))
    else:
        Q, R_pad, M_pad = jax.vmap(hessian_approx)(X, pad(U), jnp.arange(T + 1))

    R = R_pad[:-1]
    M = M_pad[:-1]

    # Levenberg-Marquardt regularization of the SQP quadratic model.
    # Adding lambda * I to both diagonal Hessian blocks is equivalent to
    # regularizing the full stage Hessian in (x, u), while leaving the
    # state-control cross term unchanged.
    lm = jnp.maximum(
        jnp.asarray(lm_regularization, dtype=Q.dtype),
        jnp.asarray(jnp.finfo(Q.dtype).eps, dtype=Q.dtype),
    )
    # Autodiff can leave tiny asymmetric roundoff in the Hessian.  The LQR
    # recursion assumes symmetric blocks, so remove it before damping.
    Q = 0.5 * (Q + jnp.swapaxes(Q, -1, -2))
    R = 0.5 * (R + jnp.swapaxes(R, -1, -2))
    Q = Q + lm * jnp.eye(nx, dtype=Q.dtype)
    R = R + lm.astype(R.dtype) * jnp.eye(nu, dtype=R.dtype)

    linearizer = linearize(
        lagrangian(cost, dynamics, constraints, x0, obstacles, h_ct_ws),
        argnums=6,
    )
    q, r_pad = linearizer(
        X, pad(U), jnp.arange(T + 1),
        pad(V[1:]), V,
        y,
    )

    r = r_pad[:-1]
    dynamics_linearizer = linearize(dynamics)
    A_pad, B_pad = dynamics_linearizer(X, pad(U), jnp.arange(T + 1))
    A = A_pad[:-1]
    B = B_pad[:-1]

    pad = lambda A: jnp.pad(A, ((0, 1), (0, 0)))
    U_pad = pad(U)
    t = jnp.arange(X.shape[0])
    g = vectorize(constraints)(X, U_pad, t)
    f = -g
    C, D = linearize(constraints)(X, U_pad, t)
    C_all, D_all, f_all = add_obstacle_constraints(C, D, f, obstacles, X)
    terminal_control_rows = jnp.linalg.norm(D_all[-1], axis=-1) > 1e-7

    D_all = D_all.at[-1].set(jnp.zeros_like(D_all[-1]))

    f_all = f_all.at[-1].set(
        jnp.where(
            terminal_control_rows,
            jnp.asarray(1e6, dtype=f_all.dtype),
            f_all[-1],
        )
    )

    n_obs = obstacles.shape[0]

    def run_nominal(_):
        Jh = jnp.zeros((T + 1, nc, sls_config.gradient_window, nx))
        dX, dU, dV, w1, y1, rho1, rho_grad1, _, a1, b1, converged_admm = constrained_solve(
            admm_config, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho, rho_grad, Jh, a, b, sls_config.gradient_window, just_nominal=True,
            slack_weight=slack_weight,
        )
        backoffs = jnp.zeros((T + 1, nc - n_obs))
        Phi_x   = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u   = jnp.zeros((T, T + 1, nu, nx))
        betaN   = jnp.ones((T + 1, T + 1, nc - n_obs)) * 1e-10
        muN     = jnp.zeros((T + 1, nc))
        return dX, dU, dV, w1, y1, rho1, rho_grad1, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x, Phi_u, a1, b1, converged_admm

    def run_sls(_):
        dX, dU, dV, w1, y1, rho1, rho_grad1, converged, converged_admm, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a1, b1,  = sls_solve_gpu(
            admm_config, sls_config, disturbance,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, rho_grad,
            Q_bar, R_bar, obstacles, X, h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, Phi_x_I_ws, Phi_u_I_ws, a, b,
            slack_weight,
        )
        return dX, dU, dV, w1, y1, rho1, rho_grad1, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a1, b1, converged_admm

    use_nominal = jnp.logical_or(
        jnp.logical_not(sls_config.enable_fastsls),
        jnp.logical_and(sls_config.enable_fastsls, sqp_iteration < sls_config.max_initial_sqp_iterations)
    )
    dX, dU, dV, w1, y1, rho1, rho_grad1, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a1, b1, converged_admm = lax.cond(
        use_nominal, run_nominal, run_sls, operand=None
    )

    return dX, dU, dV, q, r, w1, y1, rho1, rho_grad1, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a1, b1, converged_admm


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7))
def sqp(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    reference, parameter,
    W,
    x0, X_in, U_in, V_in,
    w, y, rho, rho_grad,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, Phi_x_I_ws, Phi_u_I_ws, 
    a, b, converged_admm_prev,
    Q_bar=None, R_bar=None,
):
    if Q_bar is None:
        Q_bar = jnp.broadcast_to(
            jnp.eye(X_in.shape[-1], dtype=X_in.dtype),
            (X_in.shape[0], X_in.shape[-1], X_in.shape[-1]),
        )
    if R_bar is None:
        R_bar = jnp.broadcast_to(
            jnp.eye(U_in.shape[-1], dtype=U_in.dtype),
            (U_in.shape[0], U_in.shape[-1], U_in.shape[-1]),
        )

    _cost = partial(cost, W, reference, parameter=parameter)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper_min_time, _cost, _dynamics, admm_config.num_phases, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, rho_grad0, converged, backoffs, Phi_x, Phi_u, beta_ws, mu_w, Phi_x_I_ws, Phi_u_I_ws, a, b, converged_admm = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            # converged_admm = True
            warm_flag = jnp.logical_and(jnp.array(bool(sqp_config.warm_start)), converged_admm)
            # warm_flag = jnp.logical_and(warm_flag, jnp.array(i != sls_config.max_initial_sqp_iterations))
            # Turn this off? seems to be more optimal
            w0   = lax.select(jnp.array(False), w, jnp.zeros_like(w))
            w0   = lax.select(warm_flag, w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            a0 = lax.select(jnp.array(False), a, jnp.zeros_like(a))
            # b0 = lax.select(jnp.array(False), b, jnp.zeros_like(b))
            a0 = lax.select(warm_flag, a, jnp.zeros_like(a))
            b0 = lax.select(warm_flag, b, jnp.zeros_like(b))
            rho0 = lax.select(warm_flag, rho, jnp.asarray(admm_config.initial_rho, dtype=rho.dtype))
            h_ct_ws = backoffs
            slack_weight = _slack_weight_for_sqp_iteration(
                admm_config,
                sls_config,
                sqp_config,
                i,
                X_curr.dtype,
            )
            if admm_config.enable_slack and admm_config.dynamic_slack:
                jax.debug.print(
                    "Slack schedule: SQP iteration {} weight={:.3e}",
                    i + 1,
                    slack_weight,
                )
            dX, dU, dV, q, r, w1, y1, rho1, rho_grad1, backoffs1, Phi_x1, Phi_u1, betaN, muN, Phi_x_I_next, Phi_u_I_next, a1, b1, converged_admm_new = compute_search_direction(
                sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                constraints, disturbance,
                obstacles,
                sqp_config.lm_regularization,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0, rho_grad0,
                h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, Phi_x_I_ws, Phi_u_I_ws, a0, b0, i,
                slack_weight,
                Q_bar, R_bar,
            )

            direction_finite = (
                jnp.all(jnp.isfinite(dX))
                & jnp.all(jnp.isfinite(dU))
                & jnp.all(jnp.isfinite(dV))
                & jnp.all(jnp.isfinite(q))
                & jnp.all(jnp.isfinite(r))
                & jnp.all(jnp.isfinite(w1))
                & jnp.all(jnp.isfinite(y1))
                & jnp.all(jnp.isfinite(rho1))
                & jnp.all(jnp.isfinite(rho_grad1))
                & jnp.all(jnp.isfinite(backoffs1))
                & jnp.all(jnp.isfinite(Phi_x1))
                & jnp.all(jnp.isfinite(Phi_u1))
                & jnp.all(jnp.isfinite(betaN))
                & jnp.all(jnp.isfinite(muN))
                & jnp.all(jnp.isfinite(Phi_x_I_next))
                & jnp.all(jnp.isfinite(Phi_u_I_next))
                & jnp.all(jnp.isfinite(a1))
                & jnp.all(jnp.isfinite(b1))
            )
            step = jnp.maximum(
                jnp.max(jnp.abs(dX)),
                jnp.max(jnp.abs(dU))
            )
            z_norm = jnp.maximum(
                jnp.max(jnp.abs(X_curr)),
                jnp.max(jnp.abs(U_curr))
            )

            feas_ok = feas <= sqp_config.feas_tol
            step_ok = step <= sqp_config.step_tol * (1.0 + z_norm)
            jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            T = U_curr.shape[0]
            U_curr_pad = jnp.pad(U_curr, ((0, 1), (0, 0)))
            t_curr = jnp.arange(T + 1)

            C_all, D_all = linearize(constraints)(
                X_curr,
                U_curr_pad,
                t_curr,
            )

            D_all = D_all.at[-1].set(jnp.zeros_like(D_all[-1]))

            converged1 = jnp.logical_and(
                direction_finite, jnp.logical_and(feas_ok, step_ok)
            )
            # Convergence of the nominal warm-up is not convergence of the
            # robust problem.  Stopping here skips the first SLS iteration and
            # returns the nominal branch's zero Phi_x/Phi_u placeholders.
            # Keep advancing until at least one robust SQP iteration has run.
            nominal_warmup_iteration = jnp.logical_and(
                jnp.asarray(bool(sls_config.enable_fastsls)),
                i < sls_config.max_initial_sqp_iterations,
            )
            robust_converged1 = jnp.logical_and(
                converged1, jnp.logical_not(nominal_warmup_iteration)
            )
            filter_model_evaluator = filter_model_evaluator_factory(
                model_evaluator=model_evaluator,
                constraints=constraints,
                disturbance_fn=disturbance,
                Phi_x_I=Phi_x_I_next,
                Phi_u_I=Phi_u_I_next,
                C_box=C_all,
                D_box=D_all,
                backoffs=backoffs1,
                eps_abs=admm_config.eps_abs,
                L=sls_config.gradient_window,
            )
            current_cost, current_c_filter = filter_model_evaluator(X_curr, U_curr)

            # last_iter = (i == (sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations - 1))
            # do_ls = jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter))
            do_ls = jnp.array(bool(sqp_config.line_search))
            def ls_branch(_):
                Xn, Un, Vn = parallel_filter_line_search(
                    filter_model_evaluator, X_curr, U_curr, V_curr, dX, dU, dV,
                    current_cost, current_c_filter,
                    q, r,
                    alpha_min=1e-4, theta_max=1e-2, theta_min=1e-6, eta=1e-4,
                    gamma_phi=1e-6, gamma_theta=1e-6, gamma_alpha=0.5,
                )
                return Xn, Un, Vn

            def fullstep_branch(_):
                return (X_curr + dX, U_curr + dU, V_curr + dV)

            X_next, U_next, V_next = lax.cond(
                do_ls, ls_branch, fullstep_branch, operand=None
            )
            candidate_finite = (
                jnp.all(jnp.isfinite(X_next))
                & jnp.all(jnp.isfinite(U_next))
                & jnp.all(jnp.isfinite(V_next))
            )
            failed = jnp.logical_not(
                jnp.logical_and(direction_finite, candidate_finite)
            )
            X_next = lax.select(failed, X_curr, X_next)
            U_next = lax.select(failed, U_curr, U_next)
            V_next = lax.select(failed, V_curr, V_next)

            # jax.debug.callback(
            #     save_sqp_xz_step_plot,
            #     i,
            #     X_curr,
            #     dX,
            #     X_next,
            #     ordered=True,
            # )

            keep_previous = jnp.logical_or(robust_converged1, failed)
            w_next = lax.select(keep_previous, w, w1)
            y_next = lax.select(keep_previous, y, y1)
            a_next = lax.select(keep_previous, a, a1)
            b_next = lax.select(keep_previous, b, b1)
            rho_next = lax.select(keep_previous, rho, rho1)
            rho_grad_next = lax.select(keep_previous, rho_grad, rho_grad1)
            backoffs_next = lax.select(keep_previous, backoffs, backoffs1)
            Phi_x_next = lax.select(keep_previous, Phi_x, Phi_x1)
            Phi_u_next = lax.select(keep_previous, Phi_u, Phi_u1)
            beta_next = lax.select(keep_previous, beta_ws, betaN)
            mu_next = lax.select(keep_previous, mu_w, muN)
            Phi_x_I_next = lax.select(
                keep_previous, Phi_x_I_ws, Phi_x_I_next
            )
            Phi_u_I_next = lax.select(
                keep_previous, Phi_u_I_ws, Phi_u_I_next
            )
            converged_admm_next = lax.select(
                failed, converged_admm, converged_admm_new
            )

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next, rho_grad_next,
                    jnp.logical_or(converged, jnp.logical_or(robust_converged1, failed)),
                    backoffs_next, Phi_x_next, Phi_u_next, beta_next, mu_next, Phi_x_I_next, Phi_u_I_next, a_next, b_next, converged_admm_next)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    carry0 = (0, X_in, U_in, V_in, w, y, rho, rho_grad, jnp.array(False), backoffs0, Phi_x_ws, Phi_u_ws, beta_ws, mu_ws, Phi_x_I_ws, Phi_u_I_ws, a, b, converged_admm_prev)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, rho_grad_out, converged, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a_out, b_out, converged_admm = lax.fori_loop(
        0, sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations, body, carry0,
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, rho_grad_out, backoffs, Phi_x, Phi_u, betaN, muN, Phi_x_I, Phi_u_I, a_out, b_out, converged_admm
