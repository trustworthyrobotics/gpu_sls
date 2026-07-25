from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from trajax.optimizers import linearize, quadratize, vectorize

from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    parallel_filter_line_search,
    merit_rho,
    slope,
)
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu


@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (self.max_sqp_iterations, self.feas_tol, self.step_tol, self.warm_start, self.line_search)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

@partial(jit, static_argnums=(0, 1))
def model_evaluator_helper_min_time(cost, dynamics,x0, X, U):
    T = U.shape[0]
    costs = jax.vmap(cost)(X, jnp.pad(U, [[0, 1], [0, 0]]), jnp.arange(T + 1))
    g = jnp.sum(costs)

    residual_fn = lambda t: dynamics(X[t], U[t], t) - X[t + 1]
    initial_residual = jnp.concatenate([
        x0[:-1] - X[0, :-1],
        jnp.zeros((1,), dtype=X.dtype),
    ])

    c = jnp.vstack([
        initial_residual,
        jax.vmap(residual_fn)(jnp.arange(T)),
    ])

    return g, c

def filter_model_evaluator_factory(model_evaluator, constraints, backoffs):
    # TODO: Might need to add the fact that we inflate obstalces by 1e-2
    def filter_model_evaluator(X, U):
        cost, c_dyn = model_evaluator(X, U)

        T = U.shape[0]
        U_pad = jnp.pad(U, ((0, 1), (0, 0)))
        t = jnp.arange(T + 1)

        g_base = vectorize(constraints)(X, U_pad, t)
        g_base_tight = g_base + backoffs

        g_all = g_base_tight
        g_viol = jnp.maximum(g_all, 0.0)

        c_filter = jnp.concatenate([
            c_dyn.reshape(-1),
            g_viol.reshape(-1),
        ])

        return cost, c_filter

    return filter_model_evaluator

def lagrangian(cost, dynamics, constraints, x0, obstacles, backoffs):
    def fun(x, u, t, v, v_prev, lam):
        c1 = cost(x, u, t)

        c2 = jnp.dot(v, dynamics(x, u, t))
        c3 = jnp.dot(v_prev, lax.select(t == 0, x0 - x, -x))

        g_base = constraints(x, u, t)
        n_base = g_base.shape[0]

        g_base_tight = g_base + backoffs[t, :n_base]

        if obstacles.shape[0] == 0:
            g_obs_tight = jnp.empty((0,), dtype=g_base.dtype)
        else:
            centers = obstacles[:, :2]
            radii = obstacles[:, 2]

            pos = x[:2]
            diff = pos[None, :] - centers
            dist = jnp.linalg.norm(diff, axis=-1) + 1e-6
            n = diff / dist[:, None]

            hx = jnp.abs(backoffs[t, 0])
            hy = jnp.abs(backoffs[t, 1])

            obs_backoff = jnp.abs(n[:, 0]) * hx + jnp.abs(n[:, 1]) * hy
            g_obs_tight = radii - dist + obs_backoff

        g_all = jnp.concatenate([g_base_tight, g_obs_tight], axis=0)

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
    x0, X, U, V, c,
    w, y, rho,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws,
    sqp_iteration
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
    E = disturbance(X)

    Q_bar = jnp.broadcast_to(jnp.eye(Q.shape[1]), Q.shape)
    R_bar = jnp.broadcast_to(jnp.eye(R.shape[1]), R.shape)

    n_obs = obstacles.shape[0]

    def run_nominal(_):
        dX, dU, dV, w1, y1, rho1, _, converged_admm = constrained_solve(
            admm_config, Q, q, R, r, M, A, B, c, C_all, D_all, f_all, w, y, rho
        )
        backoffs = jnp.zeros((T + 1, nc - n_obs))
        Phi_x   = jnp.zeros((T + 1, T + 1, nx, nx))
        Phi_u   = jnp.zeros((T, T + 1, nu, nx))
        betaN   = jnp.ones((T + 1, T + 1, nc - n_obs)) * 1e-10
        muN     = jnp.zeros((T + 1, nc))
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm

    def run_sls(_):
        dX, dU, dV, w1, y1, rho1, converged, converged_admm, backoffs, Phi_x, Phi_u, betaN, muN = sls_solve_gpu(
            admm_config,
            Q, q, R, r, M, A, B, c,
            C_all, D_all, f_all, w, y, rho, sls_config,
            E, Q_bar, R_bar, obstacles, X, h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws
        )
        return dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm

    use_nominal = jnp.logical_or(
        jnp.logical_not(sls_config.enable_fastsls),
        jnp.logical_and(sls_config.enable_fastsls, sqp_iteration < sls_config.max_initial_sqp_iterations)
    )
    dX, dU, dV, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm = lax.cond(
        use_nominal, run_nominal, run_sls, operand=None
    )

    return dX, dU, dV, q, r, w1, y1, rho1, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm


@partial(jit, static_argnums=(0,1,2,3,4,5,6,7))
def sqp(
    sls_config: SLSConfig, sqp_config: SQPConfig, admm_config: ADMMConfig,
    cost, dynamics, hessian_approx,
    constraints, disturbance,
    reference, parameter,
    W,
    x0, X_in, U_in, V_in,
    w, y, rho,
    obstacles,
    h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, converged_admm_prev
):
    _cost = partial(cost, W, reference)
    if hessian_approx is not None:
        _hessian_approx = partial(hessian_approx, W, reference)
    else:
        _hessian_approx = None

    _dynamics = partial(dynamics, parameter=parameter)
    model_evaluator = partial(model_evaluator_helper_min_time, _cost, _dynamics, x0)

    def body(i, carry):
        i, X_curr, U_curr, V_curr, w, y, rho, converged, backoffs, Phi_x, Phi_u, beta_ws, mu_w, converged_admm = carry

        def do_nothing(_):
            return carry

        def do_iter(_):
            g, c = model_evaluator(X_curr, U_curr)
            feas = jnp.max(jnp.abs(c))
            warm_flag = jnp.logical_and(jnp.array(bool(sqp_config.warm_start)), converged_admm)

            # Turn this off? seems to be more optimal
            w0   = lax.select(jnp.array(False), w, jnp.zeros_like(w))
            y0   = lax.select(warm_flag, y, jnp.zeros_like(y))
            rho0 = lax.select(warm_flag, rho, jnp.asarray(admm_config.initial_rho, dtype=rho.dtype))
            h_ct_ws = backoffs
            dX, dU, dV, q, r, w1, y1, rho1, backoffs1, Phi_x1, Phi_u1, betaN, muN, converged_admm_new = compute_search_direction(
                sls_config, admm_config,
                _cost, _dynamics, _hessian_approx,
                constraints, disturbance,
                obstacles,
                x0, X_curr, U_curr, V_curr, c,
                w0, y0, rho0,
                h_ct_ws, beta_ws, mu_ws, Phi_x_ws, Phi_u_ws, i
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
            # jax.debug.print("SQP Iteration {} Feas {} (<= {}) Step {} (<= {})", i, feas, sqp_config.feas_tol, step, sqp_config.step_tol)
            converged1 = jnp.logical_and(feas_ok, step_ok)
            filter_model_evaluator = filter_model_evaluator_factory(
                model_evaluator,
                constraints,
                backoffs1,
            )
            current_cost, current_c_filter = filter_model_evaluator(X_curr, U_curr)

            g, c = model_evaluator(X_curr, U_curr)

            rho_merit = merit_rho(c, dV)
            merit_fn  = merit_function_factory(rho_merit)
            current_merit = merit_fn(V_curr, g, c)
            merit_slope = slope(dX, dU, dV, c, q, r, rho_merit)
            last_iter = (i == (sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations - 1))
            do_ls = jnp.logical_and(jnp.array(bool(sqp_config.line_search)), jnp.logical_not(last_iter))
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

            X_next, U_next, V_next = lax.cond(do_ls, ls_branch, fullstep_branch, operand=None)

            w_next = lax.select(converged1, w, w1)
            y_next = lax.select(converged1, y, y1)
            rho_next = lax.select(converged1, rho, rho1)
            backoffs_next = lax.select(converged1, backoffs, backoffs1)
            Phi_x_next = lax.select(converged1, Phi_x, Phi_x1)
            Phi_u_next = lax.select(converged1, Phi_u, Phi_u1)

            return (i + 1, X_next, U_next, V_next, w_next, y_next, rho_next,
                    jnp.logical_or(converged, converged1),
                    backoffs_next, Phi_x_next, Phi_u_next, betaN, muN, converged_admm_new)

        return lax.cond(converged, do_nothing, do_iter, operand=None)

    backoffs0 = h_ct_ws
    carry0 = (0, X_in, U_in, V_in, w, y, rho, jnp.array(False), backoffs0, Phi_x_ws, Phi_u_ws, beta_ws, mu_ws, converged_admm_prev)
    total_iterations, X_out, U_out, V_out, w_out, y_out, rho_out, converged, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm = lax.fori_loop(
        0, sqp_config.max_sqp_iterations + sls_config.max_initial_sqp_iterations, body, carry0,
    )
    return X_out, U_out, V_out, w_out, y_out, rho_out, backoffs, Phi_x, Phi_u, betaN, muN, converged_admm
