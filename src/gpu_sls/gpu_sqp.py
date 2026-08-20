from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from trajax.optimizers import linearize, quadratize, vectorize

from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.optimizers import (
    merit_rho,
)
from gpu_sls.gpu_admm import ADMMConfig, constrained_solve
from gpu_sls.gpu_sls import SLSConfig, sls_solve_gpu
from gpu_sls.gpu_tube_mpc import tube_mpc


# ============================================================
# SQP Configuration
# ============================================================

@register_pytree_node_class
@dataclass(frozen=True)
class SQPConfig:
    max_sqp_iterations: int = 1
    feas_tol: float = 1e-2
    step_tol: float = 1e-4
    warm_start: bool = True
    line_search: bool = True

    def tree_flatten(self):
        children = (
            self.max_sqp_iterations,
            self.feas_tol,
            self.step_tol,
            self.warm_start,
            self.line_search,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ============================================================
# Lagrangian
# ============================================================

def lagrangian(
    cost,
    dynamics,
    constraints,
    x0,
    obstacles,
    backoffs,
):
    def fun(
        x,
        u,
        t,
        v,
        v_prev,
        lam,
    ):
        # ----------------------------------------------------
        # Objective
        # ----------------------------------------------------

        c1 = cost(x, u, t)

        # ----------------------------------------------------
        # Dynamics terms
        # ----------------------------------------------------

        c2 = jnp.dot(
            v,
            dynamics(x, u, t),
        )

        c3 = jnp.dot(
            v_prev,
            lax.select(
                t == 0,
                x0 - x,
                -x,
            ),
        )

        # ----------------------------------------------------
        # Base inequality constraints
        #
        # Convention:
        #
        #     g(x,u) <= 0
        #
        # Tightened:
        #
        #     g(x,u) + backoff <= 0
        # ----------------------------------------------------

        g_base = constraints(
            x,
            u,
            t,
        )

        n_base = g_base.shape[0]

        g_base_tight = (
            g_base
        )

        # ----------------------------------------------------
        # Obstacle constraints
        # ----------------------------------------------------

        if obstacles.shape[0] == 0:

            g_obs_tight = jnp.empty(
                (0,),
                dtype=g_base.dtype,
            )

        else:

            centers = obstacles[:, :2]
            radii = obstacles[:, 2]

            pos = x[:2]

            diff = (
                pos[None, :]
                - centers
            )

            dist = (
                jnp.linalg.norm(
                    diff,
                    axis=-1,
                )
                + 1e-6
            )

            n = (
                diff
                / dist[:, None]
            )

            # Existing backoff convention:
            #
            # backoffs[:,0] → x-direction bound
            # backoffs[:,1] → y-direction bound

            hx = jnp.abs(
                backoffs[t, 0]
            )

            hy = jnp.abs(
                backoffs[t, 1]
            )

            obs_backoff = (
                jnp.abs(n[:, 0]) * hx
                + jnp.abs(n[:, 1]) * hy
            )

            # Outside obstacle:
            #
            #     radius - distance <= 0

            g_obs_tight = (
                radii
                - dist
                + obs_backoff
            )

        g_all = jnp.concatenate(
            [
                g_base_tight,
                g_obs_tight,
            ],
            axis=0,
        )

        # ----------------------------------------------------
        # Inequality multiplier contribution
        # ----------------------------------------------------

        c4 = jnp.dot(
            lam,
            g_all,
        )

        return (
            c1
            + c2
            + c3
            + c4
        )

    return fun


# ============================================================
# Obstacle Linearization
# ============================================================

@jax.jit
def add_obstacle_constraints(
    C: jnp.ndarray,
    D: jnp.ndarray,
    f: jnp.ndarray,
    obstacles: jnp.ndarray,
    x_curr: jnp.ndarray,
    eps=1e-5,
):
    if obstacles.shape[0] == 0:
        return C, D, f

    Tp1, _, nx = C.shape
    _, _, nu = D.shape

    centers = obstacles[:, :2]
    radii = obstacles[:, 2]

    pos = x_curr[:, :2]

    diff = (
        pos[:, None, :]
        - centers[None, :, :]
    )

    dist = (
        jnp.linalg.norm(
            diff,
            axis=-1,
        )
        + eps
    )

    n = (
        diff
        / dist[..., None]
    )

    coeffs = -n

    C_obstacle = jnp.zeros(
        (
            Tp1,
            centers.shape[0],
            nx,
        ),
        dtype=C.dtype,
    )

    D_obstacle = jnp.zeros(
        (
            Tp1,
            centers.shape[0],
            nu,
        ),
        dtype=D.dtype,
    )

    C_obstacle = (
        C_obstacle
        .at[..., 0:2]
        .set(coeffs)
    )

    f_obstacle = (
        dist
        - radii[None, :]
    ).astype(f.dtype)

    C_all = jnp.concatenate(
        [
            C,
            C_obstacle,
        ],
        axis=1,
    )

    D_all = jnp.concatenate(
        [
            D,
            D_obstacle,
        ],
        axis=1,
    )

    f_all = jnp.concatenate(
        [
            f,
            f_obstacle,
        ],
        axis=1,
    )

    return (
        C_all,
        D_all,
        f_all,
    )


# ============================================================
# Nonlinear Inequality Evaluator
# ============================================================

def inequality_evaluator_factory(
    constraints,
    obstacles,
):
    """
    Evaluate the true nonlinear inequality constraints.

    Convention:

        h(X,U) <= 0

    is feasible.

    This is used by the line search so that trial SQP steps are
    checked against the nonlinear constraints rather than only
    the linearized QP constraints.
    """

    def evaluator(
        X,
        U,
        backoffs,
    ):
        T = U.shape[0]

        U_pad = jnp.pad(
            U,
            (
                (0, 1),
                (0, 0),
            ),
        )

        def stage_constraint(
            x,
            u,
            t,
        ):
            # ------------------------------------------------
            # Base nonlinear constraints
            # ------------------------------------------------

            g_base = constraints(
                x,
                u,
                t,
            )

            n_base = g_base.shape[0]

            g_base_tight = (
                g_base
                # + backoffs[t, :n_base]
            )

            # ------------------------------------------------
            # Nonlinear obstacle constraints
            # ------------------------------------------------

            if obstacles.shape[0] == 0:

                g_obs = jnp.empty(
                    (0,),
                    dtype=g_base.dtype,
                )

            else:

                centers = obstacles[:, :2]
                radii = obstacles[:, 2]

                pos = x[:2]

                diff = (
                    pos[None, :]
                    - centers
                )

                dist = (
                    jnp.linalg.norm(
                        diff,
                        axis=-1,
                    )
                    + 1e-6
                )

                n = (
                    diff
                    / dist[:, None]
                )

                hx = jnp.abs(
                    backoffs[t, 0]
                )

                hy = jnp.abs(
                    backoffs[t, 1]
                )

                obs_backoff = (
                    jnp.abs(n[:, 0]) * hx
                    + jnp.abs(n[:, 1]) * hy
                )

                # Feasible when:
                #
                #     radius - distance + tightening <= 0

                g_obs = (
                    radii
                    - dist
                    + obs_backoff
                )

            return jnp.concatenate(
                [
                    g_base_tight,
                    g_obs,
                ],
                axis=0,
            )

        h = jax.vmap(
            stage_constraint,
        )(
            X,
            U_pad,
            jnp.arange(T + 1),
        )

        return h

    return evaluator


# ============================================================
# Constraint-Aware Merit Function
# ============================================================

def constrained_merit(
    V,
    objective,
    equality_residual,
    inequality_residual,
    rho_eq,
    rho_ineq,
):
    """
    Merit:

        phi =
            objective
            + V^T c
            + 0.5 rho_eq ||c||^2
            + 0.5 rho_ineq ||max(h, 0)||^2

    where

        c = equality / dynamics residual
        h = nonlinear inequality residual

    and h <= 0 is feasible.
    """

    equality_merit = (
        objective
        + jnp.sum(
            V
            * equality_residual
        )
        + 0.5
        * rho_eq
        * jnp.sum(
            equality_residual
            * equality_residual
        )
    )

    inequality_violation = jnp.maximum(
        inequality_residual,
        0.0,
    )

    inequality_merit = (
        0.5
        * rho_ineq
        * jnp.sum(
            inequality_violation
            * inequality_violation
        )
    )

    return (
        equality_merit
        + inequality_merit
    )


# ============================================================
# Constraint-Aware Line Search
# ============================================================

def constrained_line_search(
    model_evaluator,
    inequality_evaluator,
    X,
    U,
    V,
    dX,
    dU,
    dV,
    backoffs,
    rho_eq,
    rho_ineq=100.0,
    armijo_factor=1e-4,
    alpha_0=1.0,
    alpha_mult=0.5,
    alpha_min=1e-6,
):
    """
    Backtracking line search which checks:

        1. Armijo decrease of a merit function including
           nonlinear inequality constraint violations.

        2. Maximum nonlinear inequality violation does not
           increase.

    If no acceptable step is found, alpha = 0.
    """

    dtype = X.dtype

    zero = jnp.asarray(
        0.0,
        dtype=dtype,
    )

    one = jnp.asarray(
        1.0,
        dtype=dtype,
    )

    alpha0 = jnp.asarray(
        alpha_0,
        dtype=dtype,
    )

    alpha_min_jax = jnp.asarray(
        alpha_min,
        dtype=dtype,
    )

    # --------------------------------------------------------
    # Merit along line
    #
    # z(alpha) = z + alpha dz
    # --------------------------------------------------------

    def merit_at_alpha(alpha):

        X_trial = (
            X
            + alpha * dX
        )

        U_trial = (
            U
            + alpha * dU
        )

        V_trial = (
            V
            + alpha * dV
        )

        objective_trial, c_trial = (
            model_evaluator(
                X_trial,
                U_trial,
            )
        )

        h_trial = inequality_evaluator(
            X_trial,
            U_trial,
            backoffs,
        )

        phi = constrained_merit(
            V_trial,
            objective_trial,
            c_trial,
            h_trial,
            rho_eq,
            rho_ineq,
        )

        return phi

    # --------------------------------------------------------
    # Current merit and directional derivative
    #
    # phi'(0)
    # --------------------------------------------------------

    phi0, dphi0 = jax.jvp(
        merit_at_alpha,
        (zero,),
        (one,),
    )

    # --------------------------------------------------------
    # Current nonlinear inequality violation
    # --------------------------------------------------------

    h_current = inequality_evaluator(
        X,
        U,
        backoffs,
    )

    current_violation = jnp.max(
        jnp.maximum(
            h_current,
            0.0,
        )
    )

    # --------------------------------------------------------
    # Acceptance test
    # --------------------------------------------------------

    def acceptable(alpha):

        phi_trial = merit_at_alpha(
            alpha
        )

        armijo_rhs = (
            phi0
            + armijo_factor
            * alpha
            * dphi0
        )

        armijo_ok = (
            phi_trial
            <= armijo_rhs
        )

        X_trial = (
            X
            + alpha * dX
        )

        U_trial = (
            U
            + alpha * dU
        )

        h_trial = inequality_evaluator(
            X_trial,
            U_trial,
            backoffs,
        )

        trial_violation = jnp.max(
            jnp.maximum(
                h_trial,
                0.0,
            )
        )

        # Do not allow nonlinear constraint violation
        # to increase during line search.

        constraint_ok = (
            trial_violation
            <= current_violation + 1e-6
        )

        return (
            armijo_ok
            & constraint_ok
        )

    # --------------------------------------------------------
    # Try full step first
    # --------------------------------------------------------

    accepted0 = acceptable(
        alpha0
    )

    # --------------------------------------------------------
    # Backtracking
    # --------------------------------------------------------

    def cond_fn(carry):

        alpha, accepted = carry

        return jnp.logical_and(
            jnp.logical_not(
                accepted
            ),
            alpha > alpha_min_jax,
        )

    def body_fn(carry):

        alpha, _ = carry

        alpha_new = (
            alpha
            * alpha_mult
        )

        accepted_new = acceptable(
            alpha_new
        )

        return (
            alpha_new,
            accepted_new,
        )

    alpha, accepted = lax.while_loop(
        cond_fn,
        body_fn,
        (
            alpha0,
            accepted0,
        ),
    )

    # --------------------------------------------------------
    # If no acceptable step was found, do nothing.
    # --------------------------------------------------------

    alpha = jnp.where(
        accepted,
        alpha,
        zero,
    )

    X_new = (
        X
        + alpha * dX
    )

    U_new = (
        U
        + alpha * dU
    )

    V_new = (
        V
        + alpha * dV
    )

    return (
        X_new,
        U_new,
        V_new,
        alpha,
        accepted,
    )


# ============================================================
# Search Direction
# ============================================================

@partial(
    jit,
    static_argnums=(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ),
)
def compute_search_direction(
    sls_config: SLSConfig,
    admm_config: ADMMConfig,
    cost,
    dynamics,
    hessian_approx,
    constraints,
    disturbance,
    obstacles,
    x0,
    X,
    U,
    V,
    c,
    w,
    y,
    rho,
    h_ct_ws,
    beta_ws,
    mu_ws,
    Phi_x_ws,
    Phi_u_ws,
    K_fixed,
    sqp_iteration,
):
    T = U.shape[0]

    # --------------------------------------------------------
    # Cost Hessian
    # --------------------------------------------------------

    pad_u = lambda A: jnp.pad(
        A,
        [
            [0, 1],
            [0, 0],
        ],
    )

    if hessian_approx is None:

        quadratizer = quadratize(
            cost
        )

        Q, R_pad, M_pad = quadratizer(
            X,
            pad_u(U),
            jnp.arange(T + 1),
        )

    else:

        Q, R_pad, M_pad = jax.vmap(
            hessian_approx
        )(
            X,
            pad_u(U),
            jnp.arange(T + 1),
        )

    R = R_pad[:-1]
    M = M_pad[:-1]

    # --------------------------------------------------------
    # Gradient of Lagrangian
    # --------------------------------------------------------

    linearizer = linearize(
        lagrangian(
            cost,
            dynamics,
            constraints,
            x0,
            obstacles,
            h_ct_ws,
        ),
        argnums=6,
    )

    q, r_pad = linearizer(
        X,
        pad_u(U),
        jnp.arange(T + 1),
        pad_u(V[1:]),
        V,
        y,
    )

    r = r_pad[:-1]

    # --------------------------------------------------------
    # Dynamics linearization
    # --------------------------------------------------------

    dynamics_linearizer = linearize(
        dynamics
    )

    A_pad, B_pad = dynamics_linearizer(
        X,
        pad_u(U),
        jnp.arange(T + 1),
    )

    A = A_pad[:-1]
    B = B_pad[:-1]

    # --------------------------------------------------------
    # Constraint linearization
    # --------------------------------------------------------

    U_pad = jnp.pad(
        U,
        (
            (0, 1),
            (0, 0),
        ),
    )

    t = jnp.arange(
        X.shape[0]
    )

    g = vectorize(
        constraints
    )(
        X,
        U_pad,
        t,
    )

    # Linearized form:
    #
    #     C dx + D du <= f
    #
    # where f = -g

    f = -g

    C, D = linearize(
        constraints
    )(
        X,
        U_pad,
        t,
    )

    # --------------------------------------------------------
    # Add linearized obstacle constraints
    # --------------------------------------------------------

    C_all, D_all, f_all = add_obstacle_constraints(
        C,
        D,
        f,
        obstacles,
        X,
    )

    # --------------------------------------------------------
    # Disturbance linearization
    # --------------------------------------------------------

    E = disturbance(
        X
    )

    # --------------------------------------------------------
    # SLS weighting matrices
    # --------------------------------------------------------

    Q_bar = jnp.broadcast_to(
        jnp.eye(
            Q.shape[1],
            dtype=Q.dtype,
        ),
        Q.shape,
    )

    R_bar = jnp.broadcast_to(
        jnp.eye(
            R.shape[1],
            dtype=R.dtype,
        ),
        R.shape,
    )

    # ========================================================
    # Nominal QP
    # ========================================================

    def run_nominal(_):

        (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            _,
            K_term,
            converged_admm,
        ) = constrained_solve(
            admm_config,
            Q,
            q,
            R,
            r,
            M,
            A,
            B,
            c,
            C_all,
            D_all,
            f_all,
            w,
            y,
            rho,
        )

        return (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            h_ct_ws,
            Phi_x_ws,
            Phi_u_ws,
            beta_ws,
            mu_ws,
            K_term,
        )

    # ========================================================
    # SLS Robust QP
    # ========================================================

    def run_sls(_):

        (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            converged,
            converged_admm,
            backoffs,
            Phi_x,
            Phi_u,
            betaN,
            muN,
        ) = sls_solve_gpu(
            admm_config,
            Q,
            q,
            R,
            r,
            M,
            A,
            B,
            c,
            C_all,
            D_all,
            f_all,
            w,
            y,
            rho,
            sls_config,
            E,
            Q_bar,
            R_bar,
            obstacles,
            X,
            h_ct_ws,
            beta_ws,
            mu_ws,
            Phi_x_ws,
            Phi_u_ws,
        )

        return (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            backoffs,
            Phi_x,
            Phi_u,
            betaN,
            muN,
            K_fixed,
        )

    # ========================================================
    # Tube MPC Robust QP
    # ========================================================

    def run_tube_mpc(_):

        (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            _,
            backoffs,
        ) = tube_mpc(
            admm_config,
            Q,
            q,
            R,
            r,
            M,
            A,
            B,
            c,
            C_all,
            D_all,
            f_all,
            w,
            y,
            rho,
            K_fixed,
            E,
        )

        return (
            dX,
            dU,
            dV,
            w1,
            y1,
            rho1,
            backoffs,
            Phi_x_ws,
            Phi_u_ws,
            beta_ws,
            mu_ws,
            K_fixed,
        )

    # ========================================================
    # Solver selection
    # ========================================================

    robust_enabled = jnp.logical_or(
        sls_config.enable_fastsls,
        sls_config.enable_tube_mpc,
    )

    initial_nominal_phase = (
        sqp_iteration
        < sls_config.max_initial_sqp_iterations
    )

    use_nominal = jnp.logical_or(
        jnp.logical_not(
            robust_enabled
        ),
        jnp.logical_and(
            robust_enabled,
            initial_nominal_phase,
        ),
    )

    def run_robust(_):

        return lax.cond(
            sls_config.enable_fastsls,
            run_sls,
            run_tube_mpc,
            operand=None,
        )

    (
        dX,
        dU,
        dV,
        w1,
        y1,
        rho1,
        backoffs,
        Phi_x,
        Phi_u,
        betaN,
        muN,
        K_term,
    ) = lax.cond(
        use_nominal,
        run_nominal,
        run_robust,
        operand=None,
    )

    return (
        dX,
        dU,
        dV,
        q,
        r,
        w1,
        y1,
        rho1,
        backoffs,
        Phi_x,
        Phi_u,
        betaN,
        muN,
        K_term,
    )


# ============================================================
# Nonlinear Model Evaluator
# ============================================================

@partial(
    jit,
    static_argnums=(
        0,
        1,
        2,
    ),
)
def model_evaluator_helper_min_time(
    cost,
    dynamics,
    num_phases,
    x0,
    X,
    U,
):
    T = U.shape[0]

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    costs = jax.vmap(
        cost
    )(
        X,
        jnp.pad(
            U,
            [
                [0, 1],
                [0, 0],
            ],
        ),
        jnp.arange(T + 1),
    )

    g = jnp.sum(
        costs
    )

    # --------------------------------------------------------
    # Dynamics residuals
    # --------------------------------------------------------

    residual_fn = lambda t: (
        dynamics(
            X[t],
            U[t],
            t,
        )
        - X[t + 1]
    )

    # --------------------------------------------------------
    # Initial condition residual
    # --------------------------------------------------------
    #
    # For minimum-time problems, the final num_phases states
    # are duration variables and are allowed to move.
    # --------------------------------------------------------

    initial_residual = (
        x0
        - X[0]
    )

    if num_phases > 0:

        initial_residual = (
            initial_residual
            .at[-num_phases:]
            .set(0.0)
        )

    c = jnp.vstack(
        [
            initial_residual,
            jax.vmap(
                residual_fn
            )(
                jnp.arange(T)
            ),
        ]
    )

    return (
        g,
        c,
    )


# ============================================================
# SQP Solver
# ============================================================

@partial(
    jit,
    static_argnums=(
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ),
)
def sqp(
    sls_config: SLSConfig,
    sqp_config: SQPConfig,
    admm_config: ADMMConfig,
    cost,
    dynamics,
    hessian_approx,
    constraints,
    disturbance,
    reference,
    parameter,
    W,
    x0,
    X_in,
    U_in,
    V_in,
    w,
    y,
    rho,
    obstacles,
    h_ct_ws,
    beta_ws,
    mu_ws,
    Phi_x_ws,
    Phi_u_ws,
    K_fixed,
):
    # --------------------------------------------------------
    # Bind problem parameters
    # --------------------------------------------------------

    _cost = partial(
        cost,
        W,
        reference,
    )

    if hessian_approx is not None:

        _hessian_approx = partial(
            hessian_approx,
            W,
            reference,
        )

    else:

        _hessian_approx = None

    _dynamics = partial(
        dynamics,
        parameter=parameter,
    )

    # --------------------------------------------------------
    # Nonlinear model evaluator
    #
    # Returns:
    #
    #     objective
    #     dynamics / equality residuals
    # --------------------------------------------------------

    model_evaluator = partial(
        model_evaluator_helper_min_time,
        _cost,
        _dynamics,
        admm_config.num_phases,
        x0,
    )

    # --------------------------------------------------------
    # Nonlinear inequality evaluator
    #
    # Used specifically by the line search.
    # --------------------------------------------------------

    inequality_evaluator = (
        inequality_evaluator_factory(
            constraints,
            obstacles,
        )
    )

    # ========================================================
    # SQP iteration
    # ========================================================

    def body(
        loop_index,
        carry,
    ):
        (
            i,
            X_curr,
            U_curr,
            V_curr,
            w,
            y,
            rho,
            converged,
            backoffs,
            Phi_x,
            Phi_u,
            beta_ws,
            mu_ws,
            K_fixed,
        ) = carry

        # ----------------------------------------------------
        # Do nothing if already converged
        # ----------------------------------------------------

        def do_nothing(_):
            return carry

        # ----------------------------------------------------
        # Perform SQP iteration
        # ----------------------------------------------------

        def do_iter(_):

            # =================================================
            # Evaluate current nonlinear model
            # =================================================

            g, c = model_evaluator(
                X_curr,
                U_curr,
            )

            # Dynamics / initial-condition feasibility
            feas = jnp.max(
                jnp.abs(c)
            )

            # =================================================
            # ADMM warm start
            # =================================================

            warm_flag = jnp.array(
                bool(
                    sqp_config.warm_start
                )
            )

            w0 = lax.select(
                warm_flag,
                w,
                jnp.zeros_like(w),
            )

            y0 = lax.select(
                warm_flag,
                y,
                jnp.zeros_like(y),
            )

            rho0 = lax.select(
                warm_flag,
                rho,
                jnp.asarray(
                    admm_config.initial_rho,
                    dtype=rho.dtype,
                ),
            )

            # Current robust tightening gets supplied as
            # warm start for the next search-direction solve.

            h_ct_current = backoffs

            # =================================================
            # Solve SQP subproblem
            # =================================================

            (
                dX,
                dU,
                dV,
                q,
                r,
                w1,
                y1,
                rho1,
                backoffs1,
                Phi_x1,
                Phi_u1,
                betaN,
                muN,
                K_term,
            ) = compute_search_direction(
                sls_config,
                admm_config,
                _cost,
                _dynamics,
                _hessian_approx,
                constraints,
                disturbance,
                obstacles,
                x0,
                X_curr,
                U_curr,
                V_curr,
                c,
                w0,
                y0,
                rho0,
                h_ct_current,
                beta_ws,
                mu_ws,
                Phi_x_ws,
                Phi_u_ws,
                K_fixed,
                i,
            )

            # =================================================
            # SQP convergence test
            # =================================================

            step = jnp.maximum(
                jnp.max(
                    jnp.abs(dX)
                ),
                jnp.max(
                    jnp.abs(dU)
                ),
            )

            z_norm = jnp.maximum(
                jnp.max(
                    jnp.abs(X_curr)
                ),
                jnp.max(
                    jnp.abs(U_curr)
                ),
            )

            feas_ok = (
                feas
                <= sqp_config.feas_tol
            )

            step_ok = (
                step
                <= sqp_config.step_tol
                * (1.0 + z_norm)
            )

            converged1 = jnp.logical_and(
                feas_ok,
                step_ok,
            )

            # =================================================
            # Equality merit penalty
            # =================================================

            rho_merit = merit_rho(
                c,
                dV,
            )

            # =================================================
            # Determine whether to run line search
            # =================================================

            last_iter = (
                i
                == (
                    sqp_config.max_sqp_iterations
                    + sls_config.max_initial_sqp_iterations
                    - 1
                )
            )

            do_ls = jnp.logical_and(
                jnp.array(
                    bool(
                        sqp_config.line_search
                    )
                ),
                jnp.logical_not(
                    last_iter
                ),
            )

            # =================================================
            # Constraint-aware line search
            # =================================================

            def ls_branch(_):

                (
                    Xn,
                    Un,
                    Vn,
                    alpha,
                    accepted,
                ) = constrained_line_search(
                    model_evaluator,
                    inequality_evaluator,
                    X_curr,
                    U_curr,
                    V_curr,
                    dX,
                    dU,
                    dV,

                    # IMPORTANT:
                    #
                    # Use the newly computed robust backoff
                    # because this is the tightening associated
                    # with the current search direction.
                    backoffs1,

                    rho_merit,

                    # Tune this if necessary.
                    rho_ineq=100.0,

                    armijo_factor=1e-4,
                    alpha_0=1.0,
                    alpha_mult=0.5,
                    alpha_min=1e-6,
                )

                # Useful for debugging:
                #
                # jax.debug.print(
                #     "SQP {} alpha={} accepted={}",
                #     i,
                #     alpha,
                #     accepted,
                # )

                return (
                    Xn,
                    Un,
                    Vn,
                )

            # =================================================
            # Full step
            # =================================================

            def fullstep_branch(_):

                return (
                    X_curr + dX,
                    U_curr + dU,
                    V_curr + dV,
                )

            # =================================================
            # Take a step only if NOT converged
            #
            # This fixes the bug in the original code where
            # X_next/U_next/V_next were first selected based on
            # convergence and then immediately overwritten by
            # the line-search/full-step result.
            # =================================================

            def take_step(_):

                return lax.cond(
                    do_ls,
                    ls_branch,
                    fullstep_branch,
                    operand=None,
                )

            def keep_current(_):

                return (
                    X_curr,
                    U_curr,
                    V_curr,
                )

            (
                X_next,
                U_next,
                V_next,
            ) = lax.cond(
                converged1,
                keep_current,
                take_step,
                operand=None,
            )

            # =================================================
            # Warm-start state update
            # =================================================

            w_next = lax.select(
                converged1,
                w,
                w1,
            )

            y_next = lax.select(
                converged1,
                y,
                y1,
            )

            rho_next = lax.select(
                converged1,
                rho,
                rho1,
            )

            backoffs_next = lax.select(
                converged1,
                backoffs,
                backoffs1,
            )

            Phi_x_next = lax.select(
                converged1,
                Phi_x,
                Phi_x1,
            )

            Phi_u_next = lax.select(
                converged1,
                Phi_u,
                Phi_u1,
            )

            # -------------------------------------------------
            # Advance SQP iteration
            # -------------------------------------------------

            return (
                i + 1,
                X_next,
                U_next,
                V_next,
                w_next,
                y_next,
                rho_next,
                jnp.logical_or(
                    converged,
                    converged1,
                ),
                backoffs_next,
                Phi_x_next,
                Phi_u_next,
                betaN,
                muN,
                K_term,
            )

        return lax.cond(
            converged,
            do_nothing,
            do_iter,
            operand=None,
        )

    # ========================================================
    # Initial carry
    # ========================================================

    backoffs0 = h_ct_ws

    carry0 = (
        0,
        X_in,
        U_in,
        V_in,
        w,
        y,
        rho,
        jnp.array(False),
        backoffs0,
        Phi_x_ws,
        Phi_u_ws,
        beta_ws,
        mu_ws,
        K_fixed,
    )

    # ========================================================
    # SQP loop
    # ========================================================

    (
        total_iterations,
        X_out,
        U_out,
        V_out,
        w_out,
        y_out,
        rho_out,
        converged,
        backoffs,
        Phi_x,
        Phi_u,
        betaN,
        muN,
        K_term,
    ) = lax.fori_loop(
        0,
        (
            sqp_config.max_sqp_iterations
            + sls_config.max_initial_sqp_iterations
        ),
        body,
        carry0,
    )

    return (
        X_out,
        U_out,
        V_out,
        w_out,
        y_out,
        rho_out,
        backoffs,
        Phi_x,
        Phi_u,
        betaN,
        muN,
        K_term,
    )