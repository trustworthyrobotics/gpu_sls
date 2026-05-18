from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, lax, vmap

from gpu_sls.gpu_admm import constrained_solve, ADMMConfig
from gpu_sls.external.primal_dual_ilqr.primal_dual_ilqr.primal_tvlqr import tvlqr_gpu
from gpu_sls.gpu_sls import SLSConfig, get_etas, primal_convergence_metric, add_obstacle_tightenings, calculate_cost, get_constraint_tightenings

@jax.jit
def calculate_phis_state(A: jnp.ndarray, B: jnp.ndarray, Cx: jnp.ndarray, Cxu: jnp.ndarray, Cu: jnp.ndarray):
    T = Cu.shape[0]
    nx = A.shape[1]
    nu = B.shape[-1]
    Tp1 = T + 1
    A = A[:T]
    B = B[:T]
    # def solve_one_j(j):
    #     Qj = Cx[:, j, :, :]
    #     Rj = Cu[:, j, :, :]
    #     Mj = Cxu[:, j, :, :]
    #     K = controller_pas(Qj, Rj, Mj, A, B)
    #     return K
    zeros_q = jnp.zeros((Tp1, nx), dtype=A.dtype)
    zeros_r = jnp.zeros((T,  nu), dtype=A.dtype)
    zeros_c = jnp.zeros((T,  nx), dtype=A.dtype)
    def solve_one_j(j):
        Qj = Cx[:, j, :, :]
        Rj = Cu[:, j, :, :]
        Mj = Cxu[:, j, :, :]
        K, _, _, _ = tvlqr_gpu(Qj, zeros_q, Rj, zeros_r, Mj, A, B, zeros_c)
        return K   

    K_all = jax.vmap(solve_one_j)(jnp.arange(T))
    K_kj_core = jnp.swapaxes(K_all, 0, 1)
    K_lastcol = jnp.zeros((T, 1, nu, nx), dtype=A.dtype)
    K_kj = jnp.concatenate([K_kj_core, K_lastcol], axis=1)

    BK = jnp.einsum("kxu,kjuy->kjxy", B, K_kj)
    F  = A[:, None, :, :] + BK

    I = jnp.eye(nx, dtype=A.dtype)
    I_horiz = jnp.broadcast_to(I, (Tp1, nx, nx))
    F = F.at[:, T].set(I)

    t_idx = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    use_F = (t_idx >= j_idx)
    elems = jnp.where(use_F[:, :, None, None], F, I)

    def compose(l, r):
        return jnp.einsum("...ab,...bc->...ac", r, l)

    P = lax.associative_scan(compose, elems, axis=0)
    Phix_1toT = jnp.einsum("tjab,jbn->tjan", P, I_horiz)
    Phi_x = jnp.concatenate(
        [jnp.zeros((1, Tp1, nx, nx), dtype=A.dtype), Phix_1toT],
        axis=0
    )

    Phi_x = Phi_x.at[jnp.arange(Tp1), jnp.arange(Tp1)].set(I)
    k_idx_full = jnp.arange(Tp1)[:, None]
    valid_x = (k_idx_full >= j_idx)
    Phi_x = Phi_x * valid_x[:, :, None, None]

    Phi_u = jnp.einsum("kjux,kjxn->kjun", K_kj, Phi_x[:-1])
    k_idx = jnp.arange(T)[:, None]
    valid_u = (k_idx >= j_idx)
    Phi_u = Phi_u * valid_u[:, :, None, None]

    return Phi_x, Phi_u

@jax.jit
def get_controller_state(Q: jnp.ndarray, R: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray,
                         C: jnp.ndarray, D: jnp.ndarray, eta_stage: jnp.ndarray, eta_f: jnp.ndarray):
    T, nx, _ = A.shape

    js = jnp.arange(T)
    ks = jnp.arange(T)

    def blocks_for_k(k):
        def blocks_for_j(j):
            return calculate_cost(Q[k], R[k], C[k], D[k], eta_stage[k, j])
        return vmap(blocks_for_j)(js)

    Cx_kj, Cxu_kj, Cu_kj = vmap(blocks_for_k)(ks)

    Cterm = C[-1]
    def terminal_Cx_for_j(j):
        w = eta_f[j]
        return (Cterm.T * w[None, :]) @ Cterm + Q[T]

    Cx_Nj = vmap(terminal_Cx_for_j)(jnp.arange(T))
    Cx = jnp.concatenate([Cx_kj, Cx_Nj[None, ...]], axis=0)

    Phi_x, Phi_u = calculate_phis_state(A, B, Cx, Cxu_kj, Cu_kj)
    return Phi_x, Phi_u

@jax.jit
def get_observer_gains(
    A: jnp.ndarray,
    C: jnp.ndarray,
    E: jnp.ndarray,
    F: jnp.ndarray,
    Xi: jnp.ndarray,
):
    T = A.shape[0]
    nx = A.shape[1]
    ny = C.shape[1]
    Tp1 = T + 1
    dtype = A.dtype

    A_dual = jnp.swapaxes(A, -1, -2)
    B_dual = jnp.swapaxes(C[:T], -1, -2)

    Q_dual = jnp.einsum("knw,kmw->knm", E[:T], E[:T])
    R_dual = jnp.einsum("knw,kmw->knm", F[:T], F[:T])

    M_dual = jnp.zeros((T, nx, ny), dtype=dtype)

    zeros_q = jnp.zeros((Tp1, nx), dtype=dtype)
    zeros_r = jnp.zeros((T, ny), dtype=dtype)
    zeros_c = jnp.zeros((T, nx), dtype=dtype)

    Pi0 = Xi @ Xi.T
    I_y = jnp.eye(ny, dtype=dtype)

    A_rev = A_dual[::-1]
    B_rev = B_dual[::-1]
    Q_rev = Q_dual[::-1]
    R_rev = R_dual[::-1]
    M_rev = M_dual[::-1]

    def solve_one_k(k):
        active_orig = jnp.arange(T) >= k

        active_rev = active_orig[::-1]

        Q_roll = jnp.where(active_rev[:, None, None], Q_rev, jnp.zeros_like(Q_rev))
        R_roll = jnp.where(active_rev[:, None, None], R_rev, I_y[None, :, :])
        M_roll = jnp.where(active_rev[:, None, None], M_rev, jnp.zeros_like(M_rev))

        Q_full = jnp.concatenate([Q_roll, Pi0[None, :, :]], axis=0)

        K_rev, _, _, _ = tvlqr_gpu(
            Q_full,
            zeros_q,
            R_roll,
            zeros_r,
            M_roll,
            A_rev,
            B_rev,
            zeros_c,
        )

        L_forward = K_rev[::-1]
        valid_orig = active_orig
        L_forward = L_forward * valid_orig[:, None, None]

        L_row = jnp.zeros((Tp1, ny, nx), dtype=dtype)
        L_row = L_row.at[1:].set(L_forward)
        return L_row

    L_core = jax.vmap(solve_one_k)(jnp.arange(T))
    L_last = jnp.zeros((1, Tp1, ny, nx), dtype=dtype)
    return jnp.concatenate([L_core, L_last], axis=0)

@jax.jit
def calculate_observer_phis(A: jnp.ndarray, C: jnp.ndarray, L_kj: jnp.ndarray):
    T = A.shape[0]
    nx = A.shape[1]
    ny = C.shape[1]
    Tp1 = T + 1
    dtype = A.dtype

    I = jnp.eye(nx, dtype=dtype)
    Aobs = A[:, None, :, :] + jnp.einsum(
        "tjyn,tym->tjnm",
        L_kj[1:],   # (T, T+1, ny, nx)
        C[1:],      # (T, ny, nx)
    )
    Aobs_rev = Aobs[::-1]

    t_rev = jnp.arange(T)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    k_orig = T - 1 - t_rev
    active = k_orig < j_idx

    elems = jnp.where(
        active[:, :, None, None],
        Aobs_rev,
        I[None, None, :, :],
    )

    def compose(left, right):
        return jnp.einsum("...ab,...bc->...ac", right, left)

    P_rev = lax.associative_scan(compose, elems, axis=0)
    P = P_rev[::-1]
    Phi_x_o = jnp.zeros((Tp1, Tp1, nx, nx), dtype=dtype)
    Phi_x_o = Phi_x_o.at[:, :T].set(jnp.swapaxes(P, 0, 1))

    Phi_x_o = Phi_x_o.at[jnp.arange(Tp1), jnp.arange(Tp1), :, :].set(I)

    row_j = jnp.arange(Tp1)[:, None]
    col_k = jnp.arange(Tp1)[None, :]

    valid = col_k <= row_j
    Phi_x_o = Phi_x_o * valid[:, :, None, None]

    Phi_y_o_core = jnp.einsum("jkab,kjyb->jkay", Phi_x_o[:, 1:], L_kj[1:])
    Phi_y_o = jnp.concatenate([jnp.zeros((Tp1, 1, nx, ny), dtype=dtype), Phi_y_o_core], axis=1)

    valid_y = (col_k <= row_j) & (col_k > 0)
    Phi_y_o = Phi_y_o * valid_y[:, :, None, None]

    return Phi_x_o, Phi_y_o

@jax.jit
def get_controller_obs(A: jnp.ndarray, C_obs: jnp.ndarray, E: jnp.ndarray, F: jnp.ndarray, Xi: jnp.ndarray):
    L = get_observer_gains(A, C_obs, E, F, Xi)
    Phi_x_o, Phi_y_o = calculate_observer_phis(A, C_obs, L)
    return Phi_x_o, Phi_y_o

@jax.jit
def assemble_output_feedback_phis(A: jnp.ndarray, Phi_x_f: jnp.ndarray, Phi_u_f: jnp.ndarray, Phi_x_o: jnp.ndarray, Phi_y_o: jnp.ndarray):
    # M Phi_x_o = (I - Z A) Phi_x_o
    # row 0: Phi_x_o[0]
    # row k: Phi_x_o[k] - A[k-1] @ Phi_x_o[k-1]
    M_Phi_x_o = Phi_x_o.at[1:].set(
        Phi_x_o[1:] - jnp.einsum("kab,kjbc->kjac", A, Phi_x_o[:-1])
    )

    # M Phi_y_o = (I - Z A) Phi_y_o
    M_Phi_y_o = Phi_y_o.at[1:].set(
        Phi_y_o[1:] - jnp.einsum("kab,kjbc->kjac", A, Phi_y_o[:-1])
    )

    # Phi_x_f @ M_Phi_x_o
    Phi_x_f_M_xo = jnp.einsum(
        "ktab,tjbc->kjac",
        Phi_x_f,
        M_Phi_x_o,
    )

    # Phi_u_f @ M_Phi_x_o
    Phi_u_f_M_xo = jnp.einsum(
        "ktab,tjbc->kjac",
        Phi_u_f,
        M_Phi_x_o,
    )

    # Phi_x_f @ M_Phi_y_o
    Phi_x_f_M_yo = jnp.einsum(
        "ktab,tjbc->kjac",
        Phi_x_f,
        M_Phi_y_o,
    )

    # Phi_u_f @ M_Phi_y_o
    Phi_u_f_M_yo = jnp.einsum(
        "ktab,tjbc->kjac",
        Phi_u_f,
        M_Phi_y_o,
    )

    Phi_xw = Phi_x_f + Phi_x_o - Phi_x_f_M_xo
    Phi_uw = Phi_u_f - Phi_u_f_M_xo
    Phi_xe = Phi_y_o - Phi_x_f_M_yo
    Phi_ue = -Phi_u_f_M_yo

    return Phi_xw, Phi_uw, Phi_xe, Phi_ue

@jax.jit
def get_betas_output_feedback(
    C: jnp.ndarray,
    D: jnp.ndarray,
    Phi_xw: jnp.ndarray,
    Phi_uw: jnp.ndarray,
    Phi_xe: jnp.ndarray,
    Phi_ue: jnp.ndarray,
    E: jnp.ndarray,
    F: jnp.ndarray,
):
    Tp1 = Phi_xw.shape[0]
    T = Phi_uw.shape[0]

    C_stage = C[:T]
    D_stage = D[:T]
    C_term = C[T]

    E_all = E[:Tp1]
    F_all = F[:Tp1]

    proc_stage = (
        jnp.einsum("kcx,kjxn,jnw->kjcw", C_stage, Phi_xw[:T], E_all)
        + jnp.einsum("kcu,kjun,jnw->kjcw", D_stage, Phi_uw, E_all)
    )

    meas_stage = (
        jnp.einsum("kcx,kjxy,jyv->kjcv", C_stage, Phi_xe[:T], F_all)
        + jnp.einsum("kcu,kjuy,jyv->kjcv", D_stage, Phi_ue, F_all)
    )

    proc_terminal = jnp.einsum(
        "cx,jxn,jnw->jcw",
        C_term,
        Phi_xw[T],
        E_all,
    )

    meas_terminal = jnp.einsum(
        "cx,jxy,jyv->jcv",
        C_term,
        Phi_xe[T],
        F_all,
    )

    proc_stage_norm = jnp.linalg.norm(proc_stage, axis=-1)
    meas_stage_norm = jnp.linalg.norm(meas_stage, axis=-1)

    proc_terminal_norm = jnp.linalg.norm(proc_terminal, axis=-1)
    meas_terminal_norm = jnp.linalg.norm(meas_terminal, axis=-1)

    # beta convention: beta stores squared tightening contribution.
    # get_constraint_tightenings later takes sqrt(beta).
    beta_stage = (proc_stage_norm + meas_stage_norm) ** 2
    beta_terminal = (proc_terminal_norm + meas_terminal_norm) ** 2

    beta = jnp.concatenate(
        [beta_stage, beta_terminal[None, :, :]],
        axis=0,
    )

    k_idx = jnp.arange(Tp1)[:, None]
    j_idx = jnp.arange(Tp1)[None, :]
    valid = k_idx >= j_idx

    return beta * valid[:, :, None]


@partial(jit, static_argnums=(0, 15))
def sls_of_solve_gpu(cfg: ADMMConfig, Q: jnp.ndarray, q: jnp.ndarray, R: jnp.ndarray, r: jnp.ndarray, M: jnp.ndarray,
                     A: jnp.ndarray, B: jnp.ndarray, c: jnp.ndarray,
                     C: jnp.ndarray, D: jnp.ndarray, f: jnp.ndarray,
                     w: jnp.ndarray, y: jnp.ndarray, rho: jnp.ndarray,
                     sls_config: SLSConfig, E: jnp.ndarray, Q_bar: jnp.ndarray, R_bar: jnp.ndarray,
                     obstacles: jnp.ndarray, primal_pos: jnp.ndarray, h_ct_ws: jnp.ndarray,
                     beta_ws: jnp.ndarray, mu_ws: jnp.ndarray,
                     C_output: jnp.ndarray, F: jnp.ndarray, Xi: jnp.ndarray):
    Tp1 = Q.shape[0]
    nx  = Q.shape[1]
    nu  = R.shape[1]
    ny = C_output.shape[-1]
    num_obstacles = obstacles.shape[0]
    T   = Tp1 - 1
    Phi_xw0 = jnp.zeros((Tp1, Tp1, nx, nx))
    Phi_uw0 = jnp.zeros((T, Tp1, nu, nx))
    Phi_xe0 = jnp.zeros((Tp1, Tp1, nx, ny))
    Phi_ue0 = jnp.zeros((T, Tp1, nu, ny))
    x0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)
    u0 = jnp.zeros((T, nu),  dtype=Q.dtype)
    v0 = jnp.zeros((Tp1, nx), dtype=Q.dtype)

    i0 = jnp.array(0, dtype=rho.dtype)
    converged0 = jnp.array(False)

    max_iter = jnp.array(sls_config.max_sls_iterations, dtype=jnp.int32)
    tol = jnp.array(sls_config.sls_primal_tol, dtype=Q.dtype)

    h_ct0 = h_ct_ws
    carry0 = (i0, beta_ws, x0, u0, v0, w, y, rho, converged0, converged0, h_ct0, Phi_xw0, Phi_uw0, Phi_xe0, Phi_ue0, mu_ws)
    Phi_x_o, Phi_y_o = get_controller_obs(A, C_output, E, F, Xi)
    # Phi_x_o0 = jnp.ones((Tp1, Tp1, nx, nx)) * 2
    # Phi_y_o0 = jnp.ones((Tp1, Tp1, nx, nx)) * 2
    # jax.debug.print("{}, {}", Phi_x_o.shape, Phi_y_o.shape)
    def cond_fn(carry):
        i, _, _, _, _, _, _, _, converged, _, _, _, _, _, _, _ = carry
        return jnp.logical_and(i < max_iter, jnp.logical_not(converged))

    def body_fn(carry):
        i, beta, x_curr, u_curr, v_curr, w, y, rho, converged, _, h_ct, _, _, _, _, mu = carry

        prev_rho = rho
        x_prev = x_curr
        u_prev = u_curr

        # TODO: Fix this
        # if sls_config.rti:
        #     mu_nominal = mu[: , :-num_obstacles]
        #     eta_stage, eta_f = get_etas(mu_nominal, beta)
        #     C_box = C[:, :nc - num_obstacles, :]
        #     D_box = D[:, :nc - num_obstacles, :]
        #     Phi_x, Phi_u = get_controller(Q_bar, R_bar, A, B, C_box, D_box, E, eta_stage, eta_f)
        #     beta = get_betas(C_box, D_box, Phi_x, Phi_u)
        #     h_ct = get_constraint_tightenings(beta)

        num_regular_constraints = f.shape[1] - num_obstacles
        tightened_constraints = f[:, :num_regular_constraints] - h_ct
        tightened_constraints_all = add_obstacle_tightenings(obstacles, primal_pos, h_ct, tightened_constraints)
        warm_flag = jnp.array(bool(sls_config.warm_start))

        w   = lax.select(warm_flag, w, jnp.zeros_like(w))
        y   = lax.select(warm_flag, y, jnp.zeros_like(y))
        rho = lax.select(warm_flag, rho, jnp.array(cfg.initial_rho, dtype=rho.dtype))
        x_curr, u_curr, v_curr, w, y, rho, mu, converged_admm = constrained_solve(
            cfg, Q, q, R, r, M, A, B, c, C, D, tightened_constraints_all, w, y, rho
        )

        metric = primal_convergence_metric(x_curr, u_curr, x_prev, u_prev)
        mu_nominal = mu[: , :num_regular_constraints]
        eta_stage, eta_f = get_etas(mu_nominal, beta)
        C_box = C[:, :num_regular_constraints, :]
        D_box = D[:, :num_regular_constraints, :]
        Phi_x_f, Phi_u_f = get_controller_state(Q_bar, R_bar, A, B, C_box, D_box, eta_stage, eta_f)
        Phi_xw, Phi_uw, Phi_xe, Phi_ue = assemble_output_feedback_phis(A, Phi_x_f, Phi_u_f, Phi_x_o, Phi_y_o)
        beta = get_betas_output_feedback(C_box, D_box, Phi_xw, Phi_uw, Phi_xe, Phi_ue, E, F)
        h_ct = get_constraint_tightenings(beta)
        rho = jnp.asarray(rho, dtype=prev_rho.dtype)
        w   = jnp.asarray(w,   dtype=w.dtype)
        y   = jnp.asarray(y,   dtype=y.dtype)
        converged_now = metric <= tol
        converged = jnp.logical_or(converged, converged_now)

        return (i + jnp.array(1, dtype=jnp.int32),
                beta, x_curr, u_curr, v_curr, w, y, rho, converged, converged_admm, h_ct, Phi_xw, Phi_uw, Phi_xe, Phi_ue, mu)

    carryN = jax.lax.while_loop(cond_fn, body_fn, carry0)

    _, betaN, xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_xw, Phi_uw, Phi_xe, Phi_ue, muN = carryN
    return xN, uN, vN, wN, yN, rhoN, convergedN, converged_admm, h_ct, Phi_xw, Phi_uw, Phi_xe, Phi_ue, betaN, muN