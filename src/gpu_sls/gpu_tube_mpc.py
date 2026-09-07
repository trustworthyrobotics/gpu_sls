from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from gpu_sls.gpu_admm import constrained_solve


@partial(jit, static_argnums=(0, 19))
def tube_mpc(
    admm_config,
    Q, q,
    R, r,
    M,
    A, B, c,
    C_all, D_all, f_all,
    w, y, rho, rho_grad,
    Jh, a, b, gradient_window,
    K_fixed,
    E,
    slack_weight=None,
):
    """
    Box tube MPC with uniform worst-case tube bound.

    Supports:

        K_fixed.shape == (nu, nx)

    or time-varying:

        K_fixed.shape == (T, nu, nx)
        K_fixed.shape == (T+1, nu, nx)

    Feedback:
        u_k = v_k - K_k e_k

    Error dynamics:
        e_{k+1} = (A_k - B_k K_k)e_k + E_k d_k
        ||d_k||_inf <= 1
    """

    # =========================================================
    # Dimensions
    # =========================================================

    Tp1 = Q.shape[0]
    T = Tp1 - 1
    nx = Q.shape[1]

    # =========================================================
    # Build time-indexed K for dynamics and constraints
    # =========================================================
    #
    # Dynamics need:
    #
    #     K_dyn.shape = (T, nu, nx)
    #
    # Constraints need:
    #
    #     K_con.shape = (T+1, nu, nx)
    #
    # because C_all / D_all generally have T+1 stages.
    # =========================================================
    # K_fixed = jnp.zeros_like(K_fixed)
    if K_fixed.ndim == 2:

        # Single constant controller:
        #
        # K_fixed: (nu, nx)

        K_dyn = jnp.broadcast_to(
            K_fixed[None, :, :],
            (T,) + K_fixed.shape,
        )

        K_con = jnp.broadcast_to(
            K_fixed[None, :, :],
            (Tp1,) + K_fixed.shape,
        )

    elif K_fixed.ndim == 3:

        # Time-varying controller.
        #
        # Could be:
        #   (T,   nu, nx)
        # or
        #   (T+1, nu, nx)

        if K_fixed.shape[0] == Tp1:

            K_dyn = K_fixed[:-1]
            K_con = K_fixed

        elif K_fixed.shape[0] == T:

            K_dyn = K_fixed

            # Terminal stage has no actual control action in most
            # MPC formulations. Repeating the last gain is harmless
            # if D_all[-1] == 0.
            K_con = jnp.concatenate(
                [
                    K_fixed,
                    K_fixed[-1:, :, :],
                ],
                axis=0,
            )

        else:
            raise ValueError(
                "Time-varying K_fixed must have shape "
                "(T, nu, nx) or (T+1, nu, nx)."
            )

    else:
        raise ValueError(
            "K_fixed must have shape (nu, nx), "
            "(T, nu, nx), or (T+1, nu, nx)."
        )

    # =========================================================
    # Closed-loop dynamics
    #
    # B:
    #     (T, nx, nu)
    #
    # K_dyn:
    #     (T, nu, nx)
    #
    # B @ K:
    #     (T, nx, nx)
    # =========================================================

    BK = jnp.einsum(
        "kij,kjl->kil",
        B,
        K_dyn,
    )

    A_cl = A + BK

    # =========================================================
    # Disturbance box bound
    # =========================================================

    if E.ndim == 2:

        # E:
        #     (T, nx)
        #
        # Already interpreted as componentwise disturbance bound.

        disturbance_bound = jnp.abs(E)

    else:

        # E:
        #     (T, nx, nw)
        #
        # d:
        #     ||d||_inf <= 1
        #
        # Therefore:
        #
        #     |E d| <= |E| 1

        disturbance_bound = jnp.sum(
            jnp.abs(E),
            axis=-1,
        )

    # =========================================================
    # Propagate box tube
    #
    # e_0 = 0
    #
    # e_{k+1}
    #     = |A_cl[k]| e_k
    #       + disturbance_bound[k]
    # =========================================================

    e_bounds = jnp.zeros(
        (Tp1, nx),
        dtype=Q.dtype,
    )

    def propagate(k, e_bounds):

        e_next = (
            jnp.abs(A_cl[k]) @ e_bounds[k]
            + disturbance_bound[k]
        )

        return e_bounds.at[k + 1].set(e_next)

    e_bounds = lax.fori_loop(
        0,
        T,
        propagate,
        e_bounds,
    )

    # =========================================================
    # Uniform worst-case tube radius
    #
    # Componentwise maximum over the entire horizon:
    #
    #     e_max[j] = max_k e_bounds[k,j]
    #
    # shape:
    #     (nx,)
    # =========================================================

    e_bound_max = jnp.max(
        e_bounds,
        axis=0,
    )

    # =========================================================
    # Constraint tightening
    #
    # Original:
    #
    #     C_k x_k + D_k u_k <= f_k
    #
    # x = z + e
    # u = v - K e
    #
    # therefore:
    #
    #     C z + D v + (C - D K)e <= f
    #
    # and:
    #
    #     h_k = |C_k - D_k K_k| e_max
    # =========================================================

    # D_all:
    #     (T+1, nc, nu)
    #
    # K_con:
    #     (T+1, nu, nx)
    #
    # DK:
    #     (T+1, nc, nx)

    DK = jnp.einsum(
        "kij,kjl->kil",
        D_all,
        K_con,
    )

    constraint_error_matrix = C_all + DK

    # Same e_bound_max is used for every stage.
    #
    # h_ct:
    #     (T+1, nc)

    # h_ct = jnp.einsum(
    #     "kij,j->ki",
    #     jnp.abs(constraint_error_matrix),
    #     e_bound_max,
    # )
    h_ct = jnp.einsum(
        "kij,kj->ki",
        jnp.abs(constraint_error_matrix),
        e_bounds,
    )

    # =========================================================
    # Tighten constraints
    # =========================================================

    f_tight = f_all - h_ct

    # =========================================================
    # Solve nominal MPC
    # =========================================================

    (
        dX,
        dU,
        dV,
        w1,
        y1,
        rho1,
        rho_grad1,
        mu,
        a1,
        b1,
        converged_admm,
    ) = constrained_solve(
        admm_config,
        Q, q,
        R, r,
        M,
        A, B, c,
        C_all,
        D_all,
        f_tight,
        w, y, rho, rho_grad,
        Jh, a, b, gradient_window,
        just_nominal=False,
        slack_weight=slack_weight,
    )

    return (
        dX,
        dU,
        dV,
        w1,
        y1,
        rho1,
        rho_grad1,
        e_bounds,
        h_ct,
        mu,
        a1,
        b1,
        converged_admm,
    )