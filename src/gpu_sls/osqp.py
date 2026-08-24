"""OSQP backend for the linearized finite-horizon constrained control QP.

This module is intentionally a thin replacement for the custom ADMM/LQR
``constrained_solve`` backend.  It assembles the entire horizon QP and hands it
to OSQP.

The public ``constrained_solve`` signature is kept close to the original file
so an existing caller can usually change only the imported solver module.
The ADMM split variables (w, y, rho_grad, a, b) are accepted for compatibility,
but OSQP does not need them internally.

QP convention
-------------
For stages k = 0, ..., N-1, solve

    min  sum_k [
             0.5 x_k' Q_k x_k + q_k' x_k
           + 0.5 u_k' R_k u_k + r_k' u_k
           +       x_k' M_k u_k
         ]
         + 0.5 x_N' Q_N x_N + q_N' x_N

subject to

    x_{k+1} = A_k x_k + B_k u_k + c_{k+1}

and, when C/D/f are supplied,

    C_k x_k + D_k u_k <= f_k.

``Jh`` and ``L`` are accepted only for compatibility with the previous backend;
they are not used in the OSQP problem.

Initial-state convention
------------------------
As in the original backend, ``c[0]`` stores the initial state/perturbation.
If ``cfg.num_phases > 0``, the last ``num_phases`` entries of x_0 are left
free so OSQP can optimize phase-duration perturbations; all preceding entries
of x_0 are fixed to c[0].

Important
---------
The standard Python ``osqp`` package is CPU/host code.  This function therefore
must be called outside ``jax.jit``.  JAX arrays are accepted as inputs, but they
must be concrete arrays (not tracers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import osqp

try:
    import jax.numpy as jnp
except ImportError:  # JAX is optional for this backend itself.
    jnp = None


@dataclass(frozen=True)
class ADMMConfig:
    """Compatibility configuration for the old solver API.

    Existing code that constructs ``ADMMConfig`` can keep doing so.  Only a
    subset of the fields are meaningful to OSQP; the remaining fields are
    retained so old configuration code does not need to change.
    """

    rho_update_frequency: int = 25       # compatibility only
    max_iterations: int = 400
    eps_abs: float = 1e-2
    eps_rel: float = 1e-2
    eps_abs_grad: float = 1e-2           # compatibility only
    eps_rel_grad: float = 1e-2           # compatibility only
    rho_max: float = 1e5                 # compatibility only
    initial_rho: float = 1.0
    regularized_rho_update: bool = False # compatibility only
    num_phases: int = 1

    # OSQP-specific options.
    verbose: bool = False
    polish: bool = False
    adaptive_rho: bool = True
    # Convex-QP handling. OSQP requires P >= 0.  The SQP layer has already
    # linearized the dynamics/constraints before this function is called; these
    # settings only make the supplied quadratic model admissible to OSQP.
    regularization: float = 1e-9
    convexify_hessian: bool = True
    psd_eigenvalue_floor: float = 1e-8
    report_hessian_shift: bool = False
    # Multiply the entire objective by a positive scalar before OSQP setup.
    # This does NOT change the QP minimizer; it only improves numerical scaling.
    normalize_objective: bool = True
    report_qp_scaling: bool = False


# A clearer alias for new code, while preserving ADMMConfig for drop-in use.
OSQPConfig = ADMMConfig


def _np(x: Any, name: str) -> np.ndarray:
    """Convert a concrete NumPy/JAX array to float64 NumPy."""
    try:
        out = np.asarray(x, dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a concrete NumPy/JAX array. "
            "OSQP cannot consume a JAX tracer inside jax.jit."
        ) from exc
    return out


def _require_finite(name: str, x: np.ndarray) -> None:
    """Reject NaN/Inf in QP data that must be finite."""
    if not np.all(np.isfinite(x)):
        bad = np.argwhere(~np.isfinite(x))
        idx = tuple(bad[0]) if bad.size else ()
        val = x[idx] if idx else np.nan
        raise ValueError(
            f"Non-finite value in linearized QP array {name} at {idx}: {val}. "
            "Fix this upstream before calling OSQP."
        )


def _to_output(x: np.ndarray):
    """Return JAX arrays when JAX is installed, otherwise NumPy arrays."""
    if jnp is not None:
        return jnp.asarray(x)
    return x


def _validate_shapes(Q, q, R, r, M, A, B, c):
    N = A.shape[0]
    nx = A.shape[1]
    nu = B.shape[2]

    if A.shape != (N, nx, nx):
        raise ValueError(f"A must have shape (N,nx,nx); got {A.shape}")
    if B.shape != (N, nx, nu):
        raise ValueError(f"B must have shape (N,nx,nu); got {B.shape}")
    if Q.shape != (N + 1, nx, nx):
        raise ValueError(f"Q must have shape {(N + 1, nx, nx)}; got {Q.shape}")
    if q.shape != (N + 1, nx):
        raise ValueError(f"q must have shape {(N + 1, nx)}; got {q.shape}")
    if R.shape != (N, nu, nu):
        raise ValueError(f"R must have shape {(N, nu, nu)}; got {R.shape}")
    if r.shape != (N, nu):
        raise ValueError(f"r must have shape {(N, nu)}; got {r.shape}")
    if M.shape != (N, nx, nu):
        raise ValueError(f"M must have shape {(N, nx, nu)}; got {M.shape}")
    if c.shape != (N + 1, nx):
        raise ValueError(
            f"c must have shape {(N + 1, nx)} with c[0]=x0 and "
            f"c[1:] the affine dynamics offsets; got {c.shape}"
        )

    return N, nx, nu


def _assemble_objective(
    Q,
    q,
    R,
    r,
    M,
    regularization: float,
    convexify_hessian: bool,
    psd_eigenvalue_floor: float,
    report_hessian_shift: bool,
    normalize_objective: bool,
    report_qp_scaling: bool,
):
    """Build the *already-linearized* QP objective for OSQP.

    For stage k the supplied quadratic model is

        0.5 x' Q x + x' M u + 0.5 u' R u,

    whose Hessian is [[Q, M], [M.T, R]].  OSQP is a convex-QP solver, so this
    Hessian must be PSD.  SQP models can be indefinite (especially when using
    an exact Lagrangian Hessian or free-time cross terms), even though the
    dynamics and constraints are already linearized.

    When ``convexify_hessian`` is enabled, a Levenberg-Marquardt diagonal shift
    is applied independently to each stage Hessian.  No nonlinear dynamics,
    constraints, Jacobians, or Hessians are evaluated here.
    """
    N = R.shape[0]
    nx = Q.shape[1]
    nu = R.shape[1]

    Qs = 0.5 * (Q + np.swapaxes(Q, -1, -2))
    Rs = 0.5 * (R + np.swapaxes(R, -1, -2))

    floor = max(float(psd_eigenvalue_floor), 0.0)
    worst_before = np.inf
    max_shift = 0.0

    # Convexify the joint [x_k, u_k] Hessian, not Q and R separately.
    # Q >= 0 and R >= 0 alone are insufficient when M != 0.
    for k in range(N):
        Hk = np.block([[Qs[k], M[k]], [M[k].T, Rs[k]]])
        lam_min = float(np.linalg.eigvalsh(Hk)[0])
        worst_before = min(worst_before, lam_min)
        if lam_min < floor:
            if not convexify_hessian:
                raise ValueError(
                    f"OSQP requires a convex QP, but stage {k} Hessian "
                    f"[[Q,M],[M.T,R]] has lambda_min={lam_min:.3e}. "
                    "Use a PSD/Gauss-Newton SQP Hessian or enable "
                    "cfg.convexify_hessian."
                )
            shift = floor - lam_min
            Qs[k] = Qs[k] + shift * np.eye(nx)
            Rs[k] = Rs[k] + shift * np.eye(nu)
            max_shift = max(max_shift, shift)

    # Terminal state block has no control block.
    terminal_min = float(np.linalg.eigvalsh(Qs[N])[0])
    worst_before = min(worst_before, terminal_min)
    if terminal_min < floor:
        if not convexify_hessian:
            raise ValueError(
                f"OSQP requires a convex QP, but terminal Q has "
                f"lambda_min={terminal_min:.3e}."
            )
        shift = floor - terminal_min
        Qs[N] = Qs[N] + shift * np.eye(nx)
        max_shift = max(max_shift, shift)

    n_xvars = (N + 1) * nx
    n_uvars = N * nu
    nz = n_xvars + n_uvars

    blocks = [Qs[k] for k in range(N + 1)]
    blocks += [Rs[k] for k in range(N)]
    P = sp.block_diag(blocks, format="lil")

    # OSQP uses 0.5 z' P z.  Symmetric P_xu=M, P_ux=M.T therefore gives
    # exactly x' M u in the supplied linearized QP.
    for k in range(N):
        xs = slice(k * nx, (k + 1) * nx)
        us0 = n_xvars + k * nu
        us = slice(us0, us0 + nu)
        P[xs, us] = M[k]
        P[us, xs] = M[k].T

    if regularization > 0.0:
        P = P + float(regularization) * sp.eye(nz, format="lil")

    if report_hessian_shift and max_shift > 0.0:
        print(
            "OSQP convexified supplied linearized QP Hessian: "
            f"worst lambda_min={worst_before:.3e}, max diagonal shift={max_shift:.3e}"
        )

    linear = np.concatenate([q.reshape(-1), r.reshape(-1)])
    P = P.tocsc()

    if P.data.size and not np.all(np.isfinite(P.data)):
        raise ValueError("Non-finite value found in assembled OSQP Hessian P.")
    _require_finite("linear objective", linear)

    # Scaling P and q by the same positive constant leaves argmin unchanged:
    #   argmin 0.5 z'Pz + q'z == argmin (0.5 z'Pz + q'z)/scale.
    # This is useful when SQP/free-time terms produce very large coefficients.
    objective_scale = 1.0
    if normalize_objective:
        p_abs_max = float(np.max(np.abs(P.data))) if P.data.size else 0.0
        q_abs_max = float(np.max(np.abs(linear))) if linear.size else 0.0
        objective_scale = max(1.0, p_abs_max, q_abs_max)
        if objective_scale > 1.0:
            P = P * (1.0 / objective_scale)
            linear = linear / objective_scale

    if report_qp_scaling:
        pmax_after = float(np.max(np.abs(P.data))) if P.data.size else 0.0
        qmax_after = float(np.max(np.abs(linear))) if linear.size else 0.0
        print(
            "OSQP QP scaling: "
            f"objective_scale={objective_scale:.3e}, "
            f"max|P|={pmax_after:.3e}, max|q|={qmax_after:.3e}"
        )

    return P, linear, n_xvars, nz


def _assemble_equalities(A, B, c, num_phases: int, n_xvars: int, nz: int):
    """Initial-state and dynamics equality constraints."""
    N, nx, _ = A.shape
    nu = B.shape[2]

    if not (0 <= num_phases <= nx):
        raise ValueError(
            f"num_phases must be between 0 and nx={nx}; got {num_phases}"
        )

    # Keep physical x0 fixed while allowing the augmented phase-time entries
    # at the end of x0 to be optimized.
    n_fixed_x0 = nx - num_phases
    n_eq = n_fixed_x0 + N * nx

    Aeq = sp.lil_matrix((n_eq, nz), dtype=np.float64)
    beq = np.empty(n_eq, dtype=np.float64)

    row = 0
    for i in range(n_fixed_x0):
        Aeq[row, i] = 1.0
        beq[row] = c[0, i]
        row += 1

    # x_{k+1} - A_k x_k - B_k u_k = c[k+1]
    for k in range(N):
        rows = slice(row, row + nx)
        xk = slice(k * nx, (k + 1) * nx)
        xkp1 = slice((k + 1) * nx, (k + 2) * nx)
        uk0 = n_xvars + k * nu
        uk = slice(uk0, uk0 + nu)

        Aeq[rows, xk] = -A[k]
        Aeq[rows, xkp1] = np.eye(nx)
        Aeq[rows, uk] = -B[k]
        beq[rows] = c[k + 1]
        row += nx

    return Aeq.tocsc(), beq, n_fixed_x0


def _normalize_D(D, N: int, nc: int, nu: int):
    """Accept D with N or N+1 stages; terminal D is ignored/padded."""
    if D is None:
        return np.zeros((N + 1, nc, nu), dtype=np.float64)

    D = _np(D, "D")
    if D.shape == (N, nc, nu):
        D = np.concatenate(
            [D, np.zeros((1, nc, nu), dtype=D.dtype)], axis=0
        )
    if D.shape != (N + 1, nc, nu):
        raise ValueError(
            f"D must have shape {(N, nc, nu)} or {(N + 1, nc, nu)}; "
            f"got {D.shape}"
        )
    return D


def _assemble_inequalities(
    C,
    D,
    f,
    N: int,
    nx: int,
    nu: int,
    n_xvars: int,
    nz: int,
):
    """Build the already-linearized path constraints Cx + Du <= f."""
    if C is None or f is None:
        return sp.csc_matrix((0, nz)), np.empty(0), np.empty(0), 0

    C = _np(C, "C")
    f = _np(f, "f")
    _require_finite("C", C)
    if np.any(np.isnan(f)):
        raise ValueError("NaN found in linearized inequality bounds f.")

    if C.ndim != 3 or C.shape[0] != N + 1 or C.shape[2] != nx:
        raise ValueError(
            f"C must have shape (N+1,nc,nx) = ({N + 1},nc,{nx}); "
            f"got {C.shape}"
        )

    nc = C.shape[1]
    if f.shape != (N + 1, nc):
        raise ValueError(f"f must have shape {(N + 1, nc)}; got {f.shape}")

    D = _normalize_D(D, N, nc, nu)
    _require_finite("D", D)

    n_ineq = (N + 1) * nc
    Aineq = sp.lil_matrix((n_ineq, nz), dtype=np.float64)

    for k in range(N + 1):
        rows = slice(k * nc, (k + 1) * nc)
        xk = slice(k * nx, (k + 1) * nx)
        Aineq[rows, xk] = C[k]

        if k < N:
            uk0 = n_xvars + k * nu
            uk = slice(uk0, uk0 + nu)
            Aineq[rows, uk] = D[k]

    lower = np.full(n_ineq, -np.inf, dtype=np.float64)
    upper = f.reshape(-1)
    return Aineq.tocsc(), lower, upper, nc


def constrained_solve(
    cfg: ADMMConfig,
    Q,
    q,
    R,
    r,
    M,
    A,
    B,
    c,
    C,
    D,
    f,
    w=None,
    y=None,
    rho=None,
    rho_grad=None,
    Jh=None,
    a=None,
    b=None,
    L: int = 0,
):
    """Solve the linearized constrained horizon QP using OSQP.

    Parameters intentionally mirror the previous custom ``constrained_solve``.
    ``w``, ``y``, ``rho_grad``, ``Jh``, ``a``, ``b`` and ``L`` are accepted only
    so existing call sites can stay unchanged.  They are not used to build the
    OSQP problem.

    Returns
    -------
    tuple
        ``(x_bar, u_bar, v, w_bar, y_bar, rho_final, rho_grad_final,
        mu, a_bar, b_bar, converged)``

        This mirrors the previous return layout closely:

        * ``x_bar``: optimized states, shape (N+1, nx)
        * ``u_bar``: optimized controls, shape (N, nu)
        * ``v``: OSQP dynamics equality multipliers, padded at index 0
        * ``w_bar``: nominal constraint LHS Cx+Du (compatibility value)
        * ``y_bar``: scaled inequality dual mu/rho_estimate
        * ``rho_final``: OSQP's rho estimate when available
        * ``rho_grad_final``: same value; OSQP uses one global ADMM rho
        * ``mu``: inequality dual multipliers for the combined constraints
        * ``a_bar``: zeros, retained only for return-layout compatibility
        * ``b_bar``: zeros, retained only for return-layout compatibility
        * ``converged``: True for OSQP SOLVED or SOLVED INACCURATE

    Notes
    -----
    ``w_bar`` and ``y_bar`` are reconstructed from the single OSQP formulation.
    ``a_bar`` and ``b_bar`` are returned as zeros.  No ``Jh``/gradient-window
    term is included anywhere in the OSQP QP.
    """
    Q = _np(Q, "Q")
    q = _np(q, "q")
    R = _np(R, "R")
    r = _np(r, "r")
    M = _np(M, "M")
    A = _np(A, "A")
    B = _np(B, "B")
    c = _np(c, "c")

    # These arrays define the objective and equality constraints and therefore
    # must be finite.  (OSQP bounds may be +/-inf, but the matrix data may not.)
    for _name, _arr in (
        ("Q", Q), ("q", q), ("R", R), ("r", r), ("M", M),
        ("A", A), ("B", B), ("c", c),
    ):
        _require_finite(_name, _arr)

    N, nx, nu = _validate_shapes(Q, q, R, r, M, A, B, c)

    P, linear, n_xvars, nz = _assemble_objective(
        Q,
        q,
        R,
        r,
        M,
        float(getattr(cfg, "regularization", 1e-9)),
        bool(getattr(cfg, "convexify_hessian", True)),
        float(getattr(cfg, "psd_eigenvalue_floor", 1e-8)),
        bool(getattr(cfg, "report_hessian_shift", False)),
        bool(getattr(cfg, "normalize_objective", True)),
        bool(getattr(cfg, "report_qp_scaling", False)),
    )

    num_phases = int(getattr(cfg, "num_phases", 0))
    Aeq, beq, n_fixed_x0 = _assemble_equalities(
        A, B, c, num_phases, n_xvars, nz
    )

    Aineq, lineq, uineq, nc = _assemble_inequalities(
        C, D, f, N, nx, nu, n_xvars, nz
    )

    if Aineq.shape[0] > 0:
        A_osqp = sp.vstack([Aeq, Aineq], format="csc")
        lower = np.concatenate([beq, lineq])
        upper = np.concatenate([beq, uineq])
    else:
        A_osqp = Aeq
        lower = beq.copy()
        upper = beq.copy()

    solver = osqp.OSQP()
    setup_kwargs = dict(
        P=P,
        q=linear,
        A=A_osqp,
        l=lower,
        u=upper,
        verbose=bool(getattr(cfg, "verbose", False)),
        eps_abs=float(getattr(cfg, "eps_abs", 1e-2)),
        eps_rel=float(getattr(cfg, "eps_rel", 1e-2)),
        max_iter=int(getattr(cfg, "max_iterations", 400)),
        adaptive_rho=bool(getattr(cfg, "adaptive_rho", True)),
    )

    initial_rho = float(rho if rho is not None else getattr(cfg, "initial_rho", 1.0))
    if initial_rho > 0.0:
        setup_kwargs["rho"] = initial_rho

    # Your installed OSQP uses the 1.x ``polishing`` setting.  Do not catch
    # ValueError here: OSQPException subclasses ValueError, and catching it
    # would incorrectly retry a genuinely non-convex/numerically invalid QP.
    solver.setup(
        **setup_kwargs,
        polishing=bool(getattr(cfg, "polish", False)),
    )

    result = solver.solve()

    status_val = int(getattr(result.info, "status_val", -1))
    # OSQP status values 1 and 2 are SOLVED and SOLVED INACCURATE.
    converged = status_val in (1, 2)

    if result.x is None:
        # Preserve a useful failure return instead of crashing downstream code.
        z = np.zeros(nz, dtype=np.float64)
    else:
        z = np.asarray(result.x, dtype=np.float64)

    x_bar = z[:n_xvars].reshape(N + 1, nx)
    u_bar = z[n_xvars:].reshape(N, nu)

    # Equality multipliers.  v[1:] corresponds to the dynamics rows; v[0] is
    # left zero because only part of x0 may have an explicit equality row.
    v = np.zeros((N + 1, nx), dtype=np.float64)
    dual = (
        np.asarray(result.y, dtype=np.float64)
        if result.y is not None
        else np.zeros(A_osqp.shape[0], dtype=np.float64)
    )
    dyn_dual_start = n_fixed_x0
    dyn_dual_stop = dyn_dual_start + N * nx
    if dual.size >= dyn_dual_stop:
        v[1:] = dual[dyn_dual_start:dyn_dual_stop].reshape(N, nx)

    rho_estimate = float(
        getattr(result.info, "rho_estimate", initial_rho)
    )
    if not np.isfinite(rho_estimate) or rho_estimate <= 0.0:
        rho_estimate = initial_rho if initial_rho > 0.0 else 1.0

    if nc > 0:
        C_np = _np(C, "C")
        D_np = _normalize_D(D, N, nc, nu)
        u_pad = np.vstack([u_bar, np.zeros((1, nu), dtype=u_bar.dtype)])

        z_nominal = (
            np.einsum("tmi,ti->tm", C_np, x_bar)
            + np.einsum("tmi,ti->tm", D_np, u_pad)
        )

        # Jh/L are intentionally ignored by this backend.  Keep these outputs
        # only so the caller can use the previous constrained_solve return layout.
        compatibility_L = max(int(L), 0)
        a_bar = np.zeros((N + 1, compatibility_L, nc), dtype=np.float64)

        ineq_dual_start = Aeq.shape[0]
        mu = np.maximum(
            dual[ineq_dual_start:ineq_dual_start + (N + 1) * nc]
            .reshape(N + 1, nc),
            0.0,
        )

        y_bar = mu / rho_estimate
        b_bar = np.zeros_like(a_bar)

        w_bar = z_nominal
    else:
        w_bar = np.zeros((N + 1, 0), dtype=np.float64)
        y_bar = np.zeros_like(w_bar)
        mu = np.zeros_like(w_bar)
        a_bar = np.zeros((N + 1, max(int(L), 0), 0), dtype=np.float64)
        b_bar = np.zeros_like(a_bar)

    # OSQP has one ADMM penalty parameter, so expose its estimate through both
    # legacy rho outputs.
    rho_final = rho_estimate
    rho_grad_final = rho_estimate

    return (
        _to_output(x_bar),
        _to_output(u_bar),
        _to_output(v),
        _to_output(w_bar),
        _to_output(y_bar),
        rho_final,
        rho_grad_final,
        _to_output(mu),
        _to_output(a_bar),
        _to_output(b_bar),
        converged,
    )
