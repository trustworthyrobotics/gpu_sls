"""FATROP backend for an already-linearized finite-horizon constrained control QP.

This mirrors the API/conventions of the HPIPM ``constrained_solve`` backend:

    min sum_{k=0}^{N-1} [
          0.5 x_k' Q_k x_k + q_k' x_k
        + 0.5 u_k' R_k u_k + r_k' u_k
        +       x_k' M_k u_k
    ]
        + 0.5 x_N' Q_N x_N + q_N' x_N

subject to

    x_{k+1} = A_k x_k + B_k u_k + c_{k+1}
    C_k x_k + D_k u_k <= f_k

``c[0]`` stores the initial state.  If ``cfg.num_phases > 0``, the final
``num_phases`` components of x_0 are left free; only the physical-state part
is fixed.  This is useful for free-time/minimum-time SQP formulations where
phase-duration perturbations are appended to the state.

The nonlinear/SQP layer must already have produced all linearized QP arrays.
No MJX/JAX dynamics or constraint function is evaluated in this module.

Implementation notes
--------------------
CasADi's FATROP conic interface requires

  decision ordering:   [x0, u0, x1, u1, ..., xN]
  constraint ordering: [gap0, lincon0, gap1, lincon1, ..., linconN]

where each gap block has sparsity [A_k B_k -I].  This module packs the given
arrays into exactly that representation and uses ``casadi.conic(..., 'fatrop')``.

Because FATROP's conic interface solves a strictly convex QP, the supplied
stage Hessians are optionally convexified and receive a small positive
regularization, matching the spirit of the HPIPM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import time

import numpy as np
import scipy.sparse as sp

try:
    import casadi as ca
    _CASADI_IMPORT_ERROR = None
except Exception as exc:  # defer failure until solve is requested
    ca = None
    _CASADI_IMPORT_ERROR = exc

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None




@dataclass(frozen=True)
class ADMMConfig:
    """Compatibility config plus FATROP-specific settings.

    The legacy field names are retained so this backend can be swapped into
    callers that currently construct ``ADMMConfig`` for another backend.
    """

    # Legacy compatibility fields.
    rho_update_frequency: int = 25
    max_iterations: int = 100
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    eps_abs_grad: float = 1e-4
    eps_rel_grad: float = 1e-4
    rho_max: float = 1e5
    initial_rho: float = 1.0
    regularized_rho_update: bool = False
    num_phases: int = 1

    # FATROP settings.
    fatrop_tol: float = 1e-1
    mu_init: float = 0.1
    print_level: int = 0
    warm_start: bool = True
    verbose: bool = True
    error_on_fail: bool = False

    # Convex-QP handling.
    regularization: float = 1e-9
    convexify_hessian: bool = True
    psd_eigenvalue_floor: float = 1e-8
    report_hessian_shift: bool = False

    # Positive objective scaling does not alter the minimizer and can improve
    # numerical conditioning on minimum-time QPs with very different units.
    normalize_objective: bool = True
    report_qp_scaling: bool = False


FATROPConfig = ADMMConfig


# CasADi solver objects are expensive to construct.  The sparsity pattern of
# this backend depends only on (N, nx, nu, nc), so cache one solver per shape
# and FATROP option set.  Numerical QP matrices are supplied at every call.
_SOLVER_CACHE: Dict[Tuple[Any, ...], Any] = {}


import os
import sys
import ctypes
from contextlib import contextmanager


@contextmanager
def suppress_native_stdout():
    """Suppress C/C++ printf output to stdout."""
    sys.stdout.flush()

    # Flush C stdio as well
    libc = ctypes.CDLL(None)
    libc.fflush(None)

    old_stdout = os.dup(1)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            yield
    finally:
        libc.fflush(None)
        os.dup2(old_stdout, 1)
        os.close(old_stdout)

def _np(x: Any, name: str) -> np.ndarray:
    """Convert a concrete NumPy/JAX array to float64 NumPy."""
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a concrete NumPy/JAX array. "
            "FATROP/CasADi cannot consume a JAX tracer directly inside jax.jit. "
            "Call this backend through a host callback if the outer SQP is jitted."
        ) from exc


def _to_output(x: np.ndarray):
    if jnp is not None:
        return jnp.asarray(x)
    return x


def _require_finite(name: str, x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        bad = np.argwhere(~np.isfinite(x))
        idx = tuple(bad[0]) if bad.size else ()
        val = x[idx] if idx else np.nan
        raise ValueError(
            f"Non-finite value in linearized QP array {name} at {idx}: {val}. "
            "Fix this upstream before calling FATROP."
        )


def _validate_shapes(Q, q, R, r, M, A, B, c):
    N = A.shape[0]
    nx = A.shape[1]
    nu = B.shape[2]

    expected = {
        "A": (N, nx, nx),
        "B": (N, nx, nu),
        "Q": (N + 1, nx, nx),
        "q": (N + 1, nx),
        "R": (N, nu, nu),
        "r": (N, nu),
        "M": (N, nx, nu),
        "c": (N + 1, nx),
    }
    actual = {
        "A": A.shape,
        "B": B.shape,
        "Q": Q.shape,
        "q": q.shape,
        "R": R.shape,
        "r": r.shape,
        "M": M.shape,
        "c": c.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}; got {actual[name]}")

    return N, nx, nu


def _normalize_D(D, N: int, nc: int, nu: int) -> np.ndarray:
    if D is None:
        return np.zeros((N + 1, nc, nu), dtype=np.float64)

    D = _np(D, "D")
    if D.shape == (N, nc, nu):
        D = np.concatenate(
            [D, np.zeros((1, nc, nu), dtype=D.dtype)], axis=0
        )
    if D.shape != (N + 1, nc, nu):
        raise ValueError(
            f"D must have shape {(N, nc, nu)} or {(N + 1, nc, nu)}; got {D.shape}"
        )
    return D


def _prepare_inequalities(C, D, f, N: int, nx: int, nu: int):
    """Validate one-sided stage constraints Cx + Du <= f."""
    if C is None or f is None:
        return (
            np.zeros((N + 1, 0, nx), dtype=np.float64),
            np.zeros((N + 1, 0, nu), dtype=np.float64),
            np.zeros((N + 1, 0), dtype=np.float64),
            0,
        )

    C = _np(C, "C")
    f = _np(f, "f")
    _require_finite("C", C)

    if C.ndim != 3 or C.shape[0] != N + 1 or C.shape[2] != nx:
        raise ValueError(
            f"C must have shape (N+1,nc,nx) = ({N + 1},nc,{nx}); got {C.shape}"
        )

    nc = C.shape[1]
    if f.shape != (N + 1, nc):
        raise ValueError(f"f must have shape {(N + 1, nc)}; got {f.shape}")
    if np.any(np.isnan(f)):
        raise ValueError("NaN found in linearized inequality bounds f.")
    if np.any(np.isneginf(f)):
        raise ValueError(
            "-inf found in upper inequality bounds f; QP is invalid/infeasible."
        )

    D = _normalize_D(D, N, nc, nu)
    _require_finite("D", D)
    return C, D, f, nc


def _prepare_objective(Q, q, R, r, M, cfg):
    """Symmetrize, convexify, regularize, and optionally scale the QP."""
    N = R.shape[0]
    nx = Q.shape[1]
    nu = R.shape[1]

    Qs = 0.5 * (Q + np.swapaxes(Q, -1, -2))
    Rs = 0.5 * (R + np.swapaxes(R, -1, -2))
    Ms = np.array(M, copy=True)
    qs = np.array(q, copy=True)
    rs = np.array(r, copy=True)

    floor = max(float(getattr(cfg, "psd_eigenvalue_floor", 1e-8)), 0.0)
    convexify = bool(getattr(cfg, "convexify_hessian", True))
    worst_before = np.inf
    max_shift = 0.0

    # FATROP's conic interface expects a strictly convex QP.  Convexify each
    # OCP stage block H_k = [[Q_k, M_k], [M_k.T, R_k]].
    for k in range(N):
        Hk = np.block([[Qs[k], Ms[k]], [Ms[k].T, Rs[k]]])
        lam_min = float(np.linalg.eigvalsh(Hk)[0])
        worst_before = min(worst_before, lam_min)
        if lam_min < floor:
            if not convexify:
                raise ValueError(
                    f"FATROP requires a convex QP, but stage {k} joint Hessian "
                    f"has lambda_min={lam_min:.3e}. Enable convexify_hessian or "
                    "use a PSD/Gauss-Newton SQP Hessian upstream."
                )
            shift = floor - lam_min
            Qs[k] += shift * np.eye(nx)
            Rs[k] += shift * np.eye(nu)
            max_shift = max(max_shift, shift)

    terminal_min = float(np.linalg.eigvalsh(Qs[N])[0])
    worst_before = min(worst_before, terminal_min)
    if terminal_min < floor:
        if not convexify:
            raise ValueError(
                f"FATROP requires a convex QP, but terminal Q has "
                f"lambda_min={terminal_min:.3e}."
            )
        shift = floor - terminal_min
        Qs[N] += shift * np.eye(nx)
        max_shift = max(max_shift, shift)

    # Make the QP strictly positive definite even if floor was configured to 0.
    reg = max(float(getattr(cfg, "regularization", 1e-9)), 0.0)
    if reg > 0.0:
        for k in range(N):
            Qs[k] += reg * np.eye(nx)
            Rs[k] += reg * np.eye(nu)
        Qs[N] += reg * np.eye(nx)

    if bool(getattr(cfg, "report_hessian_shift", False)) and max_shift > 0.0:
        print(
            "FATROP convexified supplied linearized QP Hessian: "
            f"worst lambda_min={worst_before:.3e}, "
            f"max diagonal shift={max_shift:.3e}"
        )

    objective_scale = 1.0
    if bool(getattr(cfg, "normalize_objective", True)):
        maxima = [
            np.max(np.abs(Qs)) if Qs.size else 0.0,
            np.max(np.abs(Rs)) if Rs.size else 0.0,
            np.max(np.abs(Ms)) if Ms.size else 0.0,
            np.max(np.abs(qs)) if qs.size else 0.0,
            np.max(np.abs(rs)) if rs.size else 0.0,
        ]
        objective_scale = max(1.0, *(float(v) for v in maxima))
        if objective_scale > 1.0:
            inv = 1.0 / objective_scale
            Qs *= inv
            Rs *= inv
            Ms *= inv
            qs *= inv
            rs *= inv

    if bool(getattr(cfg, "report_qp_scaling", False)):
        print(
            "FATROP QP scaling: "
            f"objective_scale={objective_scale:.3e}, "
            f"max|Q|={np.max(np.abs(Qs)):.3e}, "
            f"max|R|={np.max(np.abs(Rs)):.3e}, "
            f"max|M|={np.max(np.abs(Ms)):.3e}"
        )

    return Qs, qs, Rs, rs, Ms


def _require_fatrop() -> None:
    if _CASADI_IMPORT_ERROR is not None:
        raise ImportError(
            "Could not import CasADi. Install a CasADi build containing the "
            "Conic::fatrop plugin."
        ) from _CASADI_IMPORT_ERROR

    plugins = ca.CasadiMeta_plugins()
    if "Conic::fatrop" not in plugins:
        raise RuntimeError(
            "This CasADi build does not contain the Conic::fatrop plugin. "
            "Install/build CasADi with FATROP support."
        )


def _layout(N: int, nx: int, nu: int):
    """Offsets for z = [x0,u0,x1,u1,...,x_{N-1},u_{N-1},xN]."""
    x_offsets = np.empty(N + 1, dtype=np.int64)
    u_offsets = np.empty(N, dtype=np.int64)
    cursor = 0
    for k in range(N):
        x_offsets[k] = cursor
        cursor += nx
        u_offsets[k] = cursor
        cursor += nu
    x_offsets[N] = cursor
    cursor += nx
    return x_offsets, u_offsets, cursor


def _append_dense_block(rows, cols, vals, r0: int, c0: int, block: np.ndarray):
    """Append a structurally dense block, retaining explicit numerical zeros."""
    nr, nc = block.shape
    rr = np.repeat(np.arange(r0, r0 + nr, dtype=np.int64), nc)
    cc = np.tile(np.arange(c0, c0 + nc, dtype=np.int64), nr)
    rows.extend(rr.tolist())
    cols.extend(cc.tolist())
    vals.extend(np.asarray(block, dtype=np.float64).reshape(-1).tolist())


def _assemble_qp(Q, q, R, r, M, A, B, c, C, D, f):
    """Pack stage arrays into CasADi FATROP conic matrices."""
    N, nx, _ = A.shape
    nu = B.shape[2]
    nc = C.shape[1]

    xoff, uoff, nz = _layout(N, nx, nu)

    # ------------------------------- Hessian -------------------------------
    h_rows = []
    h_cols = []
    h_vals = []
    g = np.zeros(nz, dtype=np.float64)

    for k in range(N):
        Hk = np.block([[Q[k], M[k]], [M[k].T, R[k]]])
        _append_dense_block(h_rows, h_cols, h_vals, int(xoff[k]), int(xoff[k]), Hk)
        g[xoff[k] : xoff[k] + nx] = q[k]
        g[uoff[k] : uoff[k] + nu] = r[k]

    _append_dense_block(
        h_rows, h_cols, h_vals, int(xoff[N]), int(xoff[N]), Q[N]
    )
    g[xoff[N] : xoff[N] + nx] = q[N]

    H_sp = sp.coo_matrix((h_vals, (h_rows, h_cols)), shape=(nz, nz)).tocsc()

    # ---------------------------- Linear constraints -----------------------
    # Order required by FATROP:
    #   [gap0, lincon0, gap1, lincon1, ..., gap_{N-1}, lincon_{N-1}, lincon_N]
    nrows = N * nx + (N + 1) * nc
    a_rows = []
    a_cols = []
    a_vals = []
    lba = np.empty(nrows, dtype=np.float64)
    uba = np.empty(nrows, dtype=np.float64)

    row = 0
    gap_slices = []
    lin_slices = []

    for k in range(N):
        # Raw conic equality for the desired dynamics:
        #
        #   x_{k+1} = A_k x_k + B_k u_k + c_{k+1}
        #
        # is packed as
        #
        #   -A_k x_k - B_k u_k + x_{k+1} = c_{k+1}.
        #
        # CasADi/FATROP recognizes this block as the stage dynamics gap.
        gap = slice(row, row + nx)
        gap_slices.append(gap)
        _append_dense_block(a_rows, a_cols, a_vals, row, int(xoff[k]), -A[k])
        _append_dense_block(a_rows, a_cols, a_vals, row, int(uoff[k]), -B[k])

        # IMPORTANT: next-state block must be a diagonal sparse +I, not dense.
        idx = np.arange(nx, dtype=np.int64)
        a_rows.extend((row + idx).tolist())
        a_cols.extend((int(xoff[k + 1]) + idx).tolist())
        a_vals.extend((np.ones(nx, dtype=np.float64)).tolist())

        rhs = c[k + 1]
        lba[gap] = rhs
        uba[gap] = rhs
        row += nx

        lin = slice(row, row + nc)
        lin_slices.append(lin)
        if nc > 0:
            _append_dense_block(a_rows, a_cols, a_vals, row, int(xoff[k]), C[k])
            _append_dense_block(a_rows, a_cols, a_vals, row, int(uoff[k]), D[k])
            lba[lin] = -np.inf
            uba[lin] = f[k]
        row += nc

    # Terminal stage has no control and no following dynamics gap.
    lin = slice(row, row + nc)
    lin_slices.append(lin)
    if nc > 0:
        _append_dense_block(a_rows, a_cols, a_vals, row, int(xoff[N]), C[N])
        lba[lin] = -np.inf
        uba[lin] = f[N]
    row += nc

    if row != nrows:
        raise RuntimeError(f"Internal FATROP packing error: row={row}, expected {nrows}")

    A_sp = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(nrows, nz)).tocsc()

    return (
        ca.DM(H_sp),
        g,
        ca.DM(A_sp),
        lba,
        uba,
        xoff,
        uoff,
        gap_slices,
        lin_slices,
    )


def _solver_key(cfg, N: int, nx: int, nu: int, nc: int):
    tol = float(getattr(cfg, "fatrop_tol", getattr(cfg, "eps_abs", 1e-4)))
    return (
        N,
        nx,
        nu,
        nc,
        int(getattr(cfg, "max_iterations", 100)),
        tol,
        float(getattr(cfg, "mu_init", 0.1)),
        int(getattr(cfg, "print_level", 0)),
        bool(getattr(cfg, "error_on_fail", False)),
    )


def _get_solver(cfg, H_dm, A_dm, N: int, nx: int, nu: int, nc: int):
    key = _solver_key(cfg, N, nx, nu, nc)
    solver = _SOLVER_CACHE.get(key)
    if solver is not None:
        return solver

    fatrop_tol = float(
        getattr(cfg, "fatrop_tol", getattr(cfg, "eps_abs", 1e-4))
    )
    fatrop_options = {
        "max_iter": int(getattr(cfg, "max_iterations", 100)),
        "tol": fatrop_tol,
        "mu_init": float(getattr(cfg, "mu_init", 0.1)),
        "print_level": int(getattr(cfg, "print_level", 0)),
    }

    opts = {
        "structure_detection": "manual",
        "N": int(N),
        "nx": [int(nx)] * (N + 1),
        # CasADi 3.7.x FatropConicInterface internally indexes nu[k] for
        # k=0,...,N when building the terminal Hessian block, so provide the
        # terminal zero-control entry explicitly.
        "nu": [int(nu)] * N + [0],
        "ng": [int(nc)] * (N + 1),
        "fatrop": fatrop_options,
        "error_on_fail": bool(getattr(cfg, "error_on_fail", False)),
        "record_time": True,
    }

    name = f"fatrop_ocp_qp_{len(_SOLVER_CACHE)}"
    solver = ca.conic(
        name,
        "fatrop",
        {"h": H_dm.sparsity(), "a": A_dm.sparsity()},
        opts,
    )
    _SOLVER_CACHE[key] = solver
    return solver


def _build_initial_guess(A, B, c, xoff, uoff, nz: int, nx: int, nu: int):
    """Construct a dynamics-consistent zero-control primal seed."""
    N = A.shape[0]
    z0 = np.zeros(nz, dtype=np.float64)
    xk = np.asarray(c[0], dtype=np.float64).copy()
    z0[xoff[0] : xoff[0] + nx] = xk
    for k in range(N):
        uk = np.zeros(nu, dtype=np.float64)
        z0[uoff[k] : uoff[k] + nu] = uk
        xk = A[k] @ xk + B[k] @ uk + c[k + 1]
        z0[xoff[k + 1] : xoff[k + 1] + nx] = xk
    return z0


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
    """Solve an already-linearized OCP-QP with FATROP.

    ``Jh`` and ``L`` are accepted for drop-in compatibility.  ``Jh`` is ignored;
    ``L`` only controls the shape of the legacy ``a_bar``/``b_bar`` outputs.

    Returns the same legacy layout as the HPIPM backend:

      (x_bar, u_bar, v, w_bar, y_bar, rho_final, rho_grad_final,
       mu, a_bar, b_bar, converged)
    """
    del Jh, a, b, rho_grad, w
    _require_fatrop()

    Q = _np(Q, "Q")
    q = _np(q, "q")
    R = _np(R, "R")
    r = _np(r, "r")
    M = _np(M, "M")
    A = _np(A, "A")
    B = _np(B, "B")
    c = _np(c, "c")

    for name, arr in (
        ("Q", Q),
        ("q", q),
        ("R", R),
        ("r", r),
        ("M", M),
        ("A", A),
        ("B", B),
        ("c", c),
    ):
        _require_finite(name, arr)

    N, nx, nu = _validate_shapes(Q, q, R, r, M, A, B, c)
    C_np, D_np, f_np, nc = _prepare_inequalities(C, D, f, N, nx, nu)
    Qs, qs, Rs, rs, Ms = _prepare_objective(Q, q, R, r, M, cfg)

    num_phases = int(getattr(cfg, "num_phases", 0))
    if not (0 <= num_phases <= nx):
        raise ValueError(
            f"num_phases must be between 0 and nx={nx}; got {num_phases}"
        )
    n_fixed_x0 = nx - num_phases

    (
        H_dm,
        g_np,
        A_dm,
        lba,
        uba,
        xoff,
        uoff,
        gap_slices,
        lin_slices,
    ) = _assemble_qp(Qs, qs, Rs, rs, Ms, A, B, c, C_np, D_np, f_np)

    nz = int(H_dm.size1())

    # Fix only the physical part of x0.  Free phase-duration components remain
    # unbounded so the SQP/QP can optimize them.
    lbx = -np.inf * np.ones(nz, dtype=np.float64)
    ubx = np.inf * np.ones(nz, dtype=np.float64)
    if n_fixed_x0 > 0:
        lbx[xoff[0] : xoff[0] + n_fixed_x0] = c[0, :n_fixed_x0]
        ubx[xoff[0] : xoff[0] + n_fixed_x0] = c[0, :n_fixed_x0]

    z0 = _build_initial_guess(A, B, c, xoff, uoff, nz, nx, nu)

    # Optional inequality-dual warm start from the legacy scaled ADMM dual y.
    lam_a0 = np.zeros(A_dm.size1(), dtype=np.float64)
    if bool(getattr(cfg, "warm_start", True)) and y is not None and nc > 0:
        try:
            y_np = _np(y, "y")
            if y_np.shape == (N + 1, nc):
                rho_compat = float(
                    rho if rho is not None else getattr(cfg, "initial_rho", 1.0)
                )
                if np.isfinite(rho_compat) and rho_compat > 0.0:
                    mu0 = np.maximum(rho_compat * y_np, 0.0)
                    for k, sl in enumerate(lin_slices):
                        lam_a0[sl] = mu0[k]
        except Exception:
            # A warm-start mismatch should not make the QP unsolvable.
            pass

    solver = _get_solver(cfg, H_dm, A_dm, N, nx, nu, nc)

    solve_t0 = time.perf_counter()
    # sol = solver(
    #     h=H_dm,
    #     g=g_np,
    #     a=A_dm,
    #     lba=lba,
    #     uba=uba,
    #     lbx=lbx,
    #     ubx=ubx,
    #     x0=z0,
    #     lam_a0=lam_a0,
    #     lam_x0=np.zeros(nz, dtype=np.float64),
    # )
    with suppress_native_stdout():
        sol = solver(
            h=H_dm,
            g=g_np,
            a=A_dm,
            lba=lba,
            uba=uba,
            lbx=lbx,
            ubx=ubx,
            x0=z0,
            lam_a0=lam_a0,
            lam_x0=np.zeros(nz, dtype=np.float64),
        )
    solve_time = time.perf_counter() - solve_t0

    stats = solver.stats()
    converged = bool(stats.get("success", False)) or str(
        stats.get("return_status", "")
    ).lower() in {"solved", "success", "solve_succeeded"}

    z = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
    lam_a = np.asarray(sol["lam_a"], dtype=np.float64).reshape(-1)

    # ----------------------------- Primal ----------------------------------
    x_bar = np.zeros((N + 1, nx), dtype=np.float64)
    u_bar = np.zeros((N, nu), dtype=np.float64)
    for k in range(N + 1):
        x_bar[k] = z[xoff[k] : xoff[k] + nx]
    for k in range(N):
        u_bar[k] = z[uoff[k] : uoff[k] + nu]

    # ------------------------------ Duals ----------------------------------
    # The conic gap is -A x - B u + x_next - c = 0, whereas HPIPM's
    # dynamics multiplier convention corresponds to A x + B u + c - x_next = 0.
    # Negate CasADi's row multiplier to preserve the HPIPM-style return convention.
    v = np.zeros((N + 1, nx), dtype=np.float64)
    for k, sl in enumerate(gap_slices):
        v[k + 1] = -lam_a[sl]

    if nc > 0:
        u_pad = np.vstack([u_bar, np.zeros((1, nu), dtype=np.float64)])
        w_bar = (
            np.einsum("tmi,ti->tm", C_np, x_bar)
            + np.einsum("tmi,ti->tm", D_np, u_pad)
        )

        # Our general constraints are one-sided upper bounds, so positive
        # CasADi lam_a values are the corresponding upper-bound multipliers.
        mu = np.zeros((N + 1, nc), dtype=np.float64)
        for k, sl in enumerate(lin_slices):
            mu[k] = np.maximum(lam_a[sl], 0.0)
    else:
        w_bar = np.zeros((N + 1, 0), dtype=np.float64)
        mu = np.zeros_like(w_bar)

    # Legacy compatibility: FATROP has no ADMM rho/scaled dual internally.
    rho_compat = float(
        rho if rho is not None else getattr(cfg, "initial_rho", 1.0)
    )
    if not np.isfinite(rho_compat) or rho_compat <= 0.0:
        rho_compat = 1.0
    y_bar = mu / rho_compat

    compatibility_L = max(int(L), 0)
    a_bar = np.zeros((N + 1, compatibility_L, nc), dtype=np.float64)
    b_bar = np.zeros_like(a_bar)

    if bool(getattr(cfg, "verbose", False)):
        status = stats.get("return_status", "unknown")
        iters = stats.get("iter_count", "?")
        print(
            f"FATROP status={status}, iter={iters}, "
            f"solve={1e3 * solve_time:.3f} ms"
        )

    rho_final = rho_compat
    rho_grad_final = rho_compat

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
