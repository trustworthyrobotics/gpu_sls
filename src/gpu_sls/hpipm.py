"""HPIPM backend for an already-linearized finite-horizon constrained control QP.

This is a drop-in-style replacement for the previous custom ADMM/LQR or OSQP
``constrained_solve`` backend.  The nonlinear/SQP layer must already have built
all QP matrices before this function is called.  This module only packs those
matrices into HPIPM's OCP-QP representation, partially condenses the horizon,
then solves the condensed QP and expands the solution back to the original horizon.

QP convention
-------------
For k = 0, ..., N-1:

    min  sum_k [
             0.5 x_k' Q_k x_k + q_k' x_k
           + 0.5 u_k' R_k u_k + r_k' u_k
           +       x_k' M_k u_k
         ]
         + 0.5 x_N' Q_N x_N + q_N' x_N

subject to

    x_{k+1} = A_k x_k + B_k u_k + c_{k+1}
    C_k x_k + D_k u_k <= f_k

HPIPM stores the cross term as u' S x, hence S_k = M_k.T.

``Jh`` and ``L`` are accepted only for compatibility with the previous solver
API.  They are NOT used in the HPIPM problem.

Initial-state convention
------------------------
``c[0]`` stores x_0.  If ``cfg.num_phases > 0``, the last ``num_phases``
entries of x_0 are left free; only the preceding physical-state entries are
fixed.

Important
---------
The Python HPIPM wrapper is CPU/host code and must receive concrete NumPy/JAX
arrays, not JAX tracers.  Call this backend through the same host callback path
used for the OSQP backend if the outer SQP is jitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ctypes
import time

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

try:
    from hpipm_python import (
        hpipm_ocp_qp_dim,
        hpipm_ocp_qp,
        hpipm_ocp_qp_sol,
        hpipm_ocp_qp_solver_arg,
        hpipm_ocp_qp_solver,
    )
    _HPIPM_IMPORT_ERROR = None
except Exception as exc:  # defer failure until solve is actually requested
    hpipm_ocp_qp_dim = None
    hpipm_ocp_qp = None
    hpipm_ocp_qp_sol = None
    hpipm_ocp_qp_solver_arg = None
    hpipm_ocp_qp_solver = None
    _HPIPM_IMPORT_ERROR = exc


@dataclass(frozen=True)
class ADMMConfig:
    """Compatibility config plus HPIPM-specific settings.

    The legacy names are retained so existing caller code can keep constructing
    ``ADMMConfig``.  HPIPM does not use ADMM rho internally.
    """

    # Legacy compatibility fields.
    rho_update_frequency: int = 25
    max_iterations: int = 100
    eps_abs: float = 1e-4                 # compatibility fallback only
    eps_rel: float = 1e-4                 # compatibility fallback only
    eps_abs_grad: float = 1e-4
    eps_rel_grad: float = 1e-4
    rho_max: float = 1e5
    initial_rho: float = 1.0
    regularized_rho_update: bool = False
    num_phases: int = 1

    # HPIPM IPM settings.
    mode: str = "robust"                   # speed_abs | speed | balance | robust
    tol_stat: float = 1e-2
    tol_eq: float = 1e-2
    tol_ineq: float = 1e-2
    tol_comp: float = 1e-2
    tol_dual_gap: float = 1e-2
    mu0: float = 1e4
    reg_prim: float = 1e-2
    warm_start: bool = True
    verbose: bool = True

    # Convex-QP handling. HPIPM is a convex QP solver too.
    regularization: float = 1e-9
    convexify_hessian: bool = True
    psd_eigenvalue_floor: float = 1e-8
    report_hessian_shift: bool = False

    # Multiply the entire objective by one positive scalar.  This leaves the
    # minimizer unchanged and can improve conditioning.
    normalize_objective: bool = True
    report_qp_scaling: bool = False

    # Retained for API compatibility.  One-sided constraints are represented
    # with HPIPM's bound masks, not with a large artificial lower bound.
    hpipm_infinity: float = 1e15

    # Native HPIPM partial condensing.  The original N-stage OCP-QP is condensed
    # to ``condensed_horizon`` stages, solved, and then expanded back to N stages.
    partial_condensing: bool = False
    condensed_horizon: int = 20
    condensing_ric_alg: int = 0          # 0 classical, 1 square-root
    report_condensing: bool = False


HPIPMConfig = ADMMConfig


def _np(x: Any, name: str) -> np.ndarray:
    """Convert a concrete NumPy/JAX array to float64 NumPy."""
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a concrete NumPy/JAX array. "
            "HPIPM cannot consume a JAX tracer directly inside jax.jit."
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
            "Fix this upstream before calling HPIPM."
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
    """Validate Cx + Du <= f.  Jh is intentionally absent."""
    if C is None or f is None:
        return None, None, None, 0

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
        raise ValueError("-inf found in upper inequality bounds f; QP is invalid/infeasible.")

    D = _normalize_D(D, N, nc, nu)
    _require_finite("D", D)
    return C, D, f, nc


def _prepare_objective(Q, q, R, r, M, cfg: ADMMConfig):
    """Symmetrize, optionally convexify, regularize, and scale the supplied QP.

    HPIPM's stage Hessian is ordered [u; x]:

        [[R, S],
         [S.T, Q]],      with S = M.T.

    This is permutation-similar to [[Q,M],[M.T,R]], so it has the same
    eigenvalues.  No nonlinear model is evaluated here.
    """
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

    for k in range(N):
        Hk = np.block([[Qs[k], Ms[k]], [Ms[k].T, Rs[k]]])
        lam_min = float(np.linalg.eigvalsh(Hk)[0])
        worst_before = min(worst_before, lam_min)
        if lam_min < floor:
            if not convexify:
                raise ValueError(
                    f"HPIPM requires a convex QP, but stage {k} joint Hessian "
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
                f"HPIPM requires a convex QP, but terminal Q has "
                f"lambda_min={terminal_min:.3e}."
            )
        shift = floor - terminal_min
        Qs[N] += shift * np.eye(nx)
        max_shift = max(max_shift, shift)

    reg = max(float(getattr(cfg, "regularization", 1e-9)), 0.0)
    if reg > 0.0:
        for k in range(N):
            Qs[k] += reg * np.eye(nx)
            Rs[k] += reg * np.eye(nu)
        Qs[N] += reg * np.eye(nx)

    if bool(getattr(cfg, "report_hessian_shift", False)) and max_shift > 0.0:
        print(
            "HPIPM convexified supplied linearized QP Hessian: "
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
            "HPIPM QP scaling: "
            f"objective_scale={objective_scale:.3e}, "
            f"max|Q|={np.max(np.abs(Qs)):.3e}, "
            f"max|R|={np.max(np.abs(Rs)):.3e}, "
            f"max|M|={np.max(np.abs(Ms)):.3e}"
        )

    return Qs, qs, Rs, rs, Ms


def _require_hpipm() -> None:
    if _HPIPM_IMPORT_ERROR is not None:
        raise ImportError(
            "Could not import hpipm_python. Install HPIPM/BLASFEO shared "
            "libraries and the HPIPM Python wrapper, and ensure libhpipm.so "
            "and libblasfeo.so are visible through LD_LIBRARY_PATH."
        ) from _HPIPM_IMPORT_ERROR



class _DPartCondQPArg(ctypes.Structure):
    """ctypes mirror of ``struct d_part_cond_qp_arg`` from HPIPM."""

    _fields_ = [
        ("cond_arg", ctypes.c_void_p),
        ("N2", ctypes.c_int),
        ("memsize", ctypes.c_size_t),
    ]


class _DPartCondQPWs(ctypes.Structure):
    """ctypes mirror of ``struct d_part_cond_qp_ws`` from HPIPM."""

    _fields_ = [
        ("cond_workspace", ctypes.c_void_p),
        ("memsize", ctypes.c_size_t),
    ]


class _PartialCondensingContext:
    """Own native HPIPM partial-condensing memory for one QP solve.

    HPIPM's current Python wrapper exposes the OCP-QP solver but not a Python
    class for the partial-condensing module.  The shared library nevertheless
    exports the native ``d_part_cond_qp_*`` API, so this context calls that API
    through ctypes while reusing the standard Python wrapper's ``dim_struct``,
    ``qp_struct`` and ``qp_sol_struct`` pointers.
    """

    def __init__(self, dim, qp, N: int, N2: int, ric_alg: int = 0):
        if not (1 <= N2 <= N):
            raise ValueError(f"condensed_horizon must satisfy 1 <= N2 <= N={N}; got {N2}")

        self.N = int(N)
        self.N2 = int(N2)
        self.lib = ctypes.CDLL("libhpipm.so")
        lib = self.lib

        cint_p = ctypes.POINTER(ctypes.c_int)
        arg_p = ctypes.POINTER(_DPartCondQPArg)
        ws_p = ctypes.POINTER(_DPartCondQPWs)
        void_p = ctypes.c_void_p

        # Declare the native API explicitly.  This is important on 64-bit
        # systems so size_t and pointer return/argument values are not truncated.
        lib.d_part_cond_qp_compute_block_size.argtypes = [
            ctypes.c_int, ctypes.c_int, cint_p
        ]
        lib.d_part_cond_qp_compute_block_size.restype = None

        lib.d_part_cond_qp_compute_dim.argtypes = [void_p, cint_p, void_p]
        lib.d_part_cond_qp_compute_dim.restype = None

        lib.d_part_cond_qp_arg_memsize.argtypes = [ctypes.c_int]
        lib.d_part_cond_qp_arg_memsize.restype = ctypes.c_size_t
        lib.d_part_cond_qp_arg_create.argtypes = [
            ctypes.c_int, arg_p, void_p
        ]
        lib.d_part_cond_qp_arg_create.restype = None
        lib.d_part_cond_qp_arg_set_default.argtypes = [arg_p]
        lib.d_part_cond_qp_arg_set_default.restype = None
        lib.d_part_cond_qp_arg_set_ric_alg.argtypes = [ctypes.c_int, arg_p]
        lib.d_part_cond_qp_arg_set_ric_alg.restype = None
        lib.d_part_cond_qp_arg_set_comp_prim_sol.argtypes = [ctypes.c_int, arg_p]
        lib.d_part_cond_qp_arg_set_comp_prim_sol.restype = None
        lib.d_part_cond_qp_arg_set_comp_dual_sol_eq.argtypes = [ctypes.c_int, arg_p]
        lib.d_part_cond_qp_arg_set_comp_dual_sol_eq.restype = None
        lib.d_part_cond_qp_arg_set_comp_dual_sol_ineq.argtypes = [ctypes.c_int, arg_p]
        lib.d_part_cond_qp_arg_set_comp_dual_sol_ineq.restype = None

        lib.d_part_cond_qp_ws_memsize.argtypes = [
            void_p, cint_p, void_p, arg_p
        ]
        lib.d_part_cond_qp_ws_memsize.restype = ctypes.c_size_t
        lib.d_part_cond_qp_ws_create.argtypes = [
            void_p, cint_p, void_p, arg_p, ws_p, void_p
        ]
        lib.d_part_cond_qp_ws_create.restype = None

        lib.d_part_cond_qp_cond.argtypes = [void_p, void_p, arg_p, ws_p]
        lib.d_part_cond_qp_cond.restype = None
        lib.d_part_cond_qp_expand_sol.argtypes = [
            void_p, void_p, void_p, arg_p, ws_p
        ]
        lib.d_part_cond_qp_expand_sol.restype = None

        # HPIPM chooses balanced block sizes whose sum is the original N.
        self.block_size = np.zeros(N2 + 1, dtype=np.intc)
        self.block_ptr = self.block_size.ctypes.data_as(cint_p)
        lib.d_part_cond_qp_compute_block_size(N, N2, self.block_ptr)

        if int(np.sum(self.block_size)) != N:
            raise RuntimeError(
                "HPIPM returned an invalid partial-condensing block partition: "
                f"block_size={self.block_size.tolist()}, sum={int(np.sum(self.block_size))}, N={N}."
            )

        # Allocate an ordinary Python-wrapper OCP dimension object, then let the
        # native partial-condensing routine fill its per-stage dimensions.
        self.dim2 = hpipm_ocp_qp_dim(N2)
        lib.d_part_cond_qp_compute_dim(
            dim.dim_struct, self.block_ptr, self.dim2.dim_struct
        )

        self.qp2 = hpipm_ocp_qp(self.dim2)
        self.qp_sol2 = hpipm_ocp_qp_sol(self.dim2)

        # Condensing arguments and their internal native memory.  Keep every
        # buffer as an instance member so Python cannot garbage-collect it while
        # HPIPM still holds pointers into it.
        arg_memsize = int(lib.d_part_cond_qp_arg_memsize(N2))
        self.part_arg_mem = ctypes.create_string_buffer(arg_memsize)
        self.part_arg = _DPartCondQPArg()
        lib.d_part_cond_qp_arg_create(
            N2,
            ctypes.byref(self.part_arg),
            ctypes.cast(self.part_arg_mem, void_p),
        )
        lib.d_part_cond_qp_arg_set_default(ctypes.byref(self.part_arg))
        lib.d_part_cond_qp_arg_set_ric_alg(
            int(ric_alg), ctypes.byref(self.part_arg)
        )

        # We need the expanded primal, dynamics duals, and inequality duals to
        # preserve the legacy constrained_solve return tuple.
        lib.d_part_cond_qp_arg_set_comp_prim_sol(1, ctypes.byref(self.part_arg))
        lib.d_part_cond_qp_arg_set_comp_dual_sol_eq(1, ctypes.byref(self.part_arg))
        lib.d_part_cond_qp_arg_set_comp_dual_sol_ineq(1, ctypes.byref(self.part_arg))

        ws_memsize = int(
            lib.d_part_cond_qp_ws_memsize(
                dim.dim_struct,
                self.block_ptr,
                self.dim2.dim_struct,
                ctypes.byref(self.part_arg),
            )
        )
        self.part_ws_mem = ctypes.create_string_buffer(ws_memsize)
        self.part_ws = _DPartCondQPWs()
        lib.d_part_cond_qp_ws_create(
            dim.dim_struct,
            self.block_ptr,
            self.dim2.dim_struct,
            ctypes.byref(self.part_arg),
            ctypes.byref(self.part_ws),
            ctypes.cast(self.part_ws_mem, void_p),
        )

        self.original_qp = qp

    def condense(self) -> None:
        self.lib.d_part_cond_qp_cond(
            self.original_qp.qp_struct,
            self.qp2.qp_struct,
            ctypes.byref(self.part_arg),
            ctypes.byref(self.part_ws),
        )

    def expand(self, original_sol) -> None:
        self.lib.d_part_cond_qp_expand_sol(
            self.original_qp.qp_struct,
            self.qp_sol2.qp_sol_struct,
            original_sol.qp_sol_struct,
            ctypes.byref(self.part_arg),
            ctypes.byref(self.part_ws),
        )


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
    """Solve the already-linearized horizon QP with HPIPM.

    ``Jh`` and ``L`` are accepted for call-site compatibility and are ignored.
    No nonlinear dynamics/constraint evaluation or linearization occurs here.

    Returns the legacy layout:

      (x_bar, u_bar, v, w_bar, y_bar, rho_final, rho_grad_final,
       mu, a_bar, b_bar, converged)
    """
    _require_hpipm()

    Q = _np(Q, "Q")
    q = _np(q, "q")
    R = _np(R, "R")
    r = _np(r, "r")
    M = _np(M, "M")
    A = _np(A, "A")
    B = _np(B, "B")
    c = _np(c, "c")

    for name, arr in (
        ("Q", Q), ("q", q), ("R", R), ("r", r), ("M", M),
        ("A", A), ("B", B), ("c", c),
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

    # ------------------------------------------------------------------
    # HPIPM OCP-QP dimensions.
    # ------------------------------------------------------------------
    dim = hpipm_ocp_qp_dim(N)
    dim.set("nx", nx, 0, N)
    if N > 0:
        dim.set("nu", nu, 0, N - 1)
    if n_fixed_x0 > 0:
        dim.set("nbx", n_fixed_x0, 0)
    if nc > 0:
        dim.set("ng", nc, 0, N)

    qp = hpipm_ocp_qp(dim)

    # ------------------------------------------------------------------
    # Dynamics: HPIPM uses x_{k+1} = A x_k + B u_k + b.
    # ------------------------------------------------------------------
    for k in range(N):
        qp.set("A", A[k], k)
        qp.set("B", B[k], k)
        qp.set("b", c[k + 1].reshape(nx, 1), k)

    # ------------------------------------------------------------------
    # Objective. HPIPM uses u' S x, so S = M.T.
    # ------------------------------------------------------------------
    for k in range(N):
        qp.set("Q", Qs[k], k)
        qp.set("R", Rs[k], k)
        qp.set("S", Ms[k].T, k)
        qp.set("q", qs[k].reshape(nx, 1), k)
        qp.set("r", rs[k].reshape(nu, 1), k)

    qp.set("Q", Qs[N], N)
    qp.set("q", qs[N].reshape(nx, 1), N)

    # ------------------------------------------------------------------
    # Fix only the physical part of x0. Phase-duration perturbations remain free.
    # ------------------------------------------------------------------
    if n_fixed_x0 > 0:
        Jx0 = np.zeros((n_fixed_x0, nx), dtype=np.float64)
        Jx0[np.arange(n_fixed_x0), np.arange(n_fixed_x0)] = 1.0
        x0_fixed = c[0, :n_fixed_x0].reshape(n_fixed_x0, 1)
        qp.set("Jx", Jx0, 0)
        qp.set("lx", x0_fixed, 0)
        qp.set("ux", x0_fixed, 0)

    # ------------------------------------------------------------------
    # General inequalities: -inf <= Cx + Du <= f.
    #
    # HPIPM has explicit masks for inactive sides of a bound.  Using -1e15 as
    # a surrogate lower bound looks harmless, but it leaves that side active
    # in the interior-point complementarity equations.  The resulting huge
    # slack prevents the complementarity residual from converging (even for a
    # tiny, otherwise trivial QP) and HPIPM returns MAX_ITER with a usable but
    # officially invalid solution.  Disable the absent lower sides instead.
    # Likewise, mask +inf upper bounds instead of replacing them by 1e15.
    # ------------------------------------------------------------------
    if nc > 0:
        inactive_lower = np.zeros((nc, 1), dtype=np.float64)
        lower_mask = np.zeros((nc, 1), dtype=np.float64)
        for k in range(N + 1):
            upper_k = np.asarray(f_np[k], dtype=np.float64).copy()
            upper_mask = np.isfinite(upper_k).astype(np.float64)
            upper_k[~np.isfinite(upper_k)] = 0.0
            upper_k = upper_k.reshape(nc, 1)
            upper_mask = upper_mask.reshape(nc, 1)

            qp.set("C", C_np[k], k)
            if k < N:
                qp.set("D", D_np[k], k)
            qp.set("lg", inactive_lower, k)
            qp.set("lg_mask", lower_mask, k)
            qp.set("ug", upper_k, k)
            qp.set("ug_mask", upper_mask, k)

    # ------------------------------------------------------------------
    # Native HPIPM partial condensing + IPM solve.
    # ------------------------------------------------------------------
    qp_sol = hpipm_ocp_qp_sol(dim)

    # The caller commonly passes the legacy gpu_admm.ADMMConfig, which has no
    # HPIPM-specific fields.  Match this module's declared robust defaults in
    # that case.  Partial condensing is an explicit speed opt-in: leaving the
    # original OCP structure intact is more reliable for the badly scaled,
    # highly constrained minimum-time SQP subproblems.
    use_partial_condensing = bool(getattr(cfg, "partial_condensing", False)) and N > 1
    cond_ctx = None
    solve_dim = dim
    solve_qp = qp
    solve_sol = qp_sol
    cond_time = 0.0
    expand_time = 0.0

    if use_partial_condensing:
        requested_N2 = int(getattr(cfg, "condensed_horizon", 20))
        N2 = min(max(requested_N2, 1), N)
        if N2 < N:
            cond_ctx = _PartialCondensingContext(
                dim,
                qp,
                N,
                N2,
                ric_alg=int(getattr(cfg, "condensing_ric_alg", 0)),
            )
            t0 = time.perf_counter()
            cond_ctx.condense()
            cond_time = time.perf_counter() - t0
            solve_dim = cond_ctx.dim2
            solve_qp = cond_ctx.qp2
            solve_sol = cond_ctx.qp_sol2

    mode = str(getattr(cfg, "mode", "robust"))
    arg = hpipm_ocp_qp_solver_arg(solve_dim, mode)

    arg.set("iter_max", int(getattr(cfg, "max_iterations", 100)))
    arg.set("tol_stat", float(getattr(cfg, "tol_stat", 1e-4)))
    arg.set("tol_eq", float(getattr(cfg, "tol_eq", 1e-5)))
    arg.set("tol_ineq", float(getattr(cfg, "tol_ineq", 1e-5)))
    arg.set("tol_comp", float(getattr(cfg, "tol_comp", 1e-5)))
    try:
        arg.set("tol_dual_gap", float(getattr(cfg, "tol_dual_gap", 1e-5)))
    except NameError:
        pass
    arg.set("mu0", float(getattr(cfg, "mu0", 1e4)))
    arg.set("reg_prim", float(getattr(cfg, "reg_prim", 1e-12)))
    arg.set("warm_start", int(bool(getattr(cfg, "warm_start", False))))

    solver = hpipm_ocp_qp_solver(solve_dim, arg)
    solve_t0 = time.perf_counter()
    solver.solve(solve_qp, solve_sol)
    solve_time = time.perf_counter() - solve_t0

    status = int(solver.get("status"))
    converged = status == 0

    if cond_ctx is not None:
        t0 = time.perf_counter()
        cond_ctx.expand(qp_sol)
        expand_time = time.perf_counter() - t0

    if bool(getattr(cfg, "report_condensing", False)):
        if cond_ctx is None:
            print(
                f"HPIPM partial condensing: disabled/effective N2=N={N}; "
                f"solve={1e3*solve_time:.3f} ms"
            )
        else:
            print(
                "HPIPM partial condensing: "
                f"N={N} -> N2={cond_ctx.N2}, "
                f"blocks={cond_ctx.block_size.tolist()}, "
                f"condense={1e3*cond_time:.3f} ms, "
                f"solve={1e3*solve_time:.3f} ms, "
                f"expand={1e3*expand_time:.3f} ms, "
                f"total={1e3*(cond_time+solve_time+expand_time):.3f} ms"
            )

    if bool(getattr(cfg, "verbose", False)):
        res_stat = float(solver.get("max_res_stat"))
        res_eq = float(solver.get("max_res_eq"))
        res_ineq = float(solver.get("max_res_ineq"))
        res_comp = float(solver.get("max_res_comp"))
        iters = int(solver.get("iter"))
        print(
            "HPIPM status="
            f"{status}, iter={iters}, stat={res_stat:.3e}, eq={res_eq:.3e}, "
            f"ineq={res_ineq:.3e}, comp={res_comp:.3e}"
        )

    # ------------------------------------------------------------------
    # Extract primal and dual solution.
    # ------------------------------------------------------------------
    x_bar = np.zeros((N + 1, nx), dtype=np.float64)
    u_bar = np.zeros((N, nu), dtype=np.float64)
    for k in range(N + 1):
        x_bar[k] = np.asarray(qp_sol.get("x", k)).reshape(-1)
    for k in range(N):
        u_bar[k] = np.asarray(qp_sol.get("u", k)).reshape(-1)

    # Dynamics multipliers: HPIPM pi[k] belongs to transition k -> k+1.
    v = np.zeros((N + 1, nx), dtype=np.float64)
    for k in range(N):
        v[k + 1] = np.asarray(qp_sol.get("pi", k)).reshape(-1)

    if nc > 0:
        u_pad = np.vstack([u_bar, np.zeros((1, nu), dtype=np.float64)])
        w_bar = (
            np.einsum("tmi,ti->tm", C_np, x_bar)
            + np.einsum("tmi,ti->tm", D_np, u_pad)
        )

        mu = np.zeros((N + 1, nc), dtype=np.float64)
        for k in range(N + 1):
            # Only the upper general bound is meaningful in our one-sided QP.
            mu[k] = np.maximum(
                np.asarray(qp_sol.get("lam_ug", k)).reshape(-1), 0.0
            )
    else:
        w_bar = np.zeros((N + 1, 0), dtype=np.float64)
        mu = np.zeros_like(w_bar)

    # Legacy compatibility only: HPIPM has no ADMM rho or scaled dual y.
    rho_compat = float(
        rho if rho is not None else getattr(cfg, "initial_rho", 1.0)
    )
    if not np.isfinite(rho_compat) or rho_compat <= 0.0:
        rho_compat = 1.0
    y_bar = mu / rho_compat

    compatibility_L = max(int(L), 0)
    a_bar = np.zeros((N + 1, compatibility_L, nc), dtype=np.float64)
    b_bar = np.zeros_like(a_bar)

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
