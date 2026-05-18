
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

import gpu_sls.gpu_sqp

@dataclass
class MPCConfig:
    n: int
    nu: int
    N: int
    W: jnp.ndarray
    u_ref: jnp.ndarray
    dt: float


class GenericMPC:
    def __init__(
        self,
        sls_config, sqp_config, admm_config,
        config, dynamics, constraints, obstacles,
        cost,
        num_constraints: int,
        disturbance,
        X_in, U_in,
        shift: int = 1,
        output_equation = None, output_uncertainty = None
    ):
        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        self.obstacles = obstacles
        num_obstacles = self.obstacles.shape[0]
        self.h_ct_ws = jnp.zeros((config.N + 1, num_constraints - num_obstacles))
        self.beta_ws = jnp.ones((config.N + 1, config.N + 1, num_constraints - num_obstacles)) * 1e-10
        self.mu_ws = jnp.zeros((config.N + 1, num_constraints))

        self.U0 = U_in
        self.X0 = X_in
        self.V0 = jnp.zeros((config.N + 1, config.n))
        self.w = jnp.zeros((config.N + 1, num_constraints))
        self.y = jnp.zeros((config.N + 1, num_constraints))
        # TODO: Make this a parameter
        self.rho = jnp.asarray(self.admm_config.initial_rho, dtype=self.w.dtype)

        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.disturbance = disturbance


        self.output_equation = output_equation
        self.output_uncertainty = output_uncertainty
        if output_equation is None or output_uncertainty is None:
            self.output_equation = lambda x, u, t: jnp.zeros_like(x)
            self.output_uncertainty = lambda x: jnp.zeros((x.shape[0], x.shape[1], x.shape[1]))

        work = partial(
            gpu_sls.gpu_sqp.sqp,
            self.sls_config, self.sqp_config, self.admm_config,
            cost, dynamics,
            None,
            constraints, disturbance, self.output_equation, self.output_uncertainty
        )
        self._solve = jax.jit(work)

    def run(self, x0: jnp.ndarray, reference: jnp.ndarray, Xi: jnp.ndarray, parameter: Any):
        (X, U, V, w, y, rho, backoffs, Phi_x,
         Phi_u, Phi_xw, Phi_uw, Phi_xe, Phi_ue, betaN, muN) = self._solve(
            reference,
            parameter,
            self.config.W,
            x0, self.X0, self.U0, self.V0,
            self.w, self.y, self.rho,
            self.obstacles,
            self.h_ct_ws, self.beta_ws, self.mu_ws, Xi
        )


        s = self.shift

        invalid = (
            jnp.any(~jnp.isfinite(U)) |
            jnp.any(~jnp.isfinite(X)) |
            jnp.any(~jnp.isfinite(V)) |
            jnp.any(~jnp.isfinite(w)) |
            jnp.any(~jnp.isfinite(y)) |
            jnp.any(~jnp.isfinite(backoffs)) |
            jnp.any(~jnp.isfinite(betaN)) |
            jnp.any(~jnp.isfinite(muN)) |
            jnp.any(~jnp.isfinite(Phi_x)) |
            jnp.any(~jnp.isfinite(Phi_u))
        )

        def shift_and_pad(arr, pad_value=None):
            if pad_value is None:
                tail = jnp.tile(arr[-1:], (s,) + (1,) * (arr.ndim - 1))
            else:
                tail = jnp.broadcast_to(
                    pad_value,
                    (s,) + arr.shape[1:]
                )
            return jnp.concatenate([arr[s:], tail], axis=0)

        rho = jnp.asarray(rho, dtype=self.rho.dtype)

        self.U0 = jnp.where(
            invalid,
            jnp.tile(self.config.u_ref, (self.config.N, 1)),
            shift_and_pad(U),
        )

        self.X0 = jnp.where(
            invalid,
            jnp.tile(x0, (self.config.N + 1, 1)),
            shift_and_pad(X),
        )

        self.V0 = jnp.where(
            invalid,
            jnp.zeros_like(self.V0),
            shift_and_pad(V),
        )

        self.h_ct_ws = jnp.where(
            invalid,
            jnp.zeros_like(self.h_ct_ws),
            shift_and_pad(backoffs),
        )

        self.beta_ws = jnp.where(
            invalid,
            jnp.ones_like(self.beta_ws) * jnp.asarray(1e-10, dtype=self.beta_ws.dtype),
            shift_and_pad(betaN),
        )

        self.mu_ws = jnp.where(
            invalid,
            jnp.zeros_like(self.mu_ws),
            shift_and_pad(muN),
        )

        self.w = jnp.where(
            invalid,
            jnp.zeros_like(self.w),
            shift_and_pad(w),
        )

        self.y = jnp.where(
            invalid,
            jnp.zeros_like(self.y),
            shift_and_pad(y),
        )

        self.y = jnp.where(
            invalid,
            self.y,
            rho / self.rho * self.y,
        )

        self.rho = jnp.where(
            invalid,
            jnp.asarray(self.admm_config.initial_rho, dtype=self.rho.dtype),
            rho,
        )

        return U[0], X, U, V, backoffs, Phi_x, Phi_u, Phi_xw, Phi_uw, Phi_xe, Phi_ue