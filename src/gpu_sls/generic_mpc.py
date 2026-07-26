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


class GenericMPC:
    def __init__(
        self,
        sls_config,
        sqp_config,
        admm_config,
        config,
        dynamics,
        constraints,
        obstacles,
        cost,
        num_constraints: int,
        disturbance,
        X_in,
        U_in,
        shift: int = 1,
    ):
        self.sls_config = sls_config
        self.sqp_config = sqp_config
        self.admm_config = admm_config
        self.config = config
        self.shift = shift
        self.obstacles = obstacles

        num_obstacles = self.obstacles.shape[0]

        self.h_ct_ws = jnp.zeros(
            (
                config.N + 1,
                num_constraints - num_obstacles,
            )
        )

        self.beta_ws = (
            jnp.ones(
                (
                    config.N + 1,
                    config.N + 1,
                    num_constraints - num_obstacles,
                )
            )
            * 1e-10
        )

        self.mu_ws = jnp.zeros(
            (
                config.N + 1,
                num_constraints,
            )
        )

        self.Phi_x_ws = jnp.zeros(
            (
                config.N + 1,
                config.N + 1,
                config.n,
                config.n,
            )
        )

        self.Phi_u_ws = jnp.zeros(
            (
                config.N,
                config.N + 1,
                config.nu,
                config.n,
            )
        )

        self.converged_admm = False

        self.U0 = U_in
        self.X0 = X_in
        self.V0 = jnp.zeros((config.N + 1, config.n))

        self.w = jnp.zeros(
            (
                config.N + 1,
                num_constraints,
            )
        )

        self.y = jnp.zeros(
            (
                config.N + 1,
                num_constraints,
            )
        )

        self.rho = jnp.asarray(
            self.admm_config.initial_rho,
            dtype=self.w.dtype,
        )

        self.dynamics = dynamics
        self.constraints = constraints
        self.cost = cost
        self.disturbance = disturbance

        work = partial(
            gpu_sls.gpu_sqp.sqp,
            self.sls_config,
            self.sqp_config,
            self.admm_config,
            cost,
            dynamics,
            None,
            constraints,
            disturbance,
        )

        self._solve = jax.jit(work)

    def run(
        self,
        x0: jnp.ndarray,
        reference: jnp.ndarray,
        parameter: Any,
    ):
        (
            X,
            U,
            V,
            w,
            y,
            rho,
            backoffs,
            Phi_x,
            Phi_u,
            betaN,
            muN,
            converged_admm,
        ) = self._solve(
            reference,
            parameter,
            self.config.W,
            x0,
            self.X0,
            self.U0,
            self.V0,
            self.w,
            self.y,
            self.rho,
            self.obstacles,
            self.h_ct_ws,
            self.beta_ws,
            self.mu_ws,
            self.Phi_x_ws,
            self.Phi_u_ws,
            self.converged_admm,
        )

        self.converged_admm = converged_admm

        s = self.shift

        def shift_and_pad(arr, pad_value=None):
            if pad_value is None:
                tail = jnp.repeat(
                    arr[-1:],
                    repeats=s,
                    axis=0,
                )
            else:
                tail = jnp.broadcast_to(
                    pad_value,
                    (s,) + arr.shape[1:],
                )

            return jnp.concatenate(
                [arr[s:], tail],
                axis=0,
            )

        # Primal warm starts
        self.U0 = shift_and_pad(U)
        self.X0 = shift_and_pad(X)
        self.V0 = shift_and_pad(V)

        # Constraint and tube warm starts
        self.h_ct_ws = shift_and_pad(backoffs)
        self.beta_ws = shift_and_pad(betaN)
        self.mu_ws = shift_and_pad(muN)

        # ADMM warm starts
        self.w = shift_and_pad(w)
        self.y = shift_and_pad(y)

        rho = jnp.asarray(
            rho,
            dtype=self.rho.dtype,
        )

        # Preserve the scaled dual variable when rho changes.
        self.y = rho / self.rho * self.y
        self.rho = rho

        # SLS response warm starts
        self.Phi_x_ws = Phi_x
        self.Phi_u_ws = Phi_u

        return (
            U[0],
            X,
            U,
            V,
            backoffs,
            Phi_x,
            Phi_u,
        )