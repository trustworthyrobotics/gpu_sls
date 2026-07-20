# Adapted from https://github.com/iit-DLSLab/mpx/blob/main/mpx/utils/mpc_wrapper.py

from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src.dataclasses import PyTreeNode

import gpu_sls.gpu_sqp
import mpx.utils.mpc_utils as mpc_utils


class MPCData(PyTreeNode):
    """Carry state for the pure functional MPC API."""

    dt: float
    time: jnp.ndarray
    duty_factor: float
    step_freq: float
    step_height: float
    contact_time: jnp.ndarray
    liftoff: jnp.ndarray
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    W: object
    w: jnp.ndarray
    y: jnp.ndarray
    rho: jnp.ndarray
    h_ct_ws: jnp.ndarray
    beta_ws: jnp.ndarray
    mu_ws: jnp.ndarray
    Phi_x_ws: jnp.ndarray
    Phi_u_ws: jnp.ndarray


mpx_data = MPCData


def _solution_is_valid(*values):
    valid = jnp.asarray(True)
    for value in values:
        valid = jnp.logical_and(valid, jnp.all(jnp.isfinite(value)))
    return valid


def build_solver_step(
    sls_config,
    sqp_config,
    admm_config,
    cost,
    dynamics,
    hessian_approx,
    constraints,
    disturbance,
):
    """Bind a legged problem to the GPU-SLS SQP solver."""

    return partial(
        gpu_sls.gpu_sqp.sqp,
        sls_config,
        sqp_config,
        admm_config,
        cost,
        dynamics,
        hessian_approx,
        constraints,
        disturbance,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _update_warm_start(
    n_joints, horizon, shift, u_ref, initial_rho, x0,
    X_prev, U_prev, V_prev, w_prev, y_prev, rho_prev,
    h_ct_prev, beta_prev, mu_prev, Phi_x_prev, Phi_u_prev,
    X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, beta, mu,
):
    """Shift the solution for the next MPC step and extract the first command."""

    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(13 + n_joints, 13 + 2 * n_joints)
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(trajectory):
        tail = jnp.repeat(trajectory[-1:], shift, axis=0)
        return jnp.concatenate([trajectory[shift:], tail], axis=0)

    def safe_update():
        shifted_y = shift_trajectory(y)
        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(w),
            rho / rho_prev * shifted_y,
            rho,
            shift_trajectory(backoffs),
            shift_trajectory(beta),
            shift_trajectory(mu),
            Phi_x,
            Phi_u,
            U[0, :n_joints],
            X[0, q_slice],
            X[1, dq_slice],
        )

    def unsafe_update():
        return (
            jnp.tile(u_ref, (horizon, 1)),
            jnp.tile(x0, (horizon + 1, 1)),
            jnp.zeros_like(V_prev),
            jnp.zeros_like(w_prev),
            jnp.zeros_like(y_prev),
            jnp.asarray(initial_rho, dtype=rho_prev.dtype),
            jnp.zeros_like(h_ct_prev),
            jnp.ones_like(beta_prev) * 1e-10,
            jnp.zeros_like(mu_prev),
            jnp.zeros_like(Phi_x_prev),
            jnp.zeros_like(Phi_u_prev),
            U_prev[u_fallback_idx, :n_joints],
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = _solution_is_valid(
        X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, beta, mu,
    )
    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


class MPCWrapper:
    """Minimal MPC API built for `jit` and `vmap`.

    The public flow is:
    `data = wrapper.make_data()`
    `data, tau = wrapper.run(data, x0, command, contact)`

    The public API matches MPX, while the carry also contains every GPU-SLS
    warm-start value needed to keep repeated calls functional and JIT-safe.
    """

    def __init__(
        self,
        config,
        limited_memory=False,
        *,
        sls_config=None,
        sqp_config=None,
        admm_config=None,
        constraints=None,
        obstacles=None,
        num_constraints=None,
        disturbance=None,
    ):
        """Create a GPU-SLS-backed wrapper around an MPX legged config.

        Solver/problem arguments can be supplied here or exposed as attributes
        on ``config``. This keeps existing config-module-based call sites usable
        while making the new backend dependencies explicit.
        """
        del limited_memory  # Retained for compatibility with the MPX constructor.
        self.config = config
        self.nu = getattr(config, "nu", getattr(config, "m", None))
        if self.nu is None:
            raise ValueError("legged config must define `nu` (or MPX-compatible `m`)")
        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        self.default_contact = jnp.zeros(config.n_contact)
        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(self.qpos_slice.stop, self.qpos_slice.stop + 6 + config.n_joints)
        self.foot_slice = config.foot_slice

        self.model = mujoco.MjModel.from_xml_path(config.model_path)
        data = mujoco.MjData(self.model)
        mujoco.mj_fwdPosition(self.model, data)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        robot_mass = data.qM[0]

        self.contact_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]
        self.body_id = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in config.body_name
        ]
        self.contact_id_mj = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in config.contact_frame
        ]

        self.cost = config.cost
        self.hessian_approx = config.hessian_approx
        self.dynamics = config.dynamics(
            self.model,
            self.mjx_model,
            self.contact_id,
            self.body_id,
        )

        def problem_value(name, value):
            if value is not None:
                return value
            if hasattr(config, name):
                return getattr(config, name)
            raise ValueError(
                f"MPCWrapper requires `{name}` as a keyword argument or config attribute"
            )

        self.sls_config = problem_value("sls_config", sls_config)
        self.sqp_config = problem_value("sqp_config", sqp_config)
        self.admm_config = problem_value("admm_config", admm_config)
        self.constraints = problem_value("constraints", constraints)
        self.disturbance = problem_value("disturbance", disturbance)
        self.num_constraints = int(problem_value("num_constraints", num_constraints))
        if obstacles is None:
            obstacles = getattr(config, "obstacles", jnp.empty((0, 3)))
        self.obstacles = jnp.asarray(obstacles)
        num_obstacles = self.obstacles.shape[0]
        if self.num_constraints < num_obstacles:
            raise ValueError("num_constraints must include all obstacle constraints")

        # The config owns the nominal state layout, including any extra states.
        self.initial_state = jnp.asarray(config.initial_state)

        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n))
        self.initial_liftoff = jnp.zeros(3 * config.n_contact)
        self.initial_w = jnp.zeros((config.N + 1, self.num_constraints))
        self.initial_y = jnp.zeros_like(self.initial_w)
        self.initial_rho = jnp.asarray(
            self.admm_config.initial_rho,
            dtype=self.initial_w.dtype,
        )
        regular_constraints = self.num_constraints - num_obstacles
        self.initial_h_ct_ws = jnp.zeros((config.N + 1, regular_constraints))
        self.initial_beta_ws = jnp.ones(
            (config.N + 1, config.N + 1, regular_constraints)
        ) * 1e-10
        self.initial_mu_ws = jnp.zeros((config.N + 1, self.num_constraints))
        self.initial_Phi_x_ws = jnp.zeros(
            (config.N + 1, config.N + 1, config.n, config.n)
        )
        self.initial_Phi_u_ws = jnp.zeros(
            (config.N, config.N + 1, self.nu, config.n)
        )

        solve = build_solver_step(
            self.sls_config,
            self.sqp_config,
            self.admm_config,
            self.cost,
            self.dynamics,
            self.hessian_approx,
            self.constraints,
            self.disturbance,
        )
        self.solver_mode = "gpu_sls"
        self._solve = jax.jit(solve)

        self._ref_gen = partial(config.reference_generator,mass=robot_mass)
        self._timer_run = jax.jit(mpc_utils.timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.n_joints,
            config.N,
            self.shift,
            config.u_ref,
            self.admm_config.initial_rho,
        )

    def make_data(self):
        """Allocate the pytree state used by the pure functional API."""

        return MPCData(
            dt=self.config.dt,
            time=jnp.asarray(0.0, dtype=jnp.float32),
            duty_factor=self.config.duty_factor,
            step_freq=self.config.step_freq,
            step_height=self.config.step_height,
            contact_time=self.config.timer_t,
            liftoff=self.initial_liftoff,
            X0=self.initial_X0,
            U0=self.initial_U0,
            V0=self.initial_V0,
            W=self.config.W,
            w=self.initial_w,
            y=self.initial_y,
            rho=self.initial_rho,
            h_ct_ws=self.initial_h_ct_ws,
            beta_ws=self.initial_beta_ws,
            mu_ws=self.initial_mu_ws,
            Phi_x_ws=self.initial_Phi_x_ws,
            Phi_u_ws=self.initial_Phi_u_ws,
        )

    def control_output(self, x0, X, U, reference, parameter):
        del x0, X, reference, parameter
        return U[0, : self.config.n_joints]

    def _run_impl(self, data, x0, input, contact):

        current_time = data.time + jnp.asarray(
            1 / self.mpc_frequency,
            dtype=data.time.dtype,
        )

        _, contact_time = self._timer_run(
            data.duty_factor,
            data.step_freq,
            data.contact_time,
            1 / self.mpc_frequency,
        )

        reference, parameter, liftoff = self._ref_gen(
            duty_factor=data.duty_factor,
            step_freq=data.step_freq,
            step_height=data.step_height,
            t_timer=data.contact_time,
            x=x0,
            foot=x0[self.foot_slice],
            input=input,
            liftoff=data.liftoff,
            contact=contact,
            current_time=current_time,
        )

        # Reference generation and solver execution stay on the pure JAX path.
        X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, beta, mu = self._solve(
            reference,
            parameter,
            data.W,
            x0,
            data.X0,
            data.U0,
            data.V0,
            data.w,
            data.y,
            data.rho,
            self.obstacles,
            data.h_ct_ws,
            data.beta_ws,
            data.mu_ws,
            data.Phi_x_ws,
            data.Phi_u_ws,
        )
        valid_solution = _solution_is_valid(
            X, U, V, w, y, rho, backoffs, Phi_x, Phi_u, beta, mu,
        )
        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x0, X, U, reference, parameter),
            lambda _: self.control_output(x0, data.X0, data.U0, reference, parameter),
            operand=None,
        )
        # Shift the solution so the next call starts from the previous optimum.
        (
            U0, X0, V0, w0, y0, rho0, h_ct_ws, beta_ws, mu_ws,
            Phi_x_ws, Phi_u_ws, _, q, dq,
        ) = self._update_warm_start(
            x0,
            data.X0,
            data.U0,
            data.V0,
            data.w,
            data.y,
            data.rho,
            data.h_ct_ws,
            data.beta_ws,
            data.mu_ws,
            data.Phi_x_ws,
            data.Phi_u_ws,
            X,
            U,
            V,
            w,
            y,
            rho,
            backoffs,
            Phi_x,
            Phi_u,
            beta,
            mu,
        )

        data = data.replace(
            time=current_time,
            X0=X0,
            U0=U0,
            V0=V0,
            w=w0,
            y=y0,
            rho=rho0,
            h_ct_ws=h_ct_ws,
            beta_ws=beta_ws,
            mu_ws=mu_ws,
            Phi_x_ws=Phi_x_ws,
            Phi_u_ws=Phi_u_ws,
            contact_time=contact_time,
            liftoff=liftoff,
        )
        return data, tau, q, dq

    def run(self, data, x0, input, contact=None):
        """Run one MPC step and return the updated carry and torque command."""

        contact = self.default_contact if contact is None else jnp.asarray(contact)
        data, tau, _, _ = self._run_impl(data, x0, input, contact)
        return data, tau

    def reset(self, data, qpos, qvel, foot):
        """Reset the warm start around the provided measured state."""

        # Start from the config initial_state so any extra state entries keep
        # their configured default value.
        initial_state = (
            self.initial_state
            .at[self.qpos_slice].set(jnp.ravel(qpos))
            .at[self.qvel_slice].set(jnp.ravel(qvel))
            .at[self.foot_slice].set(jnp.ravel(foot))
        )
        return data.replace(
            U0=self.initial_U0,
            X0=jnp.tile(initial_state, (self.config.N + 1, 1)),
            V0=self.initial_V0,
            w=self.initial_w,
            y=self.initial_y,
            rho=self.initial_rho,
            h_ct_ws=self.initial_h_ct_ws,
            beta_ws=self.initial_beta_ws,
            mu_ws=self.initial_mu_ws,
            Phi_x_ws=self.initial_Phi_x_ws,
            Phi_u_ws=self.initial_Phi_u_ws,
            time=jnp.asarray(0.0, dtype=jnp.float32),
            contact_time=self.config.timer_t,
            liftoff=jnp.ravel(foot),
        )

    def foot_positions(self, qpos):
        """Return the flattened contact-point positions for the provided configuration."""

        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return jnp.array([self.data.geom_xpos[idx] for idx in self.contact_id_mj]).flatten()
