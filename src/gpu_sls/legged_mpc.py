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

    # Primal SQP warm starts
    X0: jnp.ndarray
    U0: jnp.ndarray
    V0: jnp.ndarray
    W: object

    # Main ADMM warm starts
    w: jnp.ndarray
    y: jnp.ndarray
    rho: jnp.ndarray

    # Gradient ADMM warm starts
    rho_grad: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray

    # Constraint/tube warm starts
    h_ct_ws: jnp.ndarray
    beta_ws: jnp.ndarray
    mu_ws: jnp.ndarray

    # SLS response warm starts
    Phi_x_ws: jnp.ndarray
    Phi_u_ws: jnp.ndarray
    Phi_x_I_ws: jnp.ndarray
    Phi_u_I_ws: jnp.ndarray

    # Solver state
    converged_admm: jnp.ndarray


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
    n_joints,
    horizon,
    shift,
    u_ref,
    initial_rho,
    x0,
    X_prev,
    U_prev,
    V_prev,
    w_prev,
    y_prev,
    rho_prev,
    rho_grad_prev,
    h_ct_prev,
    beta_prev,
    mu_prev,
    Phi_x_prev,
    Phi_u_prev,
    Phi_x_I_prev,
    Phi_u_I_prev,
    a_prev,
    b_prev,
    X,
    U,
    V,
    w,
    y,
    rho,
    rho_grad,
    backoffs,
    Phi_x,
    Phi_u,
    beta,
    mu,
    Phi_x_I,
    Phi_u_I,
    a,
    b,
    converged_admm,
):
    """Shift the new GPU-SLS solver state for the next MPC call."""

    q_slice = slice(7, 7 + n_joints)
    dq_slice = slice(13 + n_joints, 13 + 2 * n_joints)
    u_fallback_idx = 1 if horizon > 1 else 0

    def shift_trajectory(arr):
        tail = jnp.repeat(arr[-1:], shift, axis=0)
        return jnp.concatenate([arr[shift:], tail], axis=0)

    def safe_update():
        # Match GenericMPC.run(): shift/pad along the horizon axis and
        # rescale scaled-dual variables if rho changed.
        y0 = shift_trajectory(y)
        y0 = rho / rho_prev * y0

        a0 = shift_trajectory(a)
        b0 = shift_trajectory(b)
        b0 = rho_grad / rho_grad_prev * b0

        return (
            shift_trajectory(U),
            shift_trajectory(X),
            shift_trajectory(V),
            shift_trajectory(w),
            y0,
            rho,
            rho_grad,
            shift_trajectory(backoffs),
            shift_trajectory(beta),
            shift_trajectory(mu),
            Phi_x,
            Phi_u,
            Phi_x_I,
            Phi_u_I,
            a0,
            b0,
            converged_admm,
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
            jnp.asarray(initial_rho, dtype=rho_grad_prev.dtype),
            jnp.zeros_like(h_ct_prev),
            jnp.ones_like(beta_prev) * 1e-10,
            jnp.zeros_like(mu_prev),
            jnp.zeros_like(Phi_x_prev),
            jnp.zeros_like(Phi_u_prev),
            jnp.zeros_like(Phi_x_I_prev),
            jnp.zeros_like(Phi_u_I_prev),
            jnp.zeros_like(a_prev),
            jnp.zeros_like(b_prev),
            jnp.asarray(False),
            U_prev[u_fallback_idx, :n_joints],
            X_prev[1, q_slice],
            X_prev[1, dq_slice],
        )

    valid_solution = _solution_is_valid(
        X,
        U,
        V,
        w,
        y,
        rho,
        rho_grad,
        backoffs,
        Phi_x,
        Phi_u,
        beta,
        mu,
        Phi_x_I,
        Phi_u_I,
        a,
        b,
    )

    return jax.lax.cond(valid_solution, safe_update, unsafe_update)


class MPCWrapper:
    """MPX-compatible legged MPC wrapper using the new GPU-SLS SQP API.

    Public flow:
        data = wrapper.make_data()
        data, tau = wrapper.run(data, x0, command, contact)

    The carry contains all warm-start state required by the current
    gpu_sls.gpu_sqp.sqp interface.
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
        del limited_memory  # Retained for compatibility with MPX.

        self.config = config
        self.nu = getattr(config, "nu", getattr(config, "m", None))
        if self.nu is None:
            raise ValueError("legged config must define `nu` (or MPX-compatible `m`)")

        self.mpc_frequency = config.mpc_frequency
        self.shift = int(1 / (config.dt * config.mpc_frequency))
        self.default_contact = jnp.zeros(config.n_contact)

        self.qpos_slice = slice(0, 7 + config.n_joints)
        self.qvel_slice = slice(
            self.qpos_slice.stop,
            self.qpos_slice.stop + 6 + config.n_joints,
        )
        self.foot_slice = config.foot_slice

        # ------------------------------------------------------------
        # MuJoCo / MJX setup
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Problem definition
        # ------------------------------------------------------------
        self.cost = config.cost
        self.hessian_approx = getattr(config, "hessian_approx", None)
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

        if obstacles is None:
            obstacles = getattr(config, "obstacles", jnp.empty((0, 3)))
        self.obstacles = jnp.asarray(obstacles)
        num_obstacles = self.obstacles.shape[0]

        # Preserve the old explicit num_constraints API, but infer it in the
        # same way as GenericMPC when it is not supplied.
        if num_constraints is None:
            num_constraints = getattr(config, "num_constraints", None)

        if num_constraints is None:
            x_dummy = jnp.zeros((config.n,), dtype=jnp.asarray(config.initial_state).dtype)
            u_dummy = jnp.zeros((self.nu,), dtype=jnp.asarray(config.u_ref).dtype)
            t_dummy = jnp.asarray(0, dtype=jnp.int32)
            constraint_shape = jax.eval_shape(
                self.constraints,
                x_dummy,
                u_dummy,
                t_dummy,
            )
            num_constraints = constraint_shape.shape[0]

        self.num_constraints = int(num_constraints)

        if self.num_constraints < num_obstacles:
            raise ValueError("num_constraints must include all obstacle constraints")

        # ------------------------------------------------------------
        # Initial nominal state
        # ------------------------------------------------------------
        # The config owns the full state layout, including any auxiliary
        # states such as foot positions, GRFs, and a min-time T state.
        self.initial_state = jnp.asarray(config.initial_state)

        if self.initial_state.shape[0] != config.n:
            raise ValueError(
                "config.initial_state must have length config.n; "
                f"got {self.initial_state.shape[0]} and {config.n}"
            )

        self.initial_X0 = jnp.tile(self.initial_state, (config.N + 1, 1))
        self.initial_U0 = jnp.tile(config.u_ref, (config.N, 1))
        self.initial_V0 = jnp.zeros((config.N + 1, config.n), dtype=self.initial_X0.dtype)
        self.initial_liftoff = jnp.zeros(
            3 * config.n_contact,
            dtype=self.initial_X0.dtype,
        )

        # ------------------------------------------------------------
        # Main ADMM state
        # ------------------------------------------------------------
        self.initial_w = jnp.zeros(
            (config.N + 1, self.num_constraints),
            dtype=self.initial_X0.dtype,
        )
        self.initial_y = jnp.zeros_like(self.initial_w)
        self.initial_rho = jnp.asarray(
            self.admm_config.initial_rho,
            dtype=self.initial_w.dtype,
        )

        # ------------------------------------------------------------
        # Gradient ADMM state -- new API
        # ------------------------------------------------------------
        self.initial_rho_grad = jnp.asarray(
            self.admm_config.initial_rho,
            dtype=self.initial_w.dtype,
        )

        L = self.sls_config.gradient_window
        self.initial_a = jnp.zeros(
            (config.N + 1, L, self.num_constraints),
            dtype=self.initial_X0.dtype,
        )
        self.initial_b = jnp.zeros_like(self.initial_a)

        # ------------------------------------------------------------
        # Constraint / tube warm starts
        # ------------------------------------------------------------
        regular_constraints = self.num_constraints - num_obstacles

        self.initial_h_ct_ws = jnp.zeros(
            (config.N + 1, regular_constraints),
            dtype=self.initial_X0.dtype,
        )
        self.initial_beta_ws = (
            jnp.ones(
                (config.N + 1, config.N + 1, regular_constraints),
                dtype=self.initial_X0.dtype,
            )
            * 1e-10
        )
        self.initial_mu_ws = jnp.zeros(
            (config.N + 1, self.num_constraints),
            dtype=self.initial_X0.dtype,
        )

        # ------------------------------------------------------------
        # SLS response warm starts
        # ------------------------------------------------------------
        self.initial_Phi_x_ws = jnp.zeros(
            (config.N + 1, config.N + 1, config.n, config.n),
            dtype=self.initial_X0.dtype,
        )
        self.initial_Phi_u_ws = jnp.zeros(
            (config.N, config.N + 1, self.nu, config.n),
            dtype=self.initial_X0.dtype,
        )

        # Identity/gradient SLS responses -- new API
        self.initial_Phi_x_I_ws = jnp.zeros_like(self.initial_Phi_x_ws)
        self.initial_Phi_u_I_ws = jnp.zeros_like(self.initial_Phi_u_ws)

        # Solver convergence state -- new API
        self.initial_converged_admm = jnp.asarray(False)

        # ------------------------------------------------------------
        # Solver binding
        # ------------------------------------------------------------
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

        self._ref_gen = partial(config.reference_generator, mass=robot_mass)
        self._timer_run = jax.jit(mpc_utils.timer_run)
        self._update_warm_start = partial(
            _update_warm_start,
            config.n_joints,
            config.N,
            self.shift,
            config.u_ref,
            self.admm_config.initial_rho,
        )

    def reset(self, data, X_in=None, U_in=None):
        """
        Reset all MPC/SQP/SLS/ADMM warm-start state.

        If X_in / U_in are provided, use them as the primal SQP initial
        trajectory. Otherwise fall back to the wrapper's default initial
        trajectories.
        """

        # ------------------------------------------------------------
        # Primal SQP warm starts
        # ------------------------------------------------------------
        X0 = self.initial_X0 if X_in is None else jnp.asarray(X_in)
        U0 = self.initial_U0 if U_in is None else jnp.asarray(U_in)

        if X0.shape != self.initial_X0.shape:
            raise ValueError(
                f"X_in must have shape {self.initial_X0.shape}, "
                f"got {X0.shape}"
            )

        if U0.shape != self.initial_U0.shape:
            raise ValueError(
                f"U_in must have shape {self.initial_U0.shape}, "
                f"got {U0.shape}"
            )

        # ------------------------------------------------------------
        # Reset entire solver workspace
        # ------------------------------------------------------------
        return data.replace(
            # Primal SQP
            X0=X0,
            U0=U0,
            V0=jnp.zeros_like(self.initial_V0),

            # Main ADMM
            w=jnp.zeros_like(self.initial_w),
            y=jnp.zeros_like(self.initial_y),
            rho=jnp.asarray(
                self.admm_config.initial_rho,
                dtype=self.initial_rho.dtype,
            ),

            # Gradient ADMM
            rho_grad=jnp.asarray(
                self.admm_config.initial_rho,
                dtype=self.initial_rho_grad.dtype,
            ),
            a=jnp.zeros_like(self.initial_a),
            b=jnp.zeros_like(self.initial_b),

            # Constraint / tube state
            h_ct_ws=jnp.zeros_like(self.initial_h_ct_ws),
            beta_ws=jnp.ones_like(self.initial_beta_ws) * 1e-10,
            mu_ws=jnp.zeros_like(self.initial_mu_ws),

            # SLS response warm starts
            Phi_x_ws=jnp.zeros_like(self.initial_Phi_x_ws),
            Phi_u_ws=jnp.zeros_like(self.initial_Phi_u_ws),
            Phi_x_I_ws=jnp.zeros_like(self.initial_Phi_x_I_ws),
            Phi_u_I_ws=jnp.zeros_like(self.initial_Phi_u_I_ws),

            # Solver state
            converged_admm=jnp.asarray(False),

            # MPC timing state
            time=jnp.asarray(0.0, dtype=data.time.dtype),
            contact_time=self.config.timer_t,
            liftoff=self.initial_liftoff,
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
            rho_grad=self.initial_rho_grad,
            a=self.initial_a,
            b=self.initial_b,
            h_ct_ws=self.initial_h_ct_ws,
            beta_ws=self.initial_beta_ws,
            mu_ws=self.initial_mu_ws,
            Phi_x_ws=self.initial_Phi_x_ws,
            Phi_u_ws=self.initial_Phi_u_ws,
            Phi_x_I_ws=self.initial_Phi_x_I_ws,
            Phi_u_I_ws=self.initial_Phi_u_I_ws,
            converged_admm=self.initial_converged_admm,
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

        # ------------------------------------------------------------
        # New gpu_sls.gpu_sqp.sqp API
        # ------------------------------------------------------------
        (
            X,
            U,
            V,
            w,
            y,
            rho,
            rho_grad,
            backoffs,
            Phi_x,
            Phi_u,
            beta,
            mu,
            Phi_x_I,
            Phi_u_I,
            a,
            b,
            converged_admm,
        ) = self._solve(
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
            data.rho_grad,
            self.obstacles,
            data.h_ct_ws,
            data.beta_ws,
            data.mu_ws,
            data.Phi_x_ws,
            data.Phi_u_ws,
            data.Phi_x_I_ws,
            data.Phi_u_I_ws,
            data.a,
            data.b,
            data.converged_admm,
        )

        valid_solution = _solution_is_valid(
            X,
            U,
            V,
            w,
            y,
            rho,
            rho_grad,
            backoffs,
            Phi_x,
            Phi_u,
            beta,
            mu,
            Phi_x_I,
            Phi_u_I,
            a,
            b,
        )

        tau = jax.lax.cond(
            valid_solution,
            lambda _: self.control_output(x0, X, U, reference, parameter),
            lambda _: self.control_output(
                x0,
                data.X0,
                data.U0,
                reference,
                parameter,
            ),
            operand=None,
        )

        # Shift the solution so the next call starts from the previous optimum.
        (
            U0,
            X0,
            V0,
            w0,
            y0,
            rho0,
            rho_grad0,
            h_ct_ws,
            beta_ws,
            mu_ws,
            Phi_x_ws,
            Phi_u_ws,
            Phi_x_I_ws,
            Phi_u_I_ws,
            a0,
            b0,
            converged_admm0,
            _,
            q,
            dq,
        ) = self._update_warm_start(
            x0,
            data.X0,
            data.U0,
            data.V0,
            data.w,
            data.y,
            data.rho,
            data.rho_grad,
            data.h_ct_ws,
            data.beta_ws,
            data.mu_ws,
            data.Phi_x_ws,
            data.Phi_u_ws,
            data.Phi_x_I_ws,
            data.Phi_u_I_ws,
            data.a,
            data.b,
            X,
            U,
            V,
            w,
            y,
            rho,
            rho_grad,
            backoffs,
            Phi_x,
            Phi_u,
            beta,
            mu,
            Phi_x_I,
            Phi_u_I,
            a,
            b,
            converged_admm,
        )

        data = data.replace(
            time=current_time,
            X0=X0,
            U0=U0,
            V0=V0,
            w=w0,
            y=y0,
            rho=rho0,
            rho_grad=rho_grad0,
            a=a0,
            b=b0,
            h_ct_ws=h_ct_ws,
            beta_ws=beta_ws,
            mu_ws=mu_ws,
            Phi_x_ws=Phi_x_ws,
            Phi_u_ws=Phi_u_ws,
            Phi_x_I_ws=Phi_x_I_ws,
            Phi_u_I_ws=Phi_u_I_ws,
            converged_admm=converged_admm0,
            contact_time=contact_time,
            liftoff=liftoff,
        )

        return data, tau, q, dq

    def run(self, data, x0, input, contact=None):
        """Run one MPC step and return the updated carry and torque command."""

        contact = self.default_contact if contact is None else jnp.asarray(contact)
        data, tau, _, _ = self._run_impl(data, x0, input, contact)
        return data, tau

    def foot_positions(self, qpos):
        """Return flattened contact-point positions for the provided configuration."""

        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return jnp.array(
            [self.data.geom_xpos[idx] for idx in self.contact_id_mj]
        ).flatten()
