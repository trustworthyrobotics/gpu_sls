# Adapted from MPX's MJX quadruped example and the user's minimum-time quadrotor experiment.

import argparse
import os
import sys
import time
import subprocess
from dataclasses import fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from timeit import default_timer as timer

# Keep this before importing JAX.
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

# MuJoCo selects its OpenGL backend when it is imported. X11 forwarding often
# provides a DISPLAY but cannot create the GLX context required by GLFW. Use
# EGL for offscreen video rendering in headless and forwarded-SSH sessions.
_display = os.environ.get("DISPLAY", "")
_forwarded_x11 = bool(os.environ.get("SSH_CONNECTION")) and _display.startswith(
    ("localhost:", "127.0.0.1:")
)
_auto_headless = (
    "--headless" in sys.argv
    or (not _display and not os.environ.get("WAYLAND_DISPLAY"))
    or _forwarded_x11
)
if _auto_headless:
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math

import mpx
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives
import mpx.utils.sim as sim_utils
import config_go2 as config

import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# -----------------------------------------------------------------------------
# Forward-speed RTI MPC experiment configuration
# -----------------------------------------------------------------------------
# One optimized duration state T is appended to the whole-body state.
NUM_PHASES = 1

INITIAL_DURATION = 0.80
MIN_TIME = 0.15
MAX_TIME = 2.00

# Objective weights:
#   + TIME_WEIGHT * T                 -> prefer shorter horizons
#   - FORWARD_VELOCITY_WEIGHT * vx   -> prefer high forward speed
#   - FORWARD_PROGRESS_WEIGHT * x_N  -> prefer forward displacement
TIME_WEIGHT = 5.0
FORWARD_VELOCITY_WEIGHT = 10.0
FORWARD_PROGRESS_WEIGHT = 25.0

# Used only to construct the nominal gait / footstep reference.
# It is not a hard speed cap because x and vx tracking weights are zero.
FORWARD_REFERENCE_SPEED = 0.6

COMMAND = jnp.array([
    FORWARD_REFERENCE_SPEED, 0.0, 0.0,
    0.0, 0.0, 0.0,
    config.robot_height,
])


# Preserve the original GO2 whole-body state dimension before augmenting it.
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
GRF_START = 13 + 2 * config.n_joints + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact


# -----------------------------------------------------------------------------
# Fixed-horizon whole-body dynamics
# -----------------------------------------------------------------------------
def min_time_quadruped_dynamics(model, mjx_model, contact_id, body_id):
    """Return f(x,u,t,parameter) with x[-1] = optimized horizon duration T."""

    n_joints = config.n_joints
    dtau = 1.0 / config.N

    def dynamics(x, u, t, parameter):
        # Optimized physical shooting step.
        T = x[-1]
        dt = dtau * T

        # The final element T is not part of MuJoCo qpos/qvel.
        qpos = x[: n_joints + 7]
        qvel = x[n_joints + 7 : 2 * n_joints + 13]

        mjx_data = mjx.make_data(model)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)

        mjx_data = mjx.fwd_position(mjx_model, mjx_data)
        mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

        M = mjx_data.qLD
        D = mjx_data.qfrc_bias

        # parameter remains the MPX contact schedule.  We do NOT reuse it for
        # dtau, because the legged model already needs parameter[t, :4].
        contact = parameter[t, :4]

        tau = jnp.concatenate([
            jnp.zeros(6, dtype=x.dtype),
            u,
        ])

        FL_leg = mjx_data.geom_xpos[contact_id[0]]
        FR_leg = mjx_data.geom_xpos[contact_id[1]]
        RL_leg = mjx_data.geom_xpos[contact_id[2]]
        RR_leg = mjx_data.geom_xpos[contact_id[3]]

        J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
        J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
        J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
        J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

        J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)

        if len(contact_id) > 4:
            current_leg = jnp.asarray([
                mjx_data.geom_xpos[int(geom_id)] for geom_id in contact_id
            ]).flatten()
        else:
            current_leg = jnp.concatenate(
                [FL_leg, FR_leg, RL_leg, RR_leg], axis=0
            )

        alpha = 25.0
        g_dot = J.T @ qvel
        baumgarte_term = -2.0 * alpha * g_dot

        M_inv_J = jax.scipy.linalg.cho_solve((M, False), J)
        JT_M_invJ = J.T @ M_inv_J

        rhs = (
            -J.T @ jax.scipy.linalg.cho_solve((M, False), tau - D)
            + baumgarte_term
        )

        cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
        grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)

        grf = jnp.concatenate([
            grf[0:3] * contact[0],
            grf[3:6] * contact[1],
            grf[6:9] * contact[2],
            grf[9:12] * contact[3],
        ])

        acceleration = jax.scipy.linalg.cho_solve(
            (M, False),
            tau - D + J @ grf,
        )

        # Semi-implicit Euler, now with dt = T / N.
        v_next = qvel + acceleration * dt
        p_next = qpos[:3] + v_next[:3] * dt
        quat_next = math.quat_integrate(qpos[3:7], v_next[3:6], dt)
        q_next = qpos[7 : 7 + n_joints] + v_next[6 : 6 + n_joints] * dt

        physical_next = jnp.concatenate([
            p_next,
            quat_next,
            q_next,
            v_next,
            current_leg,
            grf,
        ])

        # T_{k+1} = T_k.
        return jnp.concatenate([
            physical_next,
            jnp.reshape(T, (1,)),
        ])

    return dynamics


# -----------------------------------------------------------------------------
# Forward-speed objective
# -----------------------------------------------------------------------------
def min_time_quadruped_cost(W, reference, x, u, t):
    """
    Maximize forward speed/progress while regularizing the whole-body motion.

    There is deliberately NO terminal position target.

    Minimized objective:
        regularization
        - w_vx * v_x
        + 1_{t=N} * (w_time * T - w_progress * p_x)

    There is no terminal set. Therefore this is a scalarized
    time-versus-forward-progress problem rather than a classical
    minimum-time-to-a-target problem.
    """

    nj = config.n_joints
    nc = config.n_contact

    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + nj]
    dp = x[7 + nj : 10 + nj]
    omega = x[10 + nj : 13 + nj]
    dq = x[13 + nj : 13 + 2 * nj]
    p_leg = x[13 + 2 * nj : 13 + 2 * nj + 3 * nc]

    # Stop before the compatibility duration state.
    grf = x[GRF_START:GRF_STOP]
    tau = u[:nj]

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    contact = reference["contact"][t]
    grf_ref = reference["grf"][t]

    mu_friction = 0.5
    friction_cone = (
        mu_friction * grf[2::3]
        - jnp.sqrt(
            jnp.square(grf[1::3])
            + jnp.square(grf[0::3])
            + jnp.ones(nc, dtype=x.dtype) * 1e-2
        )
    )
    friction_penalty = jnp.sum(
        mpc_objectives.penalty(friction_cone) * contact
    )

    quat_error = math.quat_sub(quat, quat_ref)
    contact_map = jnp.ones(3 * nc, dtype=x.dtype)

    regularization = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_error.T @ W["rot"] @ quat_error
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
        + (contact_map * (p_leg - p_leg_ref)).T
        @ W["contact"]
        @ (contact_map * (p_leg - p_leg_ref))
        + tau.T @ W["tau"] @ tau
        + (grf - grf_ref).T @ W["grf"] @ (grf - grf_ref)
        + friction_penalty
    )

    # Linear reward for moving forward quickly.
    forward_velocity_reward = -W["forward_velocity"] * dp[0]

    # Apply the free-final-time cost ONCE at the terminal node. This gives
    # exactly +w_time*T rather than +(N+1)*w_time*T after stage summation.
    terminal_time_progress = jnp.where(
        t == config.N,
        W["time"] * x[-1] - W["forward_progress"] * p[0],
        jnp.asarray(0.0, dtype=x.dtype),
    )

    return (
        0.5 * regularization
        + forward_velocity_reward
        + terminal_time_progress
    )


# -----------------------------------------------------------------------------
# Reference generation
# -----------------------------------------------------------------------------
# MPX treats this timestep as static. Keep a nominal/reference dt here.
# The actual whole-body dynamics use the optimized dt = T / N.
REFERENCE_DT = float(INITIAL_DURATION / config.N)

_BASE_REFERENCE_GENERATOR = partial(
    mpc_utils.reference_generator,
    config.use_terrain_estimation,
    config.N,
    REFERENCE_DT,
    config.n_joints,
    config.n_contact,
    foot0=config.p_legs0,
    q0=config.q0,
    clearence_speed=config.clearance_speed,
)


def min_time_reference_generator(*, mass, x, **kwargs):
    ref, parameter, liftoff = _BASE_REFERENCE_GENERATOR(
        mass=mass,
        x=x,
        **kwargs,
    )

    # A receding forward-moving nominal reference. The x and vx tracking
    # weights are zero, so this does not cap the optimizer at this speed; it
    # mainly gives the gait/footstep generator a sensible forward motion.
    tau = (
        jnp.arange(config.N + 1, dtype=x.dtype)
        * jnp.asarray(REFERENCE_DT, dtype=x.dtype)
    )

    p0 = x[:3]
    p_ref = jnp.tile(p0, (config.N + 1, 1))
    p_ref = p_ref.at[:, 0].set(
        p0[0] + FORWARD_REFERENCE_SPEED * tau
    )
    p_ref = p_ref.at[:, 1].set(0.0)
    p_ref = p_ref.at[:, 2].set(config.robot_height)

    dp_ref = jnp.zeros((config.N + 1, 3), dtype=x.dtype)
    dp_ref = dp_ref.at[:, 0].set(FORWARD_REFERENCE_SPEED)

    ref = {
        **ref,
        "p": p_ref,
        "dp": dp_ref,
    }

    return ref, parameter, liftoff


# -----------------------------------------------------------------------------
# Inequality constraints g(x,u,t) <= 0
# -----------------------------------------------------------------------------
def min_time_constraints(x, u, t):
    del t

    # Free horizon-duration bounds.
    time_constraints = jnp.array([
        MIN_TIME - x[-1],
        x[-1] - MAX_TIME,
    ])

    # Joint torque bounds.
    torque_constraints = jnp.concatenate([
        u - config.max_torque,
        config.min_torque - u,
    ])

    # No terminal-set constraint.
    # No obstacle constraint.
    return jnp.concatenate([
        time_constraints,
        torque_constraints,
    ])


# -----------------------------------------------------------------------------
# Robust disturbance model scaled by optimized physical dt = T / N
# -----------------------------------------------------------------------------
def make_min_time_disturbance(n: int, N: int, E_mag: float):
    def disturbance_at_state(x_k):
        dt = x_k[-1] / N

        # Preserve the original experiment's disturbance structure: only base
        # x/y are disturbed.  The duration state is never disturbed.
        diag = jnp.zeros(n, dtype=x_k.dtype)
        diag = diag.at[:2].set(E_mag * dt)
        diag = diag.at[-1].set(0.0)
        return jnp.diag(diag)

    def disturbance(X):
        return jax.vmap(disturbance_at_state)(X)

    disturbance.at_state = disturbance_at_state
    return disturbance


# -----------------------------------------------------------------------------
# Initial SQP trajectory, analogous to X_ref in the quadrotor script
# -----------------------------------------------------------------------------
def build_initial_guess(x0, reference):
    X0 = jnp.tile(x0, (config.N + 1, 1))

    nj = config.n_joints
    nc = config.n_contact

    X0 = X0.at[:, :3].set(reference["p"])
    X0 = X0.at[:, 3:7].set(reference["quat"])
    X0 = X0.at[:, 7 : 7 + nj].set(reference["q"])
    X0 = X0.at[:, 7 + nj : 10 + nj].set(reference["dp"])
    X0 = X0.at[:, 10 + nj : 13 + nj].set(reference["omega"])
    X0 = X0.at[:, 13 + nj : 13 + 2 * nj].set(0.0)
    X0 = X0.at[:, 13 + 2 * nj : 13 + 2 * nj + 3 * nc].set(
        reference["foot"]
    )
    X0 = X0.at[:, GRF_START:GRF_STOP].set(reference["grf"])
    X0 = X0.at[:, -1].set(INITIAL_DURATION)

    return X0


def force_integer_num_phases(cfg):
    """Avoid JAX slice errors if SQPConfig.num_phases was declared as 1.0."""
    if is_dataclass(cfg):
        names = {f.name for f in fields(cfg)}
        if "num_phases" in names:
            return replace(cfg, num_phases=1)

    if hasattr(cfg, "num_phases"):
        try:
            cfg.num_phases = 1
        except Exception:
            pass

    return cfg


# -----------------------------------------------------------------------------
# Configure config_go2 for the augmented free-time state
# -----------------------------------------------------------------------------
def configure_min_time_problem():
    # Append one free horizon-duration state T.
    base_initial_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.n = PHYSICAL_N + NUM_PHASES
    config.nu = config.n_joints
    config.initial_state = jnp.concatenate([
        base_initial_state,
        jnp.array([INITIAL_DURATION], dtype=base_initial_state.dtype),
    ])

    config.dynamics = min_time_quadruped_dynamics
    config.cost = min_time_quadruped_cost
    config.reference_generator = min_time_reference_generator

    dtype = config.initial_state.dtype

    # x-position and forward-velocity tracking are intentionally zero:
    # progress is driven by explicit negative rewards in the objective.
    # y/z, attitude, joints, contacts and forces remain regularized.
    config.W = {
        "pos": jnp.diag(jnp.array([0.0, 10.0, 30.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([30.0, 30.0, 5.0], dtype=dtype)),
        "q": jnp.eye(config.n_joints, dtype=dtype) * 1e-1,
        "vel": jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=dtype)),
        "omega": jnp.eye(3, dtype=dtype) * 0.2,
        "dq": jnp.eye(config.n_joints, dtype=dtype) * 1e-2,
        "contact": jnp.eye(3 * config.n_contact, dtype=dtype) * 10.0,
        "tau": jnp.eye(config.n_joints, dtype=dtype) * 1e-3,
        "grf": jnp.eye(3 * config.n_contact, dtype=dtype) * 1e-4,
        "forward_velocity": jnp.asarray(
            FORWARD_VELOCITY_WEIGHT, dtype=dtype
        ),
        "forward_progress": jnp.asarray(
            FORWARD_PROGRESS_WEIGHT, dtype=dtype
        ),
        "time": jnp.asarray(
            TIME_WEIGHT, dtype=dtype
        ),
    }


# -----------------------------------------------------------------------------
# SQP-RTI helpers
# -----------------------------------------------------------------------------
def _solution_is_valid(*values):
    valid = jnp.asarray(True)
    for value in values:
        valid = jnp.logical_and(valid, jnp.all(jnp.isfinite(value)))
    return valid


def build_rti_solve_fn(mpc):
    """Build one receding-horizon SQP-RTI step.

    The solve itself performs one SQP iteration after initialization.  The
    previous solution and all ADMM/SLS quantities are shifted by one shooting
    node and carried in MPCData.

    Important timing convention:
        data.time/contact_time describe the CURRENT measured state.
        The clock is advanced only after the returned first control has actually
        been applied in MuJoCo for dt_apply = T_opt / N.
    """

    @jax.jit
    def solve_rti(mpc_data, x0, command, contact):
        reference, parameter, liftoff = mpc._ref_gen(
            duty_factor=mpc_data.duty_factor,
            step_freq=mpc_data.step_freq,
            step_height=mpc_data.step_height,
            t_timer=mpc_data.contact_time,
            x=x0,
            foot=x0[mpc.foot_slice],
            input=command,
            liftoff=mpc_data.liftoff,
            contact=contact,
            current_time=mpc_data.time,
        )

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
        ) = mpc._solve(
            reference,
            parameter,
            mpc_data.W,
            x0,
            mpc_data.X0,
            mpc_data.U0,
            mpc_data.V0,
            mpc_data.w,
            mpc_data.y,
            mpc_data.rho,
            mpc_data.rho_grad,
            mpc.obstacles,
            mpc_data.h_ct_ws,
            mpc_data.beta_ws,
            mpc_data.mu_ws,
            mpc_data.Phi_x_ws,
            mpc_data.Phi_u_ws,
            mpc_data.Phi_x_I_ws,
            mpc_data.Phi_u_I_ws,
            mpc_data.a,
            mpc_data.b,
            mpc_data.converged_admm,
        )

        valid = _solution_is_valid(
            X, U, V, w, y, rho, rho_grad, backoffs,
            Phi_x, Phi_u, beta, mu, Phi_x_I, Phi_u_I, a, b,
        )

        tau = jax.lax.cond(
            valid,
            lambda _: U[0, : config.n_joints],
            lambda _: mpc_data.U0[0, : config.n_joints],
            operand=None,
        )

        # Use the optimizer's current free horizon duration.
        duration = jax.lax.cond(
            valid,
            lambda _: X[0, -1],
            lambda _: mpc_data.X0[0, -1],
            operand=None,
        )
        duration = jnp.clip(duration, MIN_TIME, MAX_TIME)
        dt_apply = duration / config.N

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
            _,
            _,
        ) = mpc._update_warm_start(
            x0,
            mpc_data.X0,
            mpc_data.U0,
            mpc_data.V0,
            mpc_data.w,
            mpc_data.y,
            mpc_data.rho,
            mpc_data.rho_grad,
            mpc_data.h_ct_ws,
            mpc_data.beta_ws,
            mpc_data.mu_ws,
            mpc_data.Phi_x_ws,
            mpc_data.Phi_u_ws,
            mpc_data.Phi_x_I_ws,
            mpc_data.Phi_u_I_ws,
            mpc_data.a,
            mpc_data.b,
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

        # The shifted trajectory represents the remaining horizon after
        # executing one of N optimized shooting intervals.
        remaining_duration = jnp.clip(
            duration - dt_apply,
            MIN_TIME,
            MAX_TIME,
        )
        X0 = X0.at[:, -1].set(remaining_duration)

        mpc_data = mpc_data.replace(
            # Do NOT store dt_apply in data.dt here. In the current
            # MPCData definition, dt is annotated as `float`, so PyTreeNode
            # treats it as static metadata. dt_apply is a JAX tracer inside
            # this jitted function. It is returned explicitly below instead.
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
            liftoff=liftoff,
        )

        return mpc_data, tau, X, U, dt_apply, valid

    return solve_rti


def build_clock_update_fn(mpc):
    """Advance the gait/MPC clock by the physical control interval just applied."""

    @jax.jit
    def advance_clock(mpc_data, elapsed_dt):
        _, contact_time = mpc._timer_run(
            mpc_data.duty_factor,
            mpc_data.step_freq,
            mpc_data.contact_time,
            elapsed_dt,
        )
        return mpc_data.replace(
            time=mpc_data.time + elapsed_dt,
            contact_time=contact_time,
        )

    return advance_clock


def measured_augmented_state(mpc, data, foot, duration_guess):
    """Construct the current augmented state from MuJoCo measurements."""
    return (
        mpc.initial_state
        .at[mpc.qpos_slice].set(jnp.asarray(data.qpos.copy()))
        .at[mpc.qvel_slice].set(jnp.asarray(data.qvel.copy()))
        .at[mpc.foot_slice].set(jnp.ravel(foot))
        .at[-1].set(duration_guess)
    )



def encode_state_history_video(
    model,
    qpos_history,
    qvel_history,
    filename="quadruped_fast_forward_rti.mp4",
    video_fps=60,
    width=1280,
    height=720,
):
    """Render recorded closed-loop states after the RTI rollout has finished."""
    if len(qpos_history) == 0:
        return

    video_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.25, 0.0, config.robot_height * 0.5])
    camera.distance = 2.0
    camera.azimuth = 135
    camera.elevation = -20

    process = None
    try:
        frame0_qpos = np.asarray(qpos_history[0])
        frame0_qvel = np.asarray(qvel_history[0])
        video_data.qpos[:] = frame0_qpos
        video_data.qvel[:] = frame0_qvel
        mujoco.mj_normalizeQuat(model, video_data.qpos)
        mujoco.mj_forward(model, video_data)
        renderer.update_scene(video_data, camera=camera)
        first_frame = np.asarray(renderer.render(), dtype=np.uint8)

        frame_height, frame_width = first_frame.shape[:2]
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{frame_width}x{frame_height}",
            "-framerate", str(video_fps),
            "-i", "-", "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(filename),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process.stdin.write(first_frame.tobytes())

        for qpos, qvel in zip(qpos_history[1:], qvel_history[1:]):
            video_data.qpos[:] = np.asarray(qpos)
            video_data.qvel[:] = np.asarray(qvel)
            mujoco.mj_normalizeQuat(model, video_data.qpos)
            mujoco.mj_forward(model, video_data)
            renderer.update_scene(video_data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        error_output = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}: {error_output}"
            )
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        renderer.close()

    print(f"Saved closed-loop RTI rollout to: {filename}")
    print(f"Video frames: {len(qpos_history)} at {video_fps} fps")


# -----------------------------------------------------------------------------
# Closed-loop SQP-RTI forward-speed MPC with optimized horizon time
# -----------------------------------------------------------------------------
def main(headless=False, steps=250, save_video=True):
    if _auto_headless and not headless:
        print(
            "Interactive MuJoCo rendering is unavailable in this display "
            "session; using headless simulation instead."
        )
        headless = True

    configure_min_time_problem()

    mpx_root = Path(mpx.__file__).parent
    model = mujoco.MjModel.from_xml_path(
        str(mpx_root / "data" / "go2" / "scene_mjx.xml")
    )

    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720

    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1.0 / sim_frequency
    sim_dt = model.opt.timestep

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    # ------------------------------------------------------------------
    # RTI configuration:
    #   * initialization may use max_initial_sqp_iterations
    #   * every subsequent control update performs ONE SQP iteration
    # ------------------------------------------------------------------
    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-3,
        rho_max=1e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=1,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=1e-10,
        line_search=False,
    )
    sqp_cfg = force_integer_num_phases(sqp_cfg)

    E_mag = 0.05
    disturbance = make_min_time_disturbance(
        n=config.n,
        N=config.N,
        E_mag=E_mag,
    )

    obstacles = jnp.zeros((0, 3), dtype=config.initial_state.dtype)

    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=sls_cfg,
        sqp_config=sqp_cfg,
        admm_config=admm_cfg,
        constraints=min_time_constraints,
        obstacles=obstacles,
        disturbance=disturbance,
    )

    solve_rti = build_rti_solve_fn(mpc)
    advance_clock = build_clock_update_fn(mpc)

    # ------------------------------------------------------------------
    # Initial measured state and RTI warm start
    # ------------------------------------------------------------------
    data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    mpc_data = mpc.reset(
        mpc.make_data(),
        jnp.asarray(data.qpos.copy()),
        jnp.asarray(data.qvel.copy()),
        foot,
    )

    x0 = measured_augmented_state(
        mpc,
        data,
        foot,
        jnp.asarray(INITIAL_DURATION, dtype=mpc.initial_state.dtype),
    )

    # Seed the nominal trajectory with a structured reference instead of a
    # horizon full of the same measured state.
    reference0, _, liftoff0 = mpc._ref_gen(
        duty_factor=mpc_data.duty_factor,
        step_freq=mpc_data.step_freq,
        step_height=mpc_data.step_height,
        t_timer=mpc_data.contact_time,
        x=x0,
        foot=x0[mpc.foot_slice],
        input=COMMAND,
        liftoff=mpc_data.liftoff,
        contact=contact,
        current_time=mpc_data.time,
    )
    X_guess = build_initial_guess(x0, reference0)
    U_guess = jnp.tile(config.u_ref, (config.N, 1))
    mpc_data = mpc_data.replace(
        X0=X_guess,
        U0=U_guess,
        dt=float(INITIAL_DURATION / config.N),
        liftoff=liftoff0,
    )

    # Compile once before reporting RTI execution times.  The returned state is
    # kept, so this is also the nominal initialization step.
    t_compile = timer()
    mpc_data, tau, X_pred, U_pred, dt_apply, valid = solve_rti(
        mpc_data,
        x0,
        COMMAND,
        contact,
    )
    tau.block_until_ready()
    print(f"Initial compile/solve: {1e3 * (timer() - t_compile):.2f} ms")

    # ------------------------------------------------------------------
    # Closed-loop receding-horizon execution
    # ------------------------------------------------------------------
    control_times_ms = []
    duration_history = []
    position_history = [np.asarray(data.qpos[:3]).copy()]
    forward_velocity_history = [float(data.qvel[0])]

    # Store states at a fixed VIDEO_FPS without rendering during control.
    video_fps = 60
    frame_dt = 1.0 / video_fps
    next_frame_time = 0.0
    qpos_history = [np.asarray(data.qpos).copy()]
    qvel_history = [np.asarray(data.qvel).copy()]
    next_frame_time += frame_dt

    viewer_ctx = None
    viewer = None
    if not headless:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_ctx.__enter__()
        viewer.sync()

    try:
        for iteration in range(int(steps)):
            # The first iteration already has a freshly computed RTI command
            # from the initialization call above. Later iterations remeasure
            # and solve exactly one SQP step.
            if iteration > 0:
                foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
                contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

                duration_guess = jnp.clip(
                    mpc_data.X0[0, -1],
                    MIN_TIME,
                    MAX_TIME,
                )
                x0 = measured_augmented_state(
                    mpc,
                    data,
                    foot,
                    duration_guess,
                )

                tic = timer()
                mpc_data, tau, X_pred, U_pred, dt_apply, valid = solve_rti(
                    mpc_data,
                    x0,
                    COMMAND,
                    contact,
                )
                tau.block_until_ready()
                toc = timer()
                control_times_ms.append(1e3 * (toc - tic))

            dt_apply_float = float(np.asarray(dt_apply))
            duration_now = float(np.asarray(X_pred[0, -1]))
            duration_history.append(duration_now)

            tau_np = np.asarray(
                jnp.clip(tau, config.min_torque, config.max_torque)
            )
            data.ctrl[:] = tau_np

            start_sim_time = float(data.time)
            target_sim_time = start_sim_time + dt_apply_float

            # Hold U[0] for exactly one optimized shooting interval, using
            # MuJoCo's much smaller fixed physics step internally.
            while data.time < target_sim_time - 0.5 * sim_dt:
                mujoco.mj_step(model, data)

                while data.time >= next_frame_time:
                    qpos_history.append(np.asarray(data.qpos).copy())
                    qvel_history.append(np.asarray(data.qvel).copy())
                    next_frame_time += frame_dt

                if viewer is not None and viewer.is_running():
                    viewer.sync()

            elapsed = float(data.time) - start_sim_time
            mpc_data = advance_clock(
                mpc_data,
                jnp.asarray(elapsed, dtype=mpc_data.time.dtype),
            )

            position_history.append(np.asarray(data.qpos[:3]).copy())
            forward_velocity_history.append(float(data.qvel[0]))

            print(
                f"RTI {iteration:03d} | "
                f"valid={bool(np.asarray(valid))} | "
                f"T_rem={duration_now:.4f} s | "
                f"dt={dt_apply_float:.4f} s | "
                f"vx={float(data.qvel[0]):.3f} m/s | "
                f"p={np.asarray(data.qpos[:3])}"
            )

            if viewer is not None and not viewer.is_running():
                break

    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    position_history = np.asarray(position_history)

    forward_velocity_history = np.asarray(forward_velocity_history)
    forward_distance = float(position_history[-1, 0] - position_history[0, 0])
    simulated_time = float(data.time)
    average_forward_speed = (
        forward_distance / simulated_time if simulated_time > 0.0 else 0.0
    )

    print("\nClosed-loop forward-speed RTI summary")
    print(f"  simulated time:       {simulated_time:.6f} s")
    print(f"  final position:       {np.asarray(data.qpos[:3])}")
    print(f"  forward distance:     {forward_distance:.6f} m")
    print(f"  average forward speed:{average_forward_speed: .6f} m/s")
    print(f"  final forward speed:  {float(data.qvel[0]):.6f} m/s")
    print(f"  peak forward speed:   {forward_velocity_history.max():.6f} m/s")

    if control_times_ms:
        timings = np.asarray(control_times_ms)
        print(f"  RTI mean:       {timings.mean():.3f} ms")
        print(f"  RTI median:     {np.median(timings):.3f} ms")
        print(f"  RTI max:        {timings.max():.3f} ms")

    if duration_history:
        print(f"  initial optimized T:  {duration_history[0]:.6f} s")
        print(f"  final optimized T:    {duration_history[-1]:.6f} s")
        print(
            f"  final optimized dt:   "
            f"{duration_history[-1] / config.N:.6f} s"
        )

    if save_video:
        encode_state_history_video(
            model,
            qpos_history,
            qvel_history,
            filename="quadruped_fast_forward_rti.mp4",
            video_fps=video_fps,
            width=1280,
            height=720,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Maximum number of SQP-RTI MPC updates.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable writing quadruped_fast_forward_rti.mp4.",
    )
    args = parser.parse_args()

    main(
        headless=args.headless,
        steps=args.steps,
        save_video=not args.no_video,
    )
