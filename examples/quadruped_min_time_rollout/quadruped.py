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
# Minimum-time experiment configuration
# -----------------------------------------------------------------------------
# One optimized duration state T is appended to the original whole-body state.
NUM_PHASES = 1

# The nominal horizon length is still config.N.  T changes the physical spacing
# between shooting nodes: dt_k = T / N.
INITIAL_DURATION = 0.80
MIN_TIME = 0.15
MAX_TIME = 2.00
TIME_WEIGHT = 5.0

# Fixed terminal-set task, analogous to the quadrotor minimum-time experiment.
GOAL_CENTER = jnp.array([0.50, 0.0, config.robot_height])
GOAL_HALF_WIDTH = jnp.array([0.06, 0.08, 0.05])

# Optional obstacle in the horizontal plane.
OBSTACLE_CENTER = jnp.array([0.25, 0.18])
OBSTACLE_RADIUS = 0.12
OBSTACLE_VISUAL_RADIUS = 0.08
OBSTACLE_VISUAL_HALF_HEIGHT = 0.25

# A forward-velocity command is still useful to the MPX footstep generator.
# The base-position reference itself is overwritten below with a straight line
# from the measured initial position to GOAL_CENTER.
COMMAND = jnp.array([
    0.60, 0.0, 0.0,
    0.0, 0.0, 0.0,
    config.robot_height,
])

# Preserve the original GO2 whole-body state dimension before augmenting it.
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
GRF_START = 13 + 2 * config.n_joints + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact


# -----------------------------------------------------------------------------
# Minimum-time whole-body dynamics
# -----------------------------------------------------------------------------
def min_time_quadruped_dynamics(model, mjx_model, contact_id, body_id):
    """Return f(x,u,t,parameter) with x[-1] = optimized total duration T."""

    n_joints = config.n_joints
    dtau = 1.0 / config.N

    def dynamics(x, u, t, parameter):
        # Optimized physical step length.
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
# Minimum-time cost
# -----------------------------------------------------------------------------
def min_time_quadruped_cost(W, reference, x, u, t):
    """Original whole-body regularization + a dominant minimum-time term."""

    nj = config.n_joints
    nc = config.n_contact

    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + nj]
    dp = x[7 + nj : 10 + nj]
    omega = x[10 + nj : 13 + nj]
    dq = x[13 + nj : 13 + 2 * nj]
    p_leg = x[13 + 2 * nj : 13 + 2 * nj + 3 * nc]

    # IMPORTANT: stop before the appended duration state.
    grf = x[GRF_START:GRF_STOP]
    T = x[-1]

    tau = u[:nj]

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    contact = reference["contact"][t]
    grf_ref = reference["grf"][t]

    # Keep the same friction regularization as the MPX whole-body objective.
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

    stage_cost = (
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

    terminal_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quat_error.T @ W["rot"] @ quat_error
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
    )

    tracking_cost = jnp.where(
        t == config.N,
        0.5 * terminal_cost,
        0.5 * stage_cost,
    )

    # Same structure as the quadrotor reference: T is present at every node,
    # so summing stage costs produces a positive constant multiple of T.
    return tracking_cost + W["time"] * T


# -----------------------------------------------------------------------------
# Reference generation
# -----------------------------------------------------------------------------
# Rebuild the MPX reference generator with a nominal dt consistent with the
# INITIAL_DURATION.  The contact schedule is then fixed in shooting-index space.
_BASE_REFERENCE_GENERATOR = partial(
    mpc_utils.reference_generator,
    config.use_terrain_estimation,
    config.N,
    INITIAL_DURATION / config.N,
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

    # Straight-line spatial reference, analogous to build_piecewise_reference
    # in the quadrotor experiment.
    s = jnp.linspace(0.0, 1.0, config.N + 1, dtype=x.dtype)
    p0 = x[:3]
    p_ref = (1.0 - s[:, None]) * p0 + s[:, None] * GOAL_CENTER

    # Do not impose an aggressive velocity reference that fights minimum time.
    # Keep it as a mild constant nominal velocity and use a small velocity weight.
    nominal_velocity = (GOAL_CENTER - p0) / INITIAL_DURATION
    dp_ref = jnp.tile(nominal_velocity, (config.N + 1, 1))

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
    # Duration bounds.
    time_constraints = jnp.array([
        MIN_TIME - x[-1],
        x[-1] - MAX_TIME,
    ])

    # Joint torque bounds.
    torque_constraints = jnp.concatenate([
        u - config.max_torque,
        config.min_torque - u,
    ])

    # Terminal hyperbox in base xyz.
    terminal = jnp.concatenate([
        x[:3] - GOAL_CENTER - GOAL_HALF_WIDTH,
        GOAL_CENTER - x[:3] - GOAL_HALF_WIDTH,
    ])
    terminal = jnp.where(
        t == config.N,
        terminal,
        -jnp.ones_like(terminal),
    )

    # Horizontal circular obstacle.
    dxy = x[:2] - OBSTACLE_CENTER
    obstacle = jnp.reshape(
        OBSTACLE_RADIUS**2 - dxy @ dxy,
        (1,),
    )

    return jnp.concatenate([
        time_constraints,
        torque_constraints,
        terminal,
        obstacle,
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
# Configure config_go2 for the augmented minimum-time state before creating MPC
# -----------------------------------------------------------------------------
def configure_min_time_problem():
    # Exactly one new state: T.
    base_initial_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.n = PHYSICAL_N + NUM_PHASES
    config.nu = config.n_joints
    config.initial_state = jnp.concatenate([
        base_initial_state,
        jnp.array([INITIAL_DURATION], dtype=base_initial_state.dtype),
    ])

    # Use the minimum-time-aware components defined above.
    config.dynamics = min_time_quadruped_dynamics
    config.cost = min_time_quadruped_cost
    config.reference_generator = min_time_reference_generator

    # The original MPX weights are very large for velocity tracking.  That can
    # oppose minimum-time behavior, so use gentler tracking regularization while
    # retaining posture/contact structure.
    dtype = config.initial_state.dtype
    config.W = {
        "pos": jnp.diag(jnp.array([1.0, 1.0, 10.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([20.0, 20.0, 2.0], dtype=dtype)),
        "q": jnp.eye(config.n_joints, dtype=dtype) * 1e-1,
        "vel": jnp.diag(jnp.array([0.05, 0.05, 0.1], dtype=dtype)),
        "omega": jnp.eye(3, dtype=dtype) * 0.1,
        "dq": jnp.eye(config.n_joints, dtype=dtype) * 1e-2,
        "contact": jnp.eye(3 * config.n_contact, dtype=dtype) * 10.0,
        "tau": jnp.eye(config.n_joints, dtype=dtype) * 1e-3,
        "grf": jnp.eye(3 * config.n_contact, dtype=dtype) * 1e-4,
        "time": jnp.asarray(TIME_WEIGHT, dtype=dtype),
    }


# -----------------------------------------------------------------------------
# One-shot minimum-time solve
# -----------------------------------------------------------------------------
def solve_min_time_plan(mpc, mpc_data, x0, command, contact):
    # Generate the fixed contact/foot reference and straight-line base reference.
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

    X_guess = build_initial_guess(x0, reference)
    U_guess = jnp.tile(config.u_ref, (config.N, 1))

    result = mpc._solve(
        reference,
        parameter,
        mpc_data.W,
        x0,
        X_guess,
        U_guess,
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

    return reference, parameter, liftoff, result


def render_predicted_state_video(
    model,
    X_pred,
    optimized_time,
    filename="quadruped_x_pred.mp4",
    width=1280,
    height=720,
):
    """Render one MuJoCo frame for every predicted horizon state."""
    states = np.asarray(X_pred)
    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("X_pred must contain at least two horizon states.")
    if states.shape[1] < model.nq + model.nv:
        raise ValueError(
            f"X_pred has {states.shape[1]} columns, but MuJoCo needs at "
            f"least nq + nv = {model.nq + model.nv}."
        )

    video_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([
        0.25,
        0.0,
        config.robot_height * 0.5,
    ])
    camera.distance = 2.0
    camera.azimuth = 135
    camera.elevation = -20

    frames = []
    try:
        for state in states:
            # The optimizer state begins with MuJoCo qpos followed by qvel.
            # No physics step is taken: the robot is teleported to X_pred[k].
            video_data.qpos[:] = state[:model.nq]
            video_data.qvel[:] = state[model.nq : model.nq + model.nv]
            mujoco.mj_normalizeQuat(model, video_data.qpos)
            mujoco.mj_forward(model, video_data)
            renderer.update_scene(video_data, camera=camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    prediction_fps = max(
        1.0,
        float(states.shape[0] - 1) / max(float(optimized_time), 1.0e-9),
    )
    frame_height, frame_width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{frame_width}x{frame_height}",
        "-framerate", str(prediction_fps),
        "-i", "-", "-an", "-vcodec", "libx264",
        "-pix_fmt", "yuv420p", str(filename),
    ]
    encoder = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        for frame in frames:
            encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        encoder.stdin.close()
        error_output = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
    except Exception:
        encoder.kill()
        encoder.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}: {error_output}"
        )

    print(f"Saved X_pred MuJoCo animation to: {filename}")
    print(f"Prediction frames: {len(frames)} at {prediction_fps:.3f} fps")


def render_rollout_video(
    model,
    U_pred,
    dt_opt,
    filename="quadruped_rollout.mp4",
    video_fps=60,
    width=1280,
    height=720,
):
    """
    Replay the optimized control sequence in MuJoCo and save an MP4.

    Physics runs at model.opt.timestep.
    Video frames are sampled independently at `video_fps`.
    """

    # Fresh simulation state
    video_data = mujoco.MjData(model)

    video_data.qpos = np.asarray(
        jnp.concatenate([
            config.p0,
            config.quat0,
            config.q0,
        ])
    )

    video_data.qvel[:] = 0.0
    video_data.ctrl[:] = 0.0
    video_data.time = 0.0

    mujoco.mj_forward(model, video_data)

    # ---------------------------------------------------------
    # Offscreen renderer
    # ---------------------------------------------------------
    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )

    # Camera
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    camera.lookat[:] = np.array([
        0.25,
        0.0,
        config.robot_height * 0.5,
    ])

    camera.distance = 2.0
    camera.azimuth = 135
    camera.elevation = -20

    frames = []

    frame_dt = 1.0 / video_fps
    next_frame_time = 0.0

    N = min(config.N, U_pred.shape[0])

    # ---------------------------------------------------------
    # Initial frame
    # ---------------------------------------------------------
    renderer.update_scene(
        video_data,
        camera=camera,
    )

    frames.append(renderer.render().copy())
    next_frame_time += frame_dt

    # ---------------------------------------------------------
    # Optimized rollout
    # ---------------------------------------------------------
    for k in range(N):

        tau = jnp.clip(
            U_pred[k],
            config.min_torque,
            config.max_torque,
        )

        video_data.ctrl[:] = np.asarray(tau)

        # Optimized shooting-node time
        target_time = (k + 1) * dt_opt

        # Run MuJoCo physics until the next shooting node
        while video_data.time < target_time:

            mujoco.mj_step(
                model,
                video_data,
            )

            # Render whenever simulated time crosses the
            # next video-frame timestamp.
            if video_data.time >= next_frame_time:

                renderer.update_scene(
                    video_data,
                    camera=camera,
                )

                frame = renderer.render()

                frames.append(
                    np.asarray(frame).copy()
                )

                next_frame_time += frame_dt

    renderer.close()

    # ---------------------------------------------------------
    # Encode MP4
    # ---------------------------------------------------------
    frame_height, frame_width = frames[0].shape[:2]
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{frame_width}x{frame_height}",
        "-framerate",
        str(video_fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(filename),
    ]

    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            encoder.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        encoder.stdin.close()
        error_output = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
    except Exception:
        encoder.kill()
        encoder.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}: {error_output}"
        )

    print(f"Saved rollout video to: {filename}")
    print(f"Video frames: {len(frames)}")
    print(f"Simulation duration: {video_data.time:.4f} s")


def main(headless=False, steps=None):
    if _auto_headless and not headless:
        print(
            "Interactive MuJoCo rendering is unavailable in this display "
            "session; using EGL offscreen rendering instead."
        )
        headless = True

    configure_min_time_problem()

    mpx_root = Path(mpx.__file__).parent
    model = mujoco.MjModel.from_xml_path(
        str(mpx_root / "data" / "go2" / "scene_mjx.xml")
    )

    # Increase MuJoCo offscreen framebuffer size
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720

    data = mujoco.MjData(model)

    sim_frequency = 200.0
    model.opt.timestep = 1.0 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    # ------------------------------------------------------------------
    # Solver configuration -- mirrors the quadrotor minimum-time setup.
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
        max_initial_sqp_iterations=100,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=50,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=1e-10,
        line_search=True,
    )
    sqp_cfg = force_integer_num_phases(sqp_cfg)

    E_mag = 0.05
    disturbance = make_min_time_disturbance(
        n=config.n,
        N=config.N,
        E_mag=E_mag,
    )

    # Obstacle is included directly in min_time_constraints, so keep this empty
    # to match the current GenericMPC/legged wrapper accounting convention.
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

    # ------------------------------------------------------------------
    # Initial measured state
    # ------------------------------------------------------------------
    data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

    mpc_data = mpc.reset(
        mpc.make_data(),
        jnp.asarray(data.qpos.copy()),
        jnp.asarray(data.qvel.copy()),
        foot,
    )

    x0 = (
        mpc.initial_state
        .at[mpc.qpos_slice].set(jnp.asarray(data.qpos.copy()))
        .at[mpc.qvel_slice].set(jnp.asarray(data.qvel.copy()))
        .at[mpc.foot_slice].set(foot)
        .at[-1].set(INITIAL_DURATION)
    )

    # ------------------------------------------------------------------
    # Compile + solve the minimum-time OCP once, analogous to the quadrotor
    # experiment's controller.run(...).
    # ------------------------------------------------------------------
    start = timer()
    reference, parameter, liftoff, result = solve_min_time_plan(
        mpc,
        mpc_data,
        x0,
        COMMAND,
        contact,
    )

    (
        X_pred,
        U_pred,
        V_pred,
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
    ) = result

    X_pred.block_until_ready()
    stop = timer()

    min_time = float(X_pred[0, -1])
    dt_opt = min_time / config.N

    render_predicted_state_video(
        model=model,
        X_pred=X_pred,
        optimized_time=min_time,
        filename="quadruped_x_pred.mp4",
    )

    render_rollout_video(
        model=model,
        U_pred=U_pred,
        dt_opt=dt_opt,
        filename="quadruped_min_time_rollout.mp4",
        video_fps=60,
    )

    print(f"Solve time: {1e3 * (stop - start):.2f} ms")
    print(f"Computed minimum time: {min_time:.6f} s")
    print(f"Optimized shooting dt: {dt_opt:.6f} s")
    print(f"Terminal position: {np.asarray(X_pred[-1, :3])}")
    print(f"Goal center:       {np.asarray(GOAL_CENTER)}")
    print(f"Terminal error:    {np.asarray(X_pred[-1, :3] - GOAL_CENTER)}")
    print(f"ADMM converged:    {bool(np.asarray(converged_admm))}")

    # ------------------------------------------------------------------
    # Open-loop replay at the optimized physical timing.
    # This makes the simulation timing consistent with dt = T / N.
    # ------------------------------------------------------------------
    data.qpos = np.asarray(jnp.concatenate([config.p0, config.quat0, config.q0]))
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)

    max_controls = config.N if steps is None else min(int(steps), config.N)
    sim_dt = model.opt.timestep

    def apply_control(k):
        tau = jnp.clip(U_pred[k], config.min_torque, config.max_torque)
        data.ctrl = np.asarray(tau)

        # Hit each optimized shooting time as closely as the fixed MuJoCo
        # simulation timestep permits.
        target_time = (k + 1) * dt_opt
        while data.time < target_time - 0.5 * sim_dt:
            mujoco.mj_step(model, data)

    if headless:
        trajectory = [np.asarray(data.qpos[:3]).copy()]
        for k in range(max_controls):
            apply_control(k)
            trajectory.append(np.asarray(data.qpos[:3]).copy())

        trajectory = np.asarray(trajectory)
        print(f"Replay final time: {data.time:.6f} s")
        print(f"Replay final position: {trajectory[-1]}")
        print(f"Replay goal error: {trajectory[-1] - np.asarray(GOAL_CENTER)}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            # Obstacle marker.
            obstacle_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                obstacle_geom,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([
                    OBSTACLE_VISUAL_RADIUS,
                    OBSTACLE_VISUAL_HALF_HEIGHT,
                    0.0,
                ]),
                np.array([
                    float(OBSTACLE_CENTER[0]),
                    float(OBSTACLE_CENTER[1]),
                    OBSTACLE_VISUAL_HALF_HEIGHT,
                ]),
                np.eye(3).ravel(),
                np.array([0.9, 0.1, 0.1, 0.35], dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1

            # Goal marker.
            goal_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                goal_geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.05, 0.05, 0.05]),
                np.asarray(GOAL_CENTER, dtype=np.float64),
                np.eye(3).ravel(),
                np.array([0.1, 0.8, 0.1, 0.5], dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1

        viewer.sync()

        for k in range(max_controls):
            if not viewer.is_running():
                break

            tic = timer()
            apply_control(k)
            viewer.sync()
            toc = timer()

            # This sleep is only for visualization; it does not change MuJoCo's
            # simulated time because the physics was already stepped above.
            wall_interval = dt_opt
            if toc - tic < wall_interval:
                time.sleep(wall_interval - (toc - tic))

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum number of optimized control intervals to replay.",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main(
        headless=args.headless,
        steps=args.steps,
    )
