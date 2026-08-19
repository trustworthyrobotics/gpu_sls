# Adapted from MPX's MJX quadruped example and the user's minimum-time quadrotor experiment.

import argparse
import os
import sys
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


# =============================================================================
# Minimum-time experiment configuration
# =============================================================================

# One optimized duration state T.
NUM_PHASES = 1

# -------------------------------------------------------------------------
# Receding goal
# -------------------------------------------------------------------------
# At every MPC update:
#
#       goal_x = measured_robot_x + LOOKAHEAD_DISTANCE
#
# The goal stays fixed during that individual OCP solve, then moves forward
# again at the next MPC update.
LOOKAHEAD_DISTANCE = 0.50

# Keep the robot close to the centerline and nominal height.
GOAL_Y = 0.0
GOAL_Z = config.robot_height

GOAL_HALF_WIDTH = jnp.array([
    0.06,   # x tolerance
    0.08,   # y tolerance
    0.05,   # z tolerance
])


# -------------------------------------------------------------------------
# Time optimization
# -------------------------------------------------------------------------
INITIAL_DURATION = 0.80
MIN_TIME = 0.01
MAX_TIME = 20.00

TIME_WEIGHT = 5.0


# -------------------------------------------------------------------------
# Optional obstacle
# -------------------------------------------------------------------------
OBSTACLE_CENTER = jnp.array([0.25, 0.18])
OBSTACLE_RADIUS = 0.12

OBSTACLE_VISUAL_RADIUS = 0.08
OBSTACLE_VISUAL_HALF_HEIGHT = 0.25


# -------------------------------------------------------------------------
# Gait-generator command
# -------------------------------------------------------------------------
# This is still useful for generating a sensible contact / footstep pattern.
# The actual forward objective is supplied by the moving terminal set.
COMMAND = jnp.array([
    0.60, 0.0, 0.0,
    0.0, 0.0, 0.0,
    config.robot_height,
])


# =============================================================================
# State indexing
# =============================================================================

# Original MPX physical whole-body state:
#
#   [qpos,
#    qvel,
#    foot positions,
#    GRFs]
#
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact

# We append:
#
#   goal_x
#   T
#
# so the augmented state is:
#
#   x_aug = [
#       x_physical,
#       goal_x,
#       T
#   ]
#
GOAL_X_IDX = PHYSICAL_N
T_IDX = PHYSICAL_N + 1

AUGMENTED_N = PHYSICAL_N + 2


# GRF indices remain entirely inside the original physical state.
GRF_START = (
    13
    + 2 * config.n_joints
    + 3 * config.n_contact
)

GRF_STOP = (
    GRF_START
    + 3 * config.n_contact
)


# =============================================================================
# Goal helper
# =============================================================================

def goal_from_state(x):
    """
    Construct the current fixed goal for an OCP horizon.

    goal_x is stored as a constant augmented state.
    y and z targets remain fixed.
    """
    return jnp.array([
        x[GOAL_X_IDX],
        GOAL_Y,
        GOAL_Z,
    ], dtype=x.dtype)


# =============================================================================
# Minimum-time whole-body dynamics
# =============================================================================

def min_time_quadruped_dynamics(
    model,
    mjx_model,
    contact_id,
    body_id,
):
    """
    Whole-body quadruped dynamics with augmented states

        x_aug = [x_physical, goal_x, T]

    Dynamics:

        physical state -> MuJoCo whole-body dynamics
        goal_x(k+1)    = goal_x(k)
        T(k+1)         = T(k)

    The physical integration step is

        dt = T / N
    """

    n_joints = config.n_joints
    dtau = 1.0 / config.N

    def dynamics(x, u, t, parameter):

        # -------------------------------------------------------------
        # Augmented states
        # -------------------------------------------------------------
        goal_x = x[GOAL_X_IDX]
        T = x[T_IDX]

        # Optimized physical shooting interval.
        dt = dtau * T


        # -------------------------------------------------------------
        # Extract MuJoCo physical state
        # -------------------------------------------------------------
        qpos = x[: n_joints + 7]

        qvel = x[
            n_joints + 7 :
            2 * n_joints + 13
        ]


        # -------------------------------------------------------------
        # Build MJX state
        # -------------------------------------------------------------
        mjx_data = mjx.make_data(model)

        mjx_data = mjx_data.replace(
            qpos=qpos,
            qvel=qvel,
        )

        mjx_data = mjx.fwd_position(
            mjx_model,
            mjx_data,
        )

        mjx_data = mjx.fwd_velocity(
            mjx_model,
            mjx_data,
        )


        # -------------------------------------------------------------
        # Manipulator dynamics
        # -------------------------------------------------------------
        M = mjx_data.qLD
        D = mjx_data.qfrc_bias


        # parameter remains the MPX contact schedule.
        contact = parameter[t, :4]


        # Floating-base system:
        #
        #   first 6 entries = no direct actuation
        #   remaining entries = joint torques
        tau = jnp.concatenate([
            jnp.zeros(6, dtype=x.dtype),
            u,
        ])


        # -------------------------------------------------------------
        # Foot positions
        # -------------------------------------------------------------
        FL_leg = mjx_data.geom_xpos[contact_id[0]]
        FR_leg = mjx_data.geom_xpos[contact_id[1]]
        RL_leg = mjx_data.geom_xpos[contact_id[2]]
        RR_leg = mjx_data.geom_xpos[contact_id[3]]


        # -------------------------------------------------------------
        # Contact Jacobians
        # -------------------------------------------------------------
        J_FL, _ = mjx.jac(
            mjx_model,
            mjx_data,
            FL_leg,
            body_id[0],
        )

        J_FR, _ = mjx.jac(
            mjx_model,
            mjx_data,
            FR_leg,
            body_id[1],
        )

        J_RL, _ = mjx.jac(
            mjx_model,
            mjx_data,
            RL_leg,
            body_id[2],
        )

        J_RR, _ = mjx.jac(
            mjx_model,
            mjx_data,
            RR_leg,
            body_id[3],
        )

        J = jnp.concatenate(
            [
                J_FL,
                J_FR,
                J_RL,
                J_RR,
            ],
            axis=1,
        )


        if len(contact_id) > 4:
            current_leg = jnp.asarray([
                mjx_data.geom_xpos[int(geom_id)]
                for geom_id in contact_id
            ]).flatten()
        else:
            current_leg = jnp.concatenate(
                [
                    FL_leg,
                    FR_leg,
                    RL_leg,
                    RR_leg,
                ],
                axis=0,
            )


        # -------------------------------------------------------------
        # Contact-force solve
        # -------------------------------------------------------------
        alpha = 25.0

        g_dot = J.T @ qvel
        baumgarte_term = -2.0 * alpha * g_dot


        M_inv_J = jax.scipy.linalg.cho_solve(
            (M, False),
            J,
        )

        JT_M_invJ = J.T @ M_inv_J


        rhs = (
            -J.T
            @ jax.scipy.linalg.cho_solve(
                (M, False),
                tau - D,
            )
            + baumgarte_term
        )


        cho_JT_M_invJ = jax.scipy.linalg.cho_factor(
            JT_M_invJ
        )

        grf = jax.scipy.linalg.cho_solve(
            cho_JT_M_invJ,
            rhs,
        )


        # Activate/deactivate contact forces according to contact schedule.
        grf = jnp.concatenate([
            grf[0:3] * contact[0],
            grf[3:6] * contact[1],
            grf[6:9] * contact[2],
            grf[9:12] * contact[3],
        ])


        # -------------------------------------------------------------
        # Generalized acceleration
        # -------------------------------------------------------------
        acceleration = jax.scipy.linalg.cho_solve(
            (M, False),
            tau - D + J @ grf,
        )


        # -------------------------------------------------------------
        # Semi-implicit Euler
        # -------------------------------------------------------------
        v_next = qvel + acceleration * dt

        p_next = (
            qpos[:3]
            + v_next[:3] * dt
        )

        quat_next = math.quat_integrate(
            qpos[3:7],
            v_next[3:6],
            dt,
        )

        q_next = (
            qpos[
                7 :
                7 + n_joints
            ]
            + v_next[
                6 :
                6 + n_joints
            ] * dt
        )


        physical_next = jnp.concatenate([
            p_next,
            quat_next,
            q_next,
            v_next,
            current_leg,
            grf,
        ])


        # -------------------------------------------------------------
        # Augmented dynamics
        # -------------------------------------------------------------
        #
        # goal_x(k+1) = goal_x(k)
        # T(k+1)      = T(k)
        #
        return jnp.concatenate([
            physical_next,
            jnp.reshape(goal_x, (1,)),
            jnp.reshape(T, (1,)),
        ])

    return dynamics


# =============================================================================
# Minimum-time cost
# =============================================================================

def min_time_quadruped_cost(
    W,
    reference,
    x,
    u,
    t,
):
    """
    Whole-body gait/posture regularization plus minimum-time objective.

    The moving forward target itself is enforced primarily through the
    terminal-set constraint.
    """

    nj = config.n_joints
    nc = config.n_contact


    # -------------------------------------------------------------
    # Physical state
    # -------------------------------------------------------------
    p = x[:3]

    quat = x[3:7]

    q = x[
        7 :
        7 + nj
    ]

    dp = x[
        7 + nj :
        10 + nj
    ]

    omega = x[
        10 + nj :
        13 + nj
    ]

    dq = x[
        13 + nj :
        13 + 2 * nj
    ]

    p_leg = x[
        13 + 2 * nj :
        13 + 2 * nj + 3 * nc
    ]

    grf = x[
        GRF_START :
        GRF_STOP
    ]

    T = x[T_IDX]

    tau = u[:nj]


    # -------------------------------------------------------------
    # Reference
    # -------------------------------------------------------------
    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    p_leg_ref = reference["foot"][t]
    contact = reference["contact"][t]
    grf_ref = reference["grf"][t]


    # -------------------------------------------------------------
    # Friction regularization
    # -------------------------------------------------------------
    mu_friction = 0.5

    friction_cone = (
        mu_friction * grf[2::3]
        - jnp.sqrt(
            jnp.square(grf[1::3])
            + jnp.square(grf[0::3])
            + jnp.ones(
                nc,
                dtype=x.dtype,
            ) * 1e-2
        )
    )

    friction_penalty = jnp.sum(
        mpc_objectives.penalty(
            friction_cone
        ) * contact
    )


    quat_error = math.quat_sub(
        quat,
        quat_ref,
    )

    contact_map = jnp.ones(
        3 * nc,
        dtype=x.dtype,
    )


    # -------------------------------------------------------------
    # Running regularization
    # -------------------------------------------------------------
    stage_cost = (
        (p - p_ref).T
        @ W["pos"]
        @ (p - p_ref)

        + quat_error.T
        @ W["rot"]
        @ quat_error

        + (q - q_ref).T
        @ W["q"]
        @ (q - q_ref)

        + (dp - dp_ref).T
        @ W["vel"]
        @ (dp - dp_ref)

        + (omega - omega_ref).T
        @ W["omega"]
        @ (omega - omega_ref)

        + dq.T
        @ W["dq"]
        @ dq

        + (
            contact_map
            * (p_leg - p_leg_ref)
        ).T
        @ W["contact"]
        @ (
            contact_map
            * (p_leg - p_leg_ref)
        )

        + tau.T
        @ W["tau"]
        @ tau

        + (grf - grf_ref).T
        @ W["grf"]
        @ (grf - grf_ref)

        + friction_penalty
    )


    # -------------------------------------------------------------
    # Terminal regularization
    # -------------------------------------------------------------
    terminal_cost = (
        (p - p_ref).T
        @ W["pos"]
        @ (p - p_ref)

        + quat_error.T
        @ W["rot"]
        @ quat_error

        + (q - q_ref).T
        @ W["q"]
        @ (q - q_ref)

        + (dp - dp_ref).T
        @ W["vel"]
        @ (dp - dp_ref)

        + (omega - omega_ref).T
        @ W["omega"]
        @ (omega - omega_ref)

        + dq.T
        @ W["dq"]
        @ dq
    )


    tracking_cost = jnp.where(
        t == config.N,
        0.5 * terminal_cost,
        0.5 * stage_cost,
    )


    # Positive T penalty gives minimum-time pressure.
    return (
        tracking_cost
        + W["time"] * T
    )


# =============================================================================
# Reference generation
# =============================================================================

# MPX reference_generator uses dt as a static argument.
#
# This is ONLY the nominal gait-reference timing.
# The actual OCP dynamics uses optimized dt = T / N.
REFERENCE_DT = float(
    INITIAL_DURATION / config.N
)


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


def min_time_reference_generator(
    *,
    mass,
    x,
    **kwargs,
):
    """
    Generate a reference from the currently measured position to the current
    receding target.

    At MPC update j:

        target_x = measured_x_j + 0.5
    """

    ref, parameter, liftoff = _BASE_REFERENCE_GENERATOR(
        mass=mass,
        x=x,
        **kwargs,
    )


    # -------------------------------------------------------------
    # Current state
    # -------------------------------------------------------------
    p0 = x[:3]


    # -------------------------------------------------------------
    # Receding target
    # -------------------------------------------------------------
    goal = goal_from_state(x)


    # -------------------------------------------------------------
    # Straight-line position reference
    # -------------------------------------------------------------
    s = jnp.linspace(
        0.0,
        1.0,
        config.N + 1,
        dtype=x.dtype,
    )

    p_ref = (
        (1.0 - s[:, None]) * p0
        + s[:, None] * goal
    )


    # Nominal forward velocity used only as a mild reference.
    nominal_velocity = (
        goal - p0
    ) / INITIAL_DURATION

    dp_ref = jnp.tile(
        nominal_velocity,
        (config.N + 1, 1),
    )


    ref = {
        **ref,
        "p": p_ref,
        "dp": dp_ref,
    }


    return (
        ref,
        parameter,
        liftoff,
    )


# =============================================================================
# Inequality constraints
#
# g(x,u,t) <= 0
# =============================================================================

def min_time_constraints(
    x,
    u,
    t,
):
    # -------------------------------------------------------------
    # Duration bounds
    # -------------------------------------------------------------
    T = x[T_IDX]

    time_constraints = jnp.array([
        MIN_TIME - T,
        T - MAX_TIME,
    ])


    # -------------------------------------------------------------
    # Joint torque bounds
    # -------------------------------------------------------------
    torque_constraints = jnp.concatenate([
        u - config.max_torque,
        config.min_torque - u,
    ])


    # -------------------------------------------------------------
    # Moving terminal set
    # -------------------------------------------------------------
    goal = goal_from_state(x)


    terminal = jnp.concatenate([
        x[:3]
        - goal
        - GOAL_HALF_WIDTH,

        goal
        - x[:3]
        - GOAL_HALF_WIDTH,
    ])


    # Only impose target at final shooting node.
    terminal = jnp.where(
        t == config.N,
        terminal,
        -jnp.ones_like(terminal),
    )


    # -------------------------------------------------------------
    # Horizontal circular obstacle
    # -------------------------------------------------------------
    dxy = (
        x[:2]
        - OBSTACLE_CENTER
    )

    obstacle = jnp.reshape(
        OBSTACLE_RADIUS**2
        - dxy @ dxy,
        (1,),
    )


    return jnp.concatenate([
        time_constraints,
        torque_constraints,
        terminal,
        obstacle,
    ])


# =============================================================================
# Robust disturbance model
# =============================================================================

def make_min_time_disturbance(
    n: int,
    N: int,
    E_mag: float,
):
    """
    Disturb only physical base x/y.

    Neither goal_x nor T are disturbed.
    """

    def disturbance_at_state(x_k):

        dt = x_k[T_IDX] / N

        diag = jnp.zeros(
            n,
            dtype=x_k.dtype,
        )

        # Base x/y disturbance only.
        diag = diag.at[:2].set(
            E_mag * dt
        )

        # Explicitly zero augmented variables.
        diag = diag.at[GOAL_X_IDX].set(
            0.0
        )

        diag = diag.at[T_IDX].set(
            0.0
        )

        return jnp.diag(diag)


    def disturbance(X):
        return jax.vmap(
            disturbance_at_state
        )(X)


    disturbance.at_state = disturbance_at_state

    return disturbance


# =============================================================================
# Initial SQP trajectory
# =============================================================================

def build_initial_guess(
    x0,
    reference,
):
    """
    Structured initial nominal trajectory.

    goal_x and T remain constant along the initial horizon.
    """

    X0 = jnp.tile(
        x0,
        (config.N + 1, 1),
    )

    nj = config.n_joints
    nc = config.n_contact


    X0 = X0.at[:, :3].set(
        reference["p"]
    )

    X0 = X0.at[:, 3:7].set(
        reference["quat"]
    )

    X0 = X0.at[
        :,
        7 : 7 + nj
    ].set(
        reference["q"]
    )

    X0 = X0.at[
        :,
        7 + nj :
        10 + nj
    ].set(
        reference["dp"]
    )

    X0 = X0.at[
        :,
        10 + nj :
        13 + nj
    ].set(
        reference["omega"]
    )

    X0 = X0.at[
        :,
        13 + nj :
        13 + 2 * nj
    ].set(
        0.0
    )

    X0 = X0.at[
        :,
        13 + 2 * nj :
        13 + 2 * nj + 3 * nc
    ].set(
        reference["foot"]
    )

    X0 = X0.at[
        :,
        GRF_START :
        GRF_STOP
    ].set(
        reference["grf"]
    )


    # Constant moving target across this OCP horizon.
    X0 = X0.at[
        :,
        GOAL_X_IDX
    ].set(
        x0[GOAL_X_IDX]
    )


    # Initial optimized duration guess.
    X0 = X0.at[
        :,
        T_IDX
    ].set(
        INITIAL_DURATION
    )


    return X0


# =============================================================================
# SQP configuration helper
# =============================================================================

def force_integer_num_phases(cfg):
    """Avoid JAX slice errors if SQPConfig.num_phases was declared as 1.0."""

    if is_dataclass(cfg):
        names = {
            f.name
            for f in fields(cfg)
        }

        if "num_phases" in names:
            return replace(
                cfg,
                num_phases=1,
            )


    if hasattr(
        cfg,
        "num_phases",
    ):
        try:
            cfg.num_phases = 1
        except Exception:
            pass


    return cfg


# =============================================================================
# Configure config_go2 for augmented state
# =============================================================================

def configure_min_time_problem():

    # -------------------------------------------------------------
    # Preserve original physical state
    # -------------------------------------------------------------
    base_initial_state = jnp.asarray(
        config.initial_state
    )[:PHYSICAL_N]


    # -------------------------------------------------------------
    # New augmented state:
    #
    # [physical,
    #  goal_x,
    #  T]
    # -------------------------------------------------------------
    config.n = AUGMENTED_N
    config.nu = config.n_joints


    initial_goal_x = (
        config.p0[0]
        + LOOKAHEAD_DISTANCE
    )


    config.initial_state = jnp.concatenate([
        base_initial_state,

        jnp.array(
            [
                initial_goal_x,
                INITIAL_DURATION,
            ],
            dtype=base_initial_state.dtype,
        ),
    ])


    # -------------------------------------------------------------
    # Minimum-time components
    # -------------------------------------------------------------
    config.dynamics = min_time_quadruped_dynamics
    config.cost = min_time_quadruped_cost
    config.reference_generator = min_time_reference_generator


    # -------------------------------------------------------------
    # Cost weights
    # -------------------------------------------------------------
    dtype = config.initial_state.dtype


    config.W = {

        # Low forward x tracking weight allows the minimum-time objective
        # to determine how aggressively the robot advances.
        #
        # y remains regulated around the centerline.
        # z remains strongly regulated around standing height.
        "pos": jnp.diag(
            jnp.array(
                [
                    1.0,
                    1.0,
                    300.0,
                ],
                dtype=dtype,
            )
        ),

        "rot": jnp.diag(
            jnp.array(
                [
                    20.0,
                    20.0,
                    2.0,
                ],
                dtype=dtype,
            )
        ),

        "q": (
            jnp.eye(
                config.n_joints,
                dtype=dtype,
            )
            * 1e-1
        ),

        "vel": jnp.diag(
            jnp.array(
                [
                    0.05,
                    0.05,
                    0.1,
                ],
                dtype=dtype,
            )
        ),

        "omega": (
            jnp.eye(
                3,
                dtype=dtype,
            )
            * 0.1
        ),

        "dq": (
            jnp.eye(
                config.n_joints,
                dtype=dtype,
            )
            * 1e-2
        ),

        "contact": (
            jnp.eye(
                3 * config.n_contact,
                dtype=dtype,
            )
            * 10.0
        ),

        "tau": (
            jnp.eye(
                config.n_joints,
                dtype=dtype,
            )
            * 1e-3
        ),

        "grf": (
            jnp.eye(
                3 * config.n_contact,
                dtype=dtype,
            )
            * 1e-4
        ),

        "time": jnp.asarray(
            TIME_WEIGHT,
            dtype=dtype,
        ),
    }


# =============================================================================
# SQP-RTI helpers
# =============================================================================

def _solution_is_valid(*values):

    valid = jnp.asarray(True)

    for value in values:
        valid = jnp.logical_and(
            valid,
            jnp.all(
                jnp.isfinite(value)
            ),
        )

    return valid


def build_rti_solve_fn(mpc):
    """
    Build one receding-horizon SQP-RTI step.

    At every RTI update:

        new_goal_x = current_measured_x + 0.5

    x0 already contains this newly computed goal_x.

    We then overwrite goal_x throughout the shifted nominal trajectory so the
    warm start is consistent with the new receding target.
    """

    @jax.jit
    def solve_rti(
        mpc_data,
        x0,
        command,
        contact,
    ):

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # The previous warm start contains the previous MPC iteration's
        # moving target.
        #
        # Replace goal_x at EVERY shooting node with the newly measured
        # receding goal.
        # -------------------------------------------------------------
        X0 = mpc_data.X0.at[
            :,
            GOAL_X_IDX
        ].set(
            x0[GOAL_X_IDX]
        )

        mpc_data = mpc_data.replace(
            X0=X0
        )


        # -------------------------------------------------------------
        # Generate reference toward the new moving target
        # -------------------------------------------------------------
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


        # -------------------------------------------------------------
        # One SQP-RTI step
        # -------------------------------------------------------------
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


        # -------------------------------------------------------------
        # First control
        # -------------------------------------------------------------
        tau = jax.lax.cond(
            valid,
            lambda _: U[
                0,
                : config.n_joints
            ],
            lambda _: mpc_data.U0[
                0,
                : config.n_joints
            ],
            operand=None,
        )


        # -------------------------------------------------------------
        # Current optimized remaining horizon duration
        # -------------------------------------------------------------
        duration = jax.lax.cond(
            valid,
            lambda _: X[0, T_IDX],
            lambda _: mpc_data.X0[0, T_IDX],
            operand=None,
        )


        duration = jnp.clip(
            duration,
            MIN_TIME,
            MAX_TIME,
        )


        dt_apply = (
            duration
            / config.N
        )


        # -------------------------------------------------------------
        # Shift warm-start quantities
        # -------------------------------------------------------------
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


        # -------------------------------------------------------------
        # Remaining duration after applying one shooting interval
        # -------------------------------------------------------------
        remaining_duration = jnp.clip(
            duration - dt_apply,
            MIN_TIME,
            MAX_TIME,
        )

        X0 = X0.at[
            :,
            T_IDX
        ].set(
            remaining_duration
        )


        # Keep this iteration's goal consistent throughout the shifted
        # trajectory. At the NEXT MPC call this will be replaced with
        # measured_x + 0.5 again.
        X0 = X0.at[
            :,
            GOAL_X_IDX
        ].set(
            x0[GOAL_X_IDX]
        )


        mpc_data = mpc_data.replace(
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


        return (
            mpc_data,
            tau,
            X,
            U,
            dt_apply,
            valid,
        )

    return solve_rti


def build_clock_update_fn(mpc):
    """
    Advance the gait/MPC clock by the actual physical interval that was
    executed in MuJoCo.
    """

    @jax.jit
    def advance_clock(
        mpc_data,
        elapsed_dt,
    ):

        _, contact_time = mpc._timer_run(
            mpc_data.duty_factor,
            mpc_data.step_freq,
            mpc_data.contact_time,
            elapsed_dt,
        )


        return mpc_data.replace(
            time=(
                mpc_data.time
                + elapsed_dt
            ),
            contact_time=contact_time,
        )

    return advance_clock


# =============================================================================
# Measured augmented state
# =============================================================================

def measured_augmented_state(
    mpc,
    data,
    foot,
    duration_guess,
):
    """
    Construct:

        x_aug = [
            measured physical state,
            current_x + 0.5,
            duration_guess
        ]

    This is where the receding target is moved forward at every MPC update.
    """

    current_x = jnp.asarray(
        data.qpos[0]
    )


    # =============================================================
    # MOVING GOAL
    # =============================================================
    goal_x = (
        current_x
        + LOOKAHEAD_DISTANCE
    )


    return (
        mpc.initial_state

        .at[
            mpc.qpos_slice
        ].set(
            jnp.asarray(
                data.qpos.copy()
            )
        )

        .at[
            mpc.qvel_slice
        ].set(
            jnp.asarray(
                data.qvel.copy()
            )
        )

        .at[
            mpc.foot_slice
        ].set(
            jnp.ravel(foot)
        )

        .at[
            GOAL_X_IDX
        ].set(
            goal_x
        )

        .at[
            T_IDX
        ].set(
            duration_guess
        )
    )


# =============================================================================
# Video
# =============================================================================

def encode_state_history_video(
    model,
    qpos_history,
    qvel_history,
    filename="quadruped_rti_rollout.mp4",
    video_fps=60,
    width=1280,
    height=720,
):
    """Render recorded closed-loop states after the RTI rollout has finished."""

    if len(qpos_history) == 0:
        return


    video_data = mujoco.MjData(
        model
    )

    renderer = mujoco.Renderer(
        model,
        height=height,
        width=width,
    )


    camera = mujoco.MjvCamera()

    mujoco.mjv_defaultCamera(
        camera
    )


    camera.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )

    camera.lookat[:] = np.array([
        0.25,
        0.0,
        config.robot_height * 0.5,
    ])

    camera.distance = 2.0
    camera.azimuth = 135
    camera.elevation = -20


    process = None

    try:

        frame0_qpos = np.asarray(
            qpos_history[0]
        )

        frame0_qvel = np.asarray(
            qvel_history[0]
        )


        video_data.qpos[:] = frame0_qpos
        video_data.qvel[:] = frame0_qvel


        mujoco.mj_normalizeQuat(
            model,
            video_data.qpos,
        )

        mujoco.mj_forward(
            model,
            video_data,
        )

        renderer.update_scene(
            video_data,
            camera=camera,
        )


        first_frame = np.asarray(
            renderer.render(),
            dtype=np.uint8,
        )


        frame_height, frame_width = (
            first_frame.shape[:2]
        )


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


        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


        process.stdin.write(
            first_frame.tobytes()
        )


        for qpos, qvel in zip(
            qpos_history[1:],
            qvel_history[1:],
        ):

            video_data.qpos[:] = np.asarray(
                qpos
            )

            video_data.qvel[:] = np.asarray(
                qvel
            )


            mujoco.mj_normalizeQuat(
                model,
                video_data.qpos,
            )

            mujoco.mj_forward(
                model,
                video_data,
            )


            renderer.update_scene(
                video_data,
                camera=camera,
            )

            frame = np.asarray(
                renderer.render(),
                dtype=np.uint8,
            )

            process.stdin.write(
                frame.tobytes()
            )


        process.stdin.close()

        error_output = (
            process.stderr
            .read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        return_code = process.wait()


        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed with exit code "
                f"{return_code}: {error_output}"
            )


    except Exception:

        if (
            process is not None
            and process.poll() is None
        ):
            process.kill()
            process.wait()

        raise


    finally:
        renderer.close()


    print(
        f"Saved closed-loop RTI rollout to: "
        f"{filename}"
    )

    print(
        f"Video frames: "
        f"{len(qpos_history)} "
        f"at {video_fps} fps"
    )


# =============================================================================
# Closed-loop SQP-RTI moving-goal minimum-time MPC
# =============================================================================

def main(
    headless=False,
    steps=250,
    save_video=True,
):

    if _auto_headless and not headless:
        print(
            "Interactive MuJoCo rendering is unavailable in this display "
            "session; using headless simulation instead."
        )

        headless = True


    # ------------------------------------------------------------------
    # Configure augmented moving-goal minimum-time problem
    # ------------------------------------------------------------------
    configure_min_time_problem()


    # ------------------------------------------------------------------
    # MuJoCo model
    # ------------------------------------------------------------------
    mpx_root = Path(
        mpx.__file__
    ).parent


    model = mujoco.MjModel.from_xml_path(
        str(
            mpx_root
            / "data"
            / "go2"
            / "scene_mjx.xml"
        )
    )


    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720


    data = mujoco.MjData(
        model
    )


    sim_frequency = 200.0

    model.opt.timestep = (
        1.0 / sim_frequency
    )

    sim_dt = model.opt.timestep


    contact_ids = sim_utils.geom_ids(
        model,
        config.contact_frame,
    )


    # ==================================================================
    # RTI solver configuration
    # ==================================================================

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


    sqp_cfg = force_integer_num_phases(
        sqp_cfg
    )


    # ------------------------------------------------------------------
    # Disturbance
    # ------------------------------------------------------------------
    E_mag = 0.05

    disturbance = make_min_time_disturbance(
        n=config.n,
        N=config.N,
        E_mag=E_mag,
    )


    obstacles = jnp.zeros(
        (0, 3),
        dtype=config.initial_state.dtype,
    )


    # ------------------------------------------------------------------
    # MPC wrapper
    # ------------------------------------------------------------------
    mpc = mpc_wrapper.MPCWrapper(
        config,

        sls_config=sls_cfg,

        sqp_config=sqp_cfg,

        admm_config=admm_cfg,

        constraints=min_time_constraints,

        obstacles=obstacles,

        disturbance=disturbance,
    )


    solve_rti = build_rti_solve_fn(
        mpc
    )

    advance_clock = build_clock_update_fn(
        mpc
    )


    # ==================================================================
    # Initial measured state
    # ==================================================================

    data.qpos = np.asarray(
        jnp.concatenate([
            config.p0,
            config.quat0,
            config.q0,
        ])
    )


    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.time = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    initial_position = np.asarray(
        data.qpos[:3]
    ).copy()


    foot = jnp.asarray(
        sim_utils.geom_positions(
            data,
            contact_ids,
        )
    )


    contact = jnp.asarray(
        sim_utils.estimate_contacts(
            data,
            contact_ids,
        )
    )


    # ------------------------------------------------------------------
    # MPC data
    # ------------------------------------------------------------------
    mpc_data = mpc.reset(
        mpc.make_data(),

        jnp.asarray(
            data.qpos.copy()
        ),

        jnp.asarray(
            data.qvel.copy()
        ),

        foot,
    )


    # ------------------------------------------------------------------
    # Current augmented state
    #
    # Initial goal:
    #
    #       x_goal = current_x + 0.5
    # ------------------------------------------------------------------
    x0 = measured_augmented_state(
        mpc,
        data,
        foot,

        jnp.asarray(
            INITIAL_DURATION,
            dtype=mpc.initial_state.dtype,
        ),
    )


    print(
        f"Initial base x: "
        f"{float(np.asarray(x0[0])):.4f} m"
    )

    print(
        f"Initial moving goal x: "
        f"{float(np.asarray(x0[GOAL_X_IDX])):.4f} m"
    )

    print(
        f"Lookahead distance: "
        f"{LOOKAHEAD_DISTANCE:.4f} m"
    )


    # ==================================================================
    # Structured initial nominal trajectory
    # ==================================================================

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


    X_guess = build_initial_guess(
        x0,
        reference0,
    )


    U_guess = jnp.tile(
        config.u_ref,
        (config.N, 1),
    )


    mpc_data = mpc_data.replace(
        X0=X_guess,

        U0=U_guess,

        dt=float(
            INITIAL_DURATION
            / config.N
        ),

        liftoff=liftoff0,
    )


    # ==================================================================
    # Compile once
    # ==================================================================

    t_compile = timer()


    (
        mpc_data,
        tau,
        X_pred,
        U_pred,
        dt_apply,
        valid,
    ) = solve_rti(
        mpc_data,
        x0,
        COMMAND,
        contact,
    )


    tau.block_until_ready()


    print(
        f"Initial compile/solve: "
        f"{1e3 * (timer() - t_compile):.2f} ms"
    )


    # ==================================================================
    # Closed-loop execution
    # ==================================================================

    control_times_ms = []

    duration_history = []

    goal_history = [
        float(
            np.asarray(
                x0[GOAL_X_IDX]
            )
        )
    ]

    position_history = [
        np.asarray(
            data.qpos[:3]
        ).copy()
    ]


    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------
    video_fps = 60

    frame_dt = (
        1.0 / video_fps
    )

    next_frame_time = 0.0


    qpos_history = [
        np.asarray(
            data.qpos
        ).copy()
    ]

    qvel_history = [
        np.asarray(
            data.qvel
        ).copy()
    ]


    next_frame_time += frame_dt


    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    viewer_ctx = None
    viewer = None


    if not headless:

        viewer_ctx = mujoco.viewer.launch_passive(
            model,
            data,
        )

        viewer = viewer_ctx.__enter__()

        viewer.sync()


    # ==================================================================
    # MPC loop
    # ==================================================================

    try:

        for iteration in range(
            int(steps)
        ):

            # ----------------------------------------------------------
            # First control was already computed during compile call.
            #
            # Every later iteration:
            #
            #   1. measure robot
            #   2. set goal_x = measured_x + 0.5
            #   3. execute one SQP RTI iteration
            # ----------------------------------------------------------
            if iteration > 0:

                foot = jnp.asarray(
                    sim_utils.geom_positions(
                        data,
                        contact_ids,
                    )
                )


                contact = jnp.asarray(
                    sim_utils.estimate_contacts(
                        data,
                        contact_ids,
                    )
                )


                duration_guess = jnp.clip(
                    mpc_data.X0[
                        0,
                        T_IDX
                    ],
                    MIN_TIME,
                    MAX_TIME,
                )


                # =====================================================
                # This moves the goal forward with the quadruped.
                # =====================================================
                x0 = measured_augmented_state(
                    mpc,
                    data,
                    foot,
                    duration_guess,
                )


                tic = timer()


                (
                    mpc_data,
                    tau,
                    X_pred,
                    U_pred,
                    dt_apply,
                    valid,
                ) = solve_rti(
                    mpc_data,
                    x0,
                    COMMAND,
                    contact,
                )


                tau.block_until_ready()


                toc = timer()


                control_times_ms.append(
                    1e3 * (toc - tic)
                )


            # ----------------------------------------------------------
            # Solver quantities
            # ----------------------------------------------------------
            dt_apply_float = float(
                np.asarray(
                    dt_apply
                )
            )


            duration_now = float(
                np.asarray(
                    X_pred[
                        0,
                        T_IDX
                    ]
                )
            )


            goal_x_now = float(
                np.asarray(
                    X_pred[
                        0,
                        GOAL_X_IDX
                    ]
                )
            )


            duration_history.append(
                duration_now
            )

            goal_history.append(
                goal_x_now
            )


            # ----------------------------------------------------------
            # Apply first control
            # ----------------------------------------------------------
            tau_np = np.asarray(
                jnp.clip(
                    tau,
                    config.min_torque,
                    config.max_torque,
                )
            )


            data.ctrl[:] = tau_np


            start_sim_time = float(
                data.time
            )


            target_sim_time = (
                start_sim_time
                + dt_apply_float
            )


            # ----------------------------------------------------------
            # Hold U[0] for one optimized shooting interval
            # ----------------------------------------------------------
            while (
                data.time
                < target_sim_time
                - 0.5 * sim_dt
            ):

                mujoco.mj_step(
                    model,
                    data,
                )


                while (
                    data.time
                    >= next_frame_time
                ):

                    qpos_history.append(
                        np.asarray(
                            data.qpos
                        ).copy()
                    )

                    qvel_history.append(
                        np.asarray(
                            data.qvel
                        ).copy()
                    )

                    next_frame_time += frame_dt


                if (
                    viewer is not None
                    and viewer.is_running()
                ):
                    viewer.sync()


            # ----------------------------------------------------------
            # Advance gait clock by actual MuJoCo elapsed time
            # ----------------------------------------------------------
            elapsed = (
                float(data.time)
                - start_sim_time
            )


            mpc_data = advance_clock(
                mpc_data,

                jnp.asarray(
                    elapsed,
                    dtype=mpc_data.time.dtype,
                ),
            )


            current_position = np.asarray(
                data.qpos[:3]
            ).copy()


            position_history.append(
                current_position
            )


            # ----------------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------------
            current_x = float(
                current_position[0]
            )


            print(
                f"RTI {iteration:03d} | "
                f"valid={bool(np.asarray(valid))} | "
                f"T_rem={duration_now:.4f} s | "
                f"dt={dt_apply_float:.4f} s | "
                f"x={current_x:.4f} | "
                f"goal_x={goal_x_now:.4f} | "
                f"lookahead={goal_x_now - current_x:.4f} | "
                f"p={current_position}"
            )


            # ----------------------------------------------------------
            # There is deliberately NO terminal-set break.
            #
            # The target moves another 0.5 m forward every MPC update,
            # so the experiment runs until --steps is exhausted.
            # ----------------------------------------------------------

            if (
                viewer is not None
                and not viewer.is_running()
            ):
                break


    finally:

        if viewer_ctx is not None:
            viewer_ctx.__exit__(
                None,
                None,
                None,
            )


    # ==================================================================
    # Summary
    # ==================================================================

    position_history = np.asarray(
        position_history
    )


    final_position = np.asarray(
        data.qpos[:3]
    )


    forward_distance = (
        final_position[0]
        - initial_position[0]
    )


    simulated_time = float(
        data.time
    )


    if simulated_time > 0.0:
        average_forward_speed = (
            forward_distance
            / simulated_time
        )
    else:
        average_forward_speed = 0.0


    print(
        "\nClosed-loop moving-goal RTI summary"
    )

    print(
        f"  simulated time:        "
        f"{simulated_time:.6f} s"
    )

    print(
        f"  initial position:      "
        f"{initial_position}"
    )

    print(
        f"  final position:        "
        f"{final_position}"
    )

    print(
        f"  forward displacement:  "
        f"{forward_distance:.6f} m"
    )

    print(
        f"  average forward speed: "
        f"{average_forward_speed:.6f} m/s"
    )

    print(
        f"  lookahead distance:    "
        f"{LOOKAHEAD_DISTANCE:.6f} m"
    )


    if control_times_ms:

        timings = np.asarray(
            control_times_ms
        )

        print(
            f"  RTI mean:              "
            f"{timings.mean():.3f} ms"
        )

        print(
            f"  RTI median:            "
            f"{np.median(timings):.3f} ms"
        )

        print(
            f"  RTI max:               "
            f"{timings.max():.3f} ms"
        )


    if duration_history:

        print(
            f"  initial T:             "
            f"{duration_history[0]:.6f} s"
        )

        print(
            f"  final T_rem:           "
            f"{duration_history[-1]:.6f} s"
        )


    # ==================================================================
    # Video
    # ==================================================================

    if save_video:

        encode_state_history_video(
            model,

            qpos_history,
            qvel_history,

            filename=(
                "quadruped_rti_rollout.mp4"
            ),

            video_fps=video_fps,

            width=1280,
            height=720,
        )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help=(
            "Number of SQP-RTI MPC updates. "
            "The target continuously remains 0.5 m ahead."
        ),
    )


    parser.add_argument(
        "--headless",
        action="store_true",
    )


    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Disable writing "
            "quadruped_rti_rollout.mp4."
        ),
    )


    args = parser.parse_args()


    main(
        headless=args.headless,
        steps=args.steps,
        save_video=not args.no_video,
    )