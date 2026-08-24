"""Minimum-time Go2 barrel roll with nonlinear rollout-versus-SLS-tube validation.

The contact sequence and number of shooting intervals in each phase stay fixed.
The physical durations T_i are appended to the whole-body state and optimized
in the same way as the phase times in plan_drone_racing.

The reference starts and ends on opposite sides of a hurdle.  Smooth hard
constraints keep conservative trunk and foot envelopes outside the hurdle,
so the obstacle affects the optimized motion rather than only the rendering.

Use --dry-run to validate callback shapes without compiling the full solver.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import os
# os.environ["JAX_PLATFORMS"] = "cpu"

# Configure paths and headless MuJoCo before importing JAX/MuJoCo.
DIR_PATH = Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH.parent))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_disable_jit", True)

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mujoco import mjx
from mujoco.mjx._src import math

import config_go2 as config
import mpc_utils as barrel_mpc_utils
import mpx.utils.objectives as mpc_objectives
import mpx.utils.sim as sim_utils
import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig



# Stages encoded by reference_barrel_roll_min_time.
PHASE_NAMES = (
    "stance",
    "lateral_launch",
    "flight",
    "prelanding",
    "touchdown",
    "settle",
)
NOMINAL_DURATIONS = jnp.array([0.20, 0.20, 0.24, 0.06, 0.10, 0.20])
PHASE_END_STEPS = jnp.array([10, 20, 32, 35, 40, 50], dtype=jnp.int32)
SEGMENT_LENGTHS = jnp.concatenate(
    [PHASE_END_STEPS[:1], PHASE_END_STEPS[1:] - PHASE_END_STEPS[:-1]]
)
NUM_PHASES = len(PHASE_NAMES)

if config.N != int(PHASE_END_STEPS[-1]):
    raise ValueError(
        f"The phase layout requires config.N={int(PHASE_END_STEPS[-1])}; "
        f"got {config.N}."
    )

MIN_DURATIONS = 0.30 * NOMINAL_DURATIONS
MAX_DURATIONS = 2.00 * NOMINAL_DURATIONS
# Keep the time objective on the same scale as the trajectory regularization.
# A much larger weight makes the first SQP model drive every duration toward
# its lower bound before the nonlinear dynamics and waypoint constraints have
# become feasible, which stalls convergence at a heavily infeasible rollout.
TIME_WEIGHT = 1e4
# The constraint vector mixes meters/radians with force limits, while the
# dynamics vector also contains algebraic GRF states measured in newtons.
# These tolerances correspond to 5 cm/rad for generic constraints and 0.25 N
# for the worst algebraic shooting defect.  Obstacle clearance uses its own
# much tighter dimensionless tolerance below.
FEASIBILITY_TOLERANCE = 5.0e-2
DYNAMICS_TOLERANCE = 2.5e-1
OBSTACLE_TOLERANCE = 5.0e-3

# Original MPX state: qpos, qvel, foot positions, and GRFs.
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
DURATION_SLICE = slice(PHYSICAL_N, PHYSICAL_N + NUM_PHASES)
GRF_START = 13 + 2 * config.n_joints + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact

# Phase-end pose waypoints keep the optimizer from collapsing the motion.
WAYPOINT_POSITION_HALF_WIDTHS = jnp.array(
    [
        [0.06, 0.06, 0.05],
        [0.08, 0.08, 0.07],
        [0.10, 0.10, 0.09],
        [0.10, 0.10, 0.08],
        [0.07, 0.07, 0.05],
        [0.05, 0.05, 0.03],
    ]
)
WAYPOINT_ANGLE_TOLERANCES = jnp.deg2rad(
    jnp.array([15.0, 25.0, 25.0, 18.0, 12.0, 8.0])
)
WAYPOINT_ANGULAR_SPEED_LIMITS = jnp.array(
    [5.0, 20.0, 8.0, 3.0, 1.5, 1.0]
)
WAYPOINT_LINEAR_SPEED_LIMITS = jnp.array(
    [1.5, 2.5, 2.5, 1.5, 0.6, 0.25]
)
FINAL_LINEAR_SPEED = 0.25
FINAL_ANGULAR_SPEED = 1.0
FINAL_JOINT_SPEED = 2.0
FINAL_JOINT_POSITION_TOLERANCE = 0.20
MIN_BASE_HEIGHT = 0.12
MAX_JOINT_SPEED = 25.0

# Unitree Go2 joint limits, repeated FL/FR/RL/RR.
JOINT_MIN = jnp.tile(jnp.array([-1.0472, -0.6632, -2.7210]), config.n_contact)
JOINT_MAX = jnp.tile(jnp.array([1.0472, 2.9660, -0.8370]), config.n_contact)

FRICTION_COEFFICIENT = 0.5
MAX_NORMAL_FORCE = 250.0
TANGENTIAL_FORCE_SMOOTHING = 1.0e-4
CONTACT_SOLVE_REGULARIZATION = 1.0e-6

# The roll moves in -y.  A thin hurdle spans x and sits halfway between the
# initial and final base positions.  Sizes are full MuJoCo box dimensions.
LATERAL_DISPLACEMENT = -0.60
OBSTACLE_SIZE = jnp.array([0.80, 0.05, 0.16])
OBSTACLE_CENTER = jnp.array(
    [0.0, 0.5 * LATERAL_DISPLACEMENT, 0.5 * OBSTACLE_SIZE[2]]
)
BASE_CLEARANCE_RADIUS = 0.17
FOOT_CLEARANCE_RADIUS = 0.02
JOINT_CLEARANCE_RADIUS = 0.035
SUPERELLIPSOID_POWER = 4
SUPERELLIPSOID_SCALE = 2.0 ** (1.0 / SUPERELLIPSOID_POWER)
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)


def phase_index(t):
    index = jnp.searchsorted(PHASE_END_STEPS, t, side="right")
    return jnp.clip(index, 0, NUM_PHASES - 1)


def min_time_barrel_roll_dynamics(model, mjx_model, contact_id, body_id):
    """Build dynamics with dt_k = T_phase(k) / N_phase(k)."""

    n_joints = config.n_joints

    def dynamics(x, u, t, parameter):
        phase = phase_index(t)
        dt = x[PHYSICAL_N + phase] / SEGMENT_LENGTHS[phase]

        qpos = x[: n_joints + 7]
        qvel = x[n_joints + 7 : 2 * n_joints + 13]
        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        data = mjx.fwd_position(mjx_model, data)
        data = mjx.fwd_velocity(mjx_model, data)

        mass_factor = data.qLD
        bias_force = data.qfrc_bias
        contact = parameter[t, : config.n_contact]
        tau = jnp.concatenate([jnp.zeros(6, dtype=x.dtype), u])

        feet = [data.geom_xpos[geom_id] for geom_id in contact_id[:4]]
        jacobians = [
            mjx.jac(mjx_model, data, foot, body)[0]
            for foot, body in zip(feet, body_id[:4])
        ]
        contact_jacobian = jnp.concatenate(jacobians, axis=1)
        current_feet = jnp.concatenate(feet)

        # Solve only the active contact constraints. Masking forces after a
        # four-foot solve gives incorrect forces for flight and staged landing.
        contact_mask = jnp.repeat(contact, 3)
        active_jacobian = contact_jacobian * contact_mask[None, :]
        velocity_violation = active_jacobian.T @ qvel
        baumgarte = -50.0 * velocity_violation
        mass_inv_jacobian = jax.scipy.linalg.cho_solve(
            (mass_factor, False), active_jacobian
        )
        projected_mass = active_jacobian.T @ mass_inv_jacobian
        # Inactive rows/columns receive identity pivots so the fixed-size
        # contact system remains positive definite.  Damping active pivots
        # keeps trial configurations with a nearly rank-deficient contact
        # Jacobian from producing a failed Cholesky factorization.
        projected_mass = 0.5 * (projected_mass + projected_mass.T)
        projected_mass = (
            projected_mass
            + jnp.diag(1.0 - contact_mask)
            + CONTACT_SOLVE_REGULARIZATION
            * jnp.eye(projected_mass.shape[0], dtype=projected_mass.dtype)
        )
        rhs = (
            -active_jacobian.T
            @ jax.scipy.linalg.cho_solve((mass_factor, False), tau - bias_force)
            + baumgarte
        )
        grf = jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(projected_mass), rhs
        )
        grf = grf * contact_mask

        acceleration = jax.scipy.linalg.cho_solve(
            (mass_factor, False),
            tau - bias_force + active_jacobian @ grf,
        )
        velocity_next = qvel + acceleration * dt
        position_next = qpos[:3] + velocity_next[:3] * dt
        quaternion_next = math.quat_integrate(
            qpos[3:7], velocity_next[3:6], dt
        )
        joints_next = (
            qpos[7 : 7 + n_joints]
            + velocity_next[6 : 6 + n_joints] * dt
        )
        physical_next = jnp.concatenate(
            [
                position_next,
                quaternion_next,
                joints_next,
                velocity_next,
                current_feet,
                grf,
            ]
        )

        # Every phase duration is a constant state along the horizon.
        return jnp.concatenate([physical_next, x[DURATION_SLICE]])

    return dynamics


def min_time_barrel_roll_cost(W, reference, x, u, t):
    """Whole-body tracking regularization plus sum of phase durations."""

    nj = config.n_joints
    nc = config.n_contact
    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + nj]
    dp = x[7 + nj : 10 + nj]
    omega = x[10 + nj : 13 + nj]
    dq = x[13 + nj : 13 + 2 * nj]
    feet = x[13 + 2 * nj : 13 + 2 * nj + 3 * nc]
    grf = x[GRF_START:GRF_STOP]

    p_ref = reference["p"][t]
    quat_ref = reference["quat"][t]
    q_ref = reference["q"][t]
    dp_ref = reference["dp"][t]
    omega_ref = reference["omega"][t]
    feet_ref = reference["foot"][t]
    contact = reference["contact"][t]
    grf_ref = reference["grf"][t]

    quaternion_error = math.quat_sub(quat, quat_ref)
    contact_map = jnp.repeat(contact, 3)
    friction_margin = (
        0.5 * grf[2::3]
        - jnp.sqrt(grf[0::3] ** 2 + grf[1::3] ** 2 + 1.0e-2)
    )
    friction_penalty = jnp.sum(
        mpc_objectives.penalty(friction_margin) * contact
    )
    stage_cost = (
        (p - p_ref).T @ W["pos"] @ (p - p_ref)
        + quaternion_error.T @ W["rot"] @ quaternion_error
        + (q - q_ref).T @ W["q"] @ (q - q_ref)
        + (dp - dp_ref).T @ W["vel"] @ (dp - dp_ref)
        + (omega - omega_ref).T @ W["omega"] @ (omega - omega_ref)
        + dq.T @ W["dq"] @ dq
        + (contact_map * (feet - feet_ref)).T
        @ W["contact"]
        @ (contact_map * (feet - feet_ref))
        + u.T @ W["tau"] @ u
        + (grf - grf_ref).T @ W["grf"] @ (grf - grf_ref)
        + friction_penalty
    )
    # Durations are constant states, so charging them at every node would
    # multiply the intended minimum-time objective by N + 1 and overwhelm the
    # feasibility terms.  Charge the physical phase-time sum exactly once.
    time_cost = jnp.where(
        t == config.N,
        jnp.dot(W["time"], x[DURATION_SLICE]),
        jnp.asarray(0.0, dtype=x.dtype),
    )
    return 0.5 * stage_cost + time_cost


def build_barrel_roll_reference():
    """Pad the existing N-sample barrel reference for the N+1-state SQP."""

    reference, parameter = barrel_mpc_utils.reference_barrel_roll_min_time(
        config.N,
        config.dt,
        config.n_joints,
        config.n_contact,
        config.p_legs0,
        config.q0,
        LATERAL_DISPLACEMENT,
    )
    reference = {
        name: jnp.concatenate([value, value[-1:]], axis=0)
        for name, value in reference.items()
    }
    parameter = jnp.concatenate([parameter, parameter[-1:]], axis=0)
    return reference, parameter


def build_initial_guess(x0, reference):
    nj = config.n_joints
    nc = config.n_contact
    X = jnp.tile(x0, (config.N + 1, 1))
    X = X.at[:, :3].set(reference["p"])
    X = X.at[:, 3:7].set(reference["quat"])
    X = X.at[:, 7 : 7 + nj].set(reference["q"])
    X = X.at[:, 7 + nj : 10 + nj].set(reference["dp"])
    X = X.at[:, 10 + nj : 13 + nj].set(reference["omega"])
    phase_ids = phase_index(jnp.arange(config.N, dtype=jnp.int32))
    node_dts = NOMINAL_DURATIONS[phase_ids] / SEGMENT_LENGTHS[phase_ids]
    joint_rates = (reference["q"][1:] - reference["q"][:-1]) / node_dts[:, None]
    joint_rates = jnp.concatenate(
        [joint_rates, jnp.zeros((1, nj), dtype=X.dtype)], axis=0
    )
    X = X.at[:, 13 + nj : 13 + 2 * nj].set(joint_rates)
    X = X.at[:, 13 + 2 * nj : 13 + 2 * nj + 3 * nc].set(
        reference["foot"]
    )
    X = X.at[:, GRF_START:GRF_STOP].set(reference["grf"])
    X = X.at[:, DURATION_SLICE].set(NOMINAL_DURATIONS)
    # Physical x0 is fixed; only phase durations at node zero are free.
    return X.at[0, :PHYSICAL_N].set(x0[:PHYSICAL_N])


def make_obstacle_clearance_constraints():
    """Build differentiable trunk, joint, and foot hurdle constraints.

    The hurdle's y-z rectangle is inflated by a clearance around the base and
    each robot point, then enclosed by a smooth fourth-order superellipse.
    Treating the hurdle as infinite along x is conservative for this lateral
    maneuver.
    """

    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)
    joint_ids = []
    for name in LEG_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"Could not find required leg joint: {name}")
        joint_ids.append(joint_id)
    joint_ids = jnp.asarray(joint_ids, dtype=jnp.int32)

    nj = config.n_joints
    nc = config.n_contact
    feet_start = 13 + 2 * nj

    def inflated_superellipse(points, clearance):
        radii = (
            (0.5 * OBSTACLE_SIZE[1:] + clearance)
            * SUPERELLIPSOID_SCALE
        )
        normalized = (points[:, 1:] - OBSTACLE_CENTER[1:]) / radii
        radial_power = jnp.sum(
            normalized**SUPERELLIPSOID_POWER, axis=-1
        )
        # This bounded form has the same zero boundary and feasible set as
        # 1 - radial_power, but avoids enormous residuals and Jacobians far
        # from the obstacle during SQP trial steps.
        return 2.0 / (1.0 + radial_power) - 1.0

    def constraints(x):
        qpos = x[: nj + 7]
        kinematics = mjx.make_data(model).replace(qpos=qpos)
        kinematics = mjx.fwd_position(mjx_model, kinematics)
        joint_positions = kinematics.xanchor[joint_ids]
        feet = x[feet_start : feet_start + 3 * nc].reshape(nc, 3)

        base_constraint = inflated_superellipse(
            x[:3][None, :], BASE_CLEARANCE_RADIUS
        )
        joint_constraints = inflated_superellipse(
            joint_positions, JOINT_CLEARANCE_RADIUS
        )
        foot_constraints = inflated_superellipse(feet, FOOT_CLEARANCE_RADIUS)
        return jnp.concatenate(
            [base_constraint, joint_constraints, foot_constraints]
        )

    return constraints


def make_barrel_roll_constraints(reference, obstacle_constraints):
    """Build hard timing, pose, obstacle, force, and landing limits."""

    nj = config.n_joints
    waypoint_positions = reference["p"][PHASE_END_STEPS]
    waypoint_quaternions = reference["quat"][PHASE_END_STEPS]
    terminal_joints = reference["q"][-1]
    cosine_limits = jnp.cos(0.5 * WAYPOINT_ANGLE_TOLERANCES)

    def constraints(x, u, t):
        durations = x[DURATION_SLICE]
        q = x[7 : 7 + nj]
        dp = x[7 + nj : 10 + nj]
        omega = x[10 + nj : 13 + nj]
        dq = x[13 + nj : 13 + 2 * nj]
        grf = x[GRF_START:GRF_STOP].reshape(config.n_contact, 3)

        duration_constraints = jnp.concatenate(
            [MIN_DURATIONS - durations, durations - MAX_DURATIONS]
        )
        torque_constraints = jnp.concatenate(
            [u - config.max_torque, config.min_torque - u]
        )
        joint_constraints = jnp.concatenate(
            [q - JOINT_MAX, JOINT_MIN - q, jnp.abs(dq) - MAX_JOINT_SPEED]
        )

        active = t == PHASE_END_STEPS
        position_upper = (
            x[:3][None, :]
            - waypoint_positions
            - WAYPOINT_POSITION_HALF_WIDTHS
        )
        position_lower = (
            waypoint_positions
            - x[:3][None, :]
            - WAYPOINT_POSITION_HALF_WIDTHS
        )
        position_constraints = jnp.concatenate(
            [position_upper, position_lower], axis=1
        )
        position_constraints = jnp.where(
            active[:, None],
            position_constraints,
            -jnp.ones_like(position_constraints),
        )

        normalized_quat = x[3:7] / (jnp.linalg.norm(x[3:7]) + 1.0e-8)
        alignment = jnp.abs(waypoint_quaternions @ normalized_quat)
        orientation_constraints = jnp.where(
            active,
            cosine_limits - alignment,
            -jnp.ones_like(alignment),
        )
        waypoint_rate_constraints = jnp.concatenate(
            [
                jnp.abs(dp)[None, :] - WAYPOINT_LINEAR_SPEED_LIMITS[:, None],
                jnp.abs(omega)[None, :]
                - WAYPOINT_ANGULAR_SPEED_LIMITS[:, None],
            ],
            axis=1,
        )
        waypoint_rate_constraints = jnp.where(
            active[:, None],
            waypoint_rate_constraints,
            -jnp.ones_like(waypoint_rate_constraints),
        )

        # Hard unilateral contact and friction cone. Inactive feet are omitted;
        # the dynamics already masks their GRFs to zero.
        contact = reference["contact"][t]
        # The initial GRF guess is zero.  A raw Euclidean norm has an
        # undefined derivative there, which makes the constraint Jacobian NaN
        # before the first SQP step.  Subtracting the smoothing radius keeps
        # both the value and gradient zero at zero force.
        tangent_force = (
            jnp.sqrt(
                jnp.sum(grf[:, :2] ** 2, axis=1)
                + TANGENTIAL_FORCE_SMOOTHING**2
            )
            - TANGENTIAL_FORCE_SMOOTHING
        )
        normal_force = grf[:, 2]
        contact_force_constraints = jnp.stack(
            [
                -normal_force,
                normal_force - MAX_NORMAL_FORCE,
                tangent_force - FRICTION_COEFFICIENT * normal_force,
            ],
            axis=1,
        )
        contact_force_constraints = jnp.where(
            contact[:, None] > 0.5,
            contact_force_constraints,
            -jnp.ones_like(contact_force_constraints),
        )

        final_constraints = jnp.concatenate(
            [
                jnp.abs(dp) - FINAL_LINEAR_SPEED,
                jnp.abs(omega) - FINAL_ANGULAR_SPEED,
                jnp.abs(q - terminal_joints) - FINAL_JOINT_POSITION_TOLERANCE,
                jnp.abs(dq) - FINAL_JOINT_SPEED,
            ]
        )
        final_constraints = jnp.where(
            t == config.N, final_constraints, -jnp.ones_like(final_constraints)
        )

        return jnp.concatenate(
            [
                duration_constraints,
                torque_constraints,
                joint_constraints,
                position_constraints.reshape(-1),
                orientation_constraints,
                waypoint_rate_constraints.reshape(-1),
                contact_force_constraints.reshape(-1),
                final_constraints,
                jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
                obstacle_constraints(x),
            ]
        )

    return constraints


def make_phase_scaled_disturbance(n, magnitude=0.02):
    """Disturb base position, scaled by the optimized phase timestep."""

    def disturbance_at_state(x, t):
        phase = phase_index(t)
        dt = x[PHYSICAL_N + phase] / SEGMENT_LENGTHS[phase]
        diagonal = jnp.zeros(n, dtype=x.dtype).at[:3].set(magnitude * dt)
        diagonal = diagonal.at[DURATION_SLICE].set(0.0)
        return jnp.diag(diagonal)

    def disturbance(X):
        return jax.vmap(disturbance_at_state)(X, jnp.arange(X.shape[0]))

    disturbance.at_state = disturbance_at_state
    return disturbance


def unused_reference_generator(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("This one-shot experiment supplies its reference directly.")


def configure_problem():
    """Configure the Go2 module for six free duration states."""

    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.n = PHYSICAL_N + NUM_PHASES
    config.nu = config.n_joints
    config.initial_state = jnp.concatenate([base_state, NOMINAL_DURATIONS])
    config.dynamics = min_time_barrel_roll_dynamics
    config.cost = min_time_barrel_roll_cost
    config.hessian_approx = None
    config.reference_generator = unused_reference_generator

    dtype = config.initial_state.dtype
    config.W = {
        "pos": jnp.diag(jnp.array([20.0, 20.0, 60.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([200.0, 100.0, 100.0], dtype=dtype)),
        "q": jnp.eye(config.n_joints, dtype=dtype) * 2.0,
        "vel": jnp.diag(jnp.array([0.5, 0.5, 1.0], dtype=dtype)),
        "omega": jnp.diag(jnp.array([1.0, 2.0, 2.0], dtype=dtype)),
        "dq": jnp.eye(config.n_joints, dtype=dtype) * 1.0e-1,
        "contact": jnp.eye(3 * config.n_contact, dtype=dtype) * 50.0,
        "tau": jnp.eye(config.n_joints, dtype=dtype) * 1.0e-2,
        "grf": jnp.eye(3 * config.n_contact, dtype=dtype) * 1.0e-3,
        "time": jnp.ones(NUM_PHASES, dtype=dtype) * TIME_WEIGHT,
    }


def solve_barrel_roll(mpc, data, x0, reference, parameter):
    """Solve the minimum-time problem directly from the nominal reference."""

    X_guess = build_initial_guess(x0, reference)
    U_guess = jnp.tile(config.u_ref, (config.N, 1))

    # Match auxiliary foot states to the dynamics at the reference poses and
    # use the closest simple friction-feasible version of its predicted GRFs.
    # Raw predicted forces can be tensile while the reference is still
    # dynamically inconsistent; seeding those values directly trades an
    # equality defect for a much worse inequality violation.
    predicted_next = jax.vmap(
        lambda x, u, t: mpc.dynamics(x, u, t, parameter)
    )(
        X_guess[:-1],
        U_guess,
        jnp.arange(config.N, dtype=jnp.int32),
    )
    foot_slice = slice(13 + 2 * config.n_joints, GRF_START)
    X_guess = X_guess.at[1:, foot_slice].set(predicted_next[:, foot_slice])

    predicted_grf = predicted_next[:, GRF_START:GRF_STOP].reshape(
        config.N, config.n_contact, 3
    )
    normal_force = jnp.clip(predicted_grf[..., 2], 0.0, MAX_NORMAL_FORCE)
    tangential_force = predicted_grf[..., :2]
    tangential_norm = jnp.linalg.norm(tangential_force, axis=-1)
    tangential_scale = jnp.minimum(
        1.0,
        FRICTION_COEFFICIENT
        * normal_force
        / jnp.maximum(tangential_norm, TANGENTIAL_FORCE_SMOOTHING),
    )
    projected_grf = jnp.concatenate(
        [tangential_force * tangential_scale[..., None], normal_force[..., None]],
        axis=-1,
    )
    X_guess = X_guess.at[1:, GRF_START:GRF_STOP].set(
        projected_grf.reshape(config.N, -1)
    )

    def run_solver(W, X, U, workspace):
        (
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
        ) = workspace
        return mpc._solve(
            reference,
            parameter,
            W,
            x0,
            X,
            U,
            V,
            w,
            y,
            rho,
            rho_grad,
            mpc.obstacles,
            backoffs,
            beta,
            mu,
            Phi_x,
            Phi_u,
            Phi_x_I,
            Phi_u_I,
            a,
            b,
            converged_admm,
        )

    initial_workspace = (
        data.V0,
        data.w,
        data.y,
        data.rho,
        data.rho_grad,
        data.h_ct_ws,
        data.Phi_x_ws,
        data.Phi_u_ws,
        data.beta_ws,
        data.mu_ws,
        data.Phi_x_I_ws,
        data.Phi_u_I_ws,
        data.a,
        data.b,
        data.converged_admm,
    )

    return run_solver(data.W, X_guess, U_guess, initial_workspace)


def phase_node_times(phase_times):
    dts = np.asarray(phase_times) / np.asarray(SEGMENT_LENGTHS)
    transition_phases = np.searchsorted(
        np.asarray(PHASE_END_STEPS), np.arange(config.N), side="right"
    )
    return np.concatenate([[0.0], np.cumsum(dts[transition_phases])])



def get_trajectory_tubes(Phi_x):
    """Compute the per-state SLS tube radius at each trajectory node."""
    return jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)


def rollout_optimized_dynamics(dynamics, X_nominal, U, parameter):
    """Sequentially roll out the exact experiment dynamics.

    This deliberately does NOT use mjx.step.  Starting from the optimized
    initial state, it repeatedly evaluates the same dynamics callback used by
    the optimizer:

        x_roll[k+1] = dynamics(x_roll[k], U[k], k, parameter)

    Using X_nominal[0] is important because the phase-duration states at node
    zero are optimization variables.
    """
    x = jnp.asarray(X_nominal[0])
    states = [x]

    for k in range(U.shape[0]):
        x = dynamics(
            x,
            U[k],
            jnp.asarray(k, dtype=jnp.int32),
            parameter,
        )
        states.append(x)

    return jnp.stack(states, axis=0)


def build_state_labels(n_state):
    """Human-readable labels for the current Go2 augmented state."""
    joint_names = (
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    )

    labels = [
        "base_x [m]",
        "base_y [m]",
        "base_z [m]",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]

    labels += [f"{name}_q [rad]" for name in joint_names]

    labels += [
        "base_vx [m/s]",
        "base_vy [m/s]",
        "base_vz [m/s]",
        "base_wx [rad/s]",
        "base_wy [rad/s]",
        "base_wz [rad/s]",
    ]

    labels += [f"{name}_dq [rad/s]" for name in joint_names]

    for leg in ("FL", "FR", "RL", "RR"):
        labels += [
            f"{leg}_foot_x [m]",
            f"{leg}_foot_y [m]",
            f"{leg}_foot_z [m]",
        ]

    for leg in ("FL", "FR", "RL", "RR"):
        labels += [
            f"{leg}_grf_x [N]",
            f"{leg}_grf_y [N]",
            f"{leg}_grf_z [N]",
        ]

    labels += [f"T_{name} [s]" for name in PHASE_NAMES]

    if len(labels) < n_state:
        labels += [f"state_{i}" for i in range(len(labels), n_state)]

    return labels[:n_state]


def sanitize_plot_name(name):
    result = []
    for character in name:
        if character.isalnum() or character in ("-", "_"):
            result.append(character)
        else:
            result.append("_")
    return "".join(result).strip("_")


def align_rollout_deviation_with_tubes(deviation, tubes, node_times):
    """Align SLS tube nodes with rollout-error nodes.

    If Phi_x produces N+1 tube rows, they correspond directly to x_0,...,x_N.

    If Phi_x produces N rows while X has N+1 states, the state-response rows are
    interpreted as the uncertain successor states x_1,...,x_N.  The fixed
    initial state x_0 is therefore omitted from the comparison.
    """
    deviation = np.asarray(deviation)
    tubes = np.asarray(tubes)
    node_times = np.asarray(node_times)

    if tubes.ndim != 2:
        raise ValueError(
            "Expected get_trajectory_tubes(Phi_x) to produce shape "
            f"(time, state); got {tubes.shape}."
        )

    if deviation.ndim != 2:
        raise ValueError(
            f"Expected rollout deviation to be 2-D; got {deviation.shape}."
        )

    if tubes.shape[1] != deviation.shape[1]:
        raise ValueError(
            "State dimension mismatch between rollout deviation and tube: "
            f"{deviation.shape[1]} versus {tubes.shape[1]}."
        )

    if tubes.shape[0] == deviation.shape[0]:
        return deviation, tubes, node_times[: deviation.shape[0]]

    if tubes.shape[0] == deviation.shape[0] - 1:
        return deviation[1:], tubes, node_times[1 : 1 + tubes.shape[0]]

    raise ValueError(
        "Could not align Phi_x tube horizon with trajectory horizon: "
        f"deviation={deviation.shape}, tubes={tubes.shape}."
    )


def plot_rollout_deviation_vs_tubes(
    X_nominal,
    X_rollout,
    Phi_x,
    node_times,
    output_dir,
):
    """Plot signed nonlinear rollout error against +/- SLS tube per state."""
    X_nominal = np.asarray(X_nominal)
    X_rollout = np.asarray(X_rollout)
    node_times = np.asarray(node_times)

    if X_rollout.shape != X_nominal.shape:
        raise ValueError(
            f"Rollout shape {X_rollout.shape} != nominal shape {X_nominal.shape}."
        )

    deviation = X_rollout - X_nominal
    tubes = np.asarray(get_trajectory_tubes(jnp.asarray(Phi_x)))

    deviation_plot, tube_plot, time_plot = align_rollout_deviation_with_tubes(
        deviation,
        tubes,
        node_times,
    )

    labels = build_state_labels(deviation.shape[1])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abs_deviation = np.abs(deviation_plot)
    tube_excess = abs_deviation - tube_plot
    inside = abs_deviation <= tube_plot + 1.0e-12

    # Avoid undefined ratios when both tube and deviation are exactly zero.
    ratio = np.zeros_like(abs_deviation)
    positive_tube = tube_plot > 1.0e-12
    ratio[positive_tube] = (
        abs_deviation[positive_tube] / tube_plot[positive_tube]
    )
    ratio[~positive_tube & (abs_deviation > 1.0e-12)] = np.inf

    print("")
    print("Nonlinear dynamics rollout versus SLS tubes")
    print(f"  X nominal shape:    {X_nominal.shape}")
    print(f"  X rollout shape:    {X_rollout.shape}")
    print(f"  Phi_x shape:        {np.asarray(Phi_x).shape}")
    print(f"  tube radius shape:  {tubes.shape}")
    print(f"  compared shape:     {deviation_plot.shape}")
    print(f"  maximum |error|:    {np.max(abs_deviation):.6e}")
    print(f"  maximum tube:       {np.max(tube_plot):.6e}")
    print(f"  maximum excess:     {np.max(tube_excess):.6e}")
    print(f"  fraction contained: {np.mean(inside):.6f}")

    if np.any(np.isfinite(ratio)):
        finite_ratio = ratio[np.isfinite(ratio)]
        print(f"  maximum finite |error|/tube: {np.max(finite_ratio):.6e}")
    if np.any(np.isinf(ratio)):
        print("  WARNING: nonzero deviation occurred where tube radius is zero.")

    phase_end_steps = np.asarray(PHASE_END_STEPS, dtype=int)

    for state_index, label in enumerate(labels):
        error_i = deviation_plot[:, state_index]
        tube_i = tube_plot[:, state_index]
        outside_i = np.abs(error_i) > tube_i + 1.0e-12

        fig, ax = plt.subplots(figsize=(9.5, 4.8))

        ax.fill_between(
            time_plot,
            -tube_i,
            tube_i,
            alpha=0.22,
            label="+/- SLS tube",
        )
        ax.plot(
            time_plot,
            error_i,
            linewidth=1.8,
            label="dynamics rollout - optimized trajectory",
        )
        ax.plot(
            time_plot,
            tube_i,
            linestyle="--",
            linewidth=0.9,
            label="+ tube",
        )
        ax.plot(
            time_plot,
            -tube_i,
            linestyle="--",
            linewidth=0.9,
            label="- tube",
        )
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)

        if np.any(outside_i):
            ax.scatter(
                time_plot[outside_i],
                error_i[outside_i],
                marker="x",
                s=28,
                label="outside tube",
                zorder=5,
            )

        # Draw phase boundaries when the corresponding node is present in the
        # comparison time vector.
        for phase_end in phase_end_steps[:-1]:
            if phase_end < len(node_times):
                phase_time = node_times[phase_end]
                if time_plot[0] <= phase_time <= time_plot[-1]:
                    ax.axvline(
                        phase_time,
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.45,
                    )

        max_error = np.max(np.abs(error_i))
        max_tube = np.max(tube_i)
        max_excess = np.max(np.abs(error_i) - tube_i)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("State deviation")
        ax.set_title(
            f"{label} | max |error|={max_error:.3e}, "
            f"max tube={max_tube:.3e}, max excess={max_excess:.3e}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        plot_path = output_dir / (
            f"{state_index:03d}_{sanitize_plot_name(label)}.png"
        )
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Summary heatmap: positive values mean the rollout left the tube.
    fig, ax = plt.subplots(figsize=(11, 6.5))
    image = ax.imshow(
        tube_excess.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[
            float(time_plot[0]),
            float(time_plot[-1]),
            -0.5,
            deviation_plot.shape[1] - 0.5,
        ],
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State dimension")
    ax.set_title("Rollout tube excess: |x_rollout - X| - tube")
    fig.colorbar(image, ax=ax, label="Tube excess")
    fig.tight_layout()
    fig.savefig(
        output_dir / "rollout_tube_excess_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "deviation": deviation,
        "deviation_compared": deviation_plot,
        "tube_radii": tubes,
        "tube_radii_compared": tube_plot,
        "tube_excess": tube_excess,
        "inside_tube": inside,
        "error_to_tube_ratio": ratio,
        "comparison_times": time_plot,
    }

def evaluate_constraint_violation(constraints, X, U):
    """Return maximum positive g(x,u,t) for constraints expressed as g <= 0."""

    terminal_control = jnp.zeros((1, U.shape[1]), dtype=U.dtype)
    controls = jnp.concatenate([U, terminal_control], axis=0)
    values = jax.vmap(constraints)(
        X, controls, jnp.arange(X.shape[0], dtype=jnp.int32)
    )
    positive_violation = jnp.maximum(values, 0.0)
    flat_index = jnp.argmax(positive_violation)
    node_index = flat_index // positive_violation.shape[1]
    constraint_index = flat_index % positive_violation.shape[1]
    return (
        float(jnp.max(positive_violation)),
        int(node_index),
        int(constraint_index),
    )


def evaluate_dynamics_defect(dynamics, X, U, parameter):
    """Return the largest absolute nonlinear shooting defect."""

    predicted_next = jax.vmap(
        lambda x, u, t: dynamics(x, u, t, parameter)
    )(
        X[:-1],
        U,
        jnp.arange(U.shape[0], dtype=jnp.int32),
    )
    absolute_defect = jnp.abs(predicted_next - X[1:])
    flat_index = jnp.argmax(absolute_defect)
    node_index = flat_index // absolute_defect.shape[1]
    state_index = flat_index % absolute_defect.shape[1]
    return (
        float(jnp.max(absolute_defect)),
        int(node_index),
        int(state_index),
    )


def waypoint_diagnostics(reference, X):
    """Return position and sign-invariant quaternion errors at phase ends."""

    states = np.asarray(X)
    positions = np.asarray(reference["p"])[np.asarray(PHASE_END_STEPS)]
    quaternions = np.asarray(reference["quat"])[np.asarray(PHASE_END_STEPS)]
    actual_quaternions = states[np.asarray(PHASE_END_STEPS), 3:7].copy()
    actual_quaternions /= np.linalg.norm(
        actual_quaternions, axis=1, keepdims=True
    )
    quaternions = quaternions.copy()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    alignment = np.abs(np.sum(actual_quaternions * quaternions, axis=1))
    angle_error = 2.0 * np.arccos(np.clip(alignment, -1.0, 1.0))
    position_error = np.abs(
        states[np.asarray(PHASE_END_STEPS), :3] - positions
    )
    return position_error, np.degrees(angle_error)


def dry_run(reference, parameter, constraints):
    x0 = jnp.asarray(config.initial_state)
    X0 = build_initial_guess(x0, reference)
    U0 = jnp.tile(config.u_ref, (config.N, 1))
    constraint_shape = jax.eval_shape(
        constraints, x0, U0[0], jnp.asarray(0)
    ).shape
    print("Obstacle barrel-roll minimum-time setup is valid.")
    print(f"State/control dimensions: {config.n}/{config.nu}")
    print(f"Reference/parameter shapes: {reference['p'].shape}/{parameter.shape}")
    print(f"Initial guess shape: {X0.shape}")
    print(f"Constraint count per node: {constraint_shape[0]}")
    print(
        "Obstacle center/size: "
        f"{np.asarray(OBSTACLE_CENTER)} / {np.asarray(OBSTACLE_SIZE)} m"
    )
    print("Nominal phase times:")
    for name, duration, intervals in zip(
        PHASE_NAMES, NOMINAL_DURATIONS, SEGMENT_LENGTHS
    ):
        print(
            f"  {name:16s} {float(duration):.3f} s "
            f"({int(intervals)} intervals)"
        )


def main(*, dry_run_only=False, output_dir=DIR_PATH):
    configure_problem()
    reference, parameter = build_barrel_roll_reference()
    obstacle_constraints = make_obstacle_clearance_constraints()
    constraints = make_barrel_roll_constraints(reference, obstacle_constraints)

    if dry_run_only:
        dry_run(reference, parameter, constraints)
        return

    admm_config = ADMMConfig(
        eps_abs=1.0e-1,
        eps_rel=1.0e-3,
        rho_max=1.0e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        num_phases=NUM_PHASES,
    )
    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )
    sqp_config = SQPConfig(
        max_sqp_iterations=50,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        line_search=True,
        lm_regularization=1.0e-2,
    )
    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=sls_config,
        sqp_config=sqp_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=config.initial_state.dtype),
        disturbance=make_phase_scaled_disturbance(config.n),
    )

    # Initialize the physical state and feet from the optimizer's model.
    model_data = mujoco.MjData(mpc.model)
    model_data.qpos[:] = np.asarray(
        jnp.concatenate([config.p0, config.quat0, config.q0])
    )
    model_data.qvel[:] = 0.0
    mujoco.mj_forward(mpc.model, model_data)
    feet = jnp.asarray(
        sim_utils.geom_positions(model_data, mpc.contact_id_mj)
    )
    x0 = (
        jnp.asarray(config.initial_state)
        .at[mpc.qpos_slice].set(jnp.asarray(model_data.qpos))
        .at[mpc.qvel_slice].set(jnp.asarray(model_data.qvel))
        .at[mpc.foot_slice].set(feet)
        .at[DURATION_SLICE].set(NOMINAL_DURATIONS)
    )
    data = mpc.make_data()

    start = timer()
    result = solve_barrel_roll(mpc, data, x0, reference, parameter)
    X, U = result[:2]
    # Current MPC solve API:
    #   X, U, V, w, y, rho, rho_grad, backoffs, Phi_x, Phi_u, ...
    Phi_x = result[8]
    converged_admm = result[-1]
    X.block_until_ready()
    solve_time = timer() - start

    # IMPORTANT: this is a sequential rollout of mpc.dynamics itself.
    # It never calls mjx.step.
    rollout_start = timer()
    X_rollout = rollout_optimized_dynamics(
        mpc.dynamics,
        X,
        U,
        parameter,
    )
    X_rollout.block_until_ready()
    rollout_time = timer() - rollout_start

    phase_times = np.asarray(X[0, DURATION_SLICE])
    total_time = float(np.sum(phase_times))
    max_violation, violation_node, violation_index = (
        evaluate_constraint_violation(constraints, X, U)
    )
    max_dynamics_defect, defect_node, defect_state = evaluate_dynamics_defect(
        mpc.dynamics, X, U, parameter
    )
    position_errors, angle_errors = waypoint_diagnostics(reference, X)
    obstacle_values = jax.vmap(obstacle_constraints)(X)
    max_obstacle_violation = float(jnp.max(jnp.maximum(obstacle_values, 0.0)))
    max_base_obstacle_violation = float(
        jnp.max(jnp.maximum(obstacle_values[:, 0], 0.0))
    )
    max_joint_obstacle_violation = float(
        jnp.max(jnp.maximum(obstacle_values[:, 1 : 1 + len(LEG_JOINT_NAMES)], 0.0))
    )
    max_foot_obstacle_violation = float(
        jnp.max(
            jnp.maximum(obstacle_values[:, 1 + len(LEG_JOINT_NAMES) :], 0.0)
        )
    )
    is_feasible = bool(
        np.isfinite(max_violation)
        and np.isfinite(max_dynamics_defect)
        and max_violation <= FEASIBILITY_TOLERANCE
        and max_dynamics_defect <= DYNAMICS_TOLERANCE
        and max_obstacle_violation <= OBSTACLE_TOLERANCE
    )

    print(f"Solve time: {solve_time:.3f} s")
    print(f"Dynamics rollout time: {rollout_time:.3f} s")
    print("Optimized phase times:")
    for name, duration in zip(PHASE_NAMES, phase_times):
        print(f"  {name:16s} {duration:.6f} s")
    print(f"Computed minimum barrel-roll time: {total_time:.6f} s")
    print(f"ADMM converged: {bool(np.asarray(converged_admm))}")
    print(
        "Maximum constraint violation: "
        f"{max_violation:.6e} at node {violation_node}, "
        f"constraint {violation_index}"
    )
    print(
        "Maximum dynamics defect: "
        f"{max_dynamics_defect:.6e} at transition {defect_node}, "
        f"state {defect_state}"
    )
    print(f"Maximum obstacle violation: {max_obstacle_violation:.6e}")
    print(
        "Base/joint/foot obstacle violations: "
        f"{max_base_obstacle_violation:.6e} / "
        f"{max_joint_obstacle_violation:.6e} / "
        f"{max_foot_obstacle_violation:.6e}"
    )
    print(f"Trajectory feasible: {is_feasible}")
    print("Phase-end waypoint errors:")
    for name, position_error, angle_error in zip(
        PHASE_NAMES, position_errors, angle_errors
    ):
        print(
            f"  {name:16s} position={position_error} "
            f"orientation={angle_error:.3f} deg"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_times = phase_node_times(phase_times)
    rollout_plot_dir = output_dir / "rollout_tube_plots"
    rollout_diagnostics = plot_rollout_deviation_vs_tubes(
        X,
        X_rollout,
        Phi_x,
        node_times,
        rollout_plot_dir,
    )

    result_path = (
        output_dir
        / "quadruped_barrel_roll_obstacle_min_time_rollout_tube_test.npz"
    )
    np.savez(
        result_path,
        X=np.asarray(X),
        U=np.asarray(U),
        X_rollout=np.asarray(X_rollout),
        Phi_x=np.asarray(Phi_x),
        rollout_deviation=rollout_diagnostics["deviation"],
        rollout_deviation_compared=rollout_diagnostics["deviation_compared"],
        tube_radii=rollout_diagnostics["tube_radii"],
        tube_radii_compared=rollout_diagnostics["tube_radii_compared"],
        tube_excess=rollout_diagnostics["tube_excess"],
        inside_tube=rollout_diagnostics["inside_tube"],
        error_to_tube_ratio=rollout_diagnostics["error_to_tube_ratio"],
        comparison_times=rollout_diagnostics["comparison_times"],
        phase_names=np.asarray(PHASE_NAMES),
        phase_end_steps=np.asarray(PHASE_END_STEPS),
        phase_times=phase_times,
        node_times=node_times,
        max_constraint_violation=np.asarray(max_violation),
        max_dynamics_defect=np.asarray(max_dynamics_defect),
        waypoint_position_errors=position_errors,
        waypoint_orientation_errors_deg=angle_errors,
        obstacle_center=np.asarray(OBSTACLE_CENTER),
        obstacle_size=np.asarray(OBSTACLE_SIZE),
        max_obstacle_violation=np.asarray(max_obstacle_violation),
        max_base_obstacle_violation=np.asarray(max_base_obstacle_violation),
        max_joint_obstacle_violation=np.asarray(max_joint_obstacle_violation),
        max_foot_obstacle_violation=np.asarray(max_foot_obstacle_violation),
        lateral_displacement=np.asarray(LATERAL_DISPLACEMENT),
        feasible=np.asarray(is_feasible),
        admm_converged=np.asarray(bool(np.asarray(converged_admm))),
        reference_version=np.asarray(4),
    )
    print(f"Saved trajectory: {result_path}")
    if not is_feasible:
        print(
            "WARNING: trajectory exceeds the configured feasibility "
            "tolerances; do not treat it as a valid motion plan."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dimensions and callbacks without solving.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIR_PATH,
        help="Directory for the NPZ result.",
    )
    args = parser.parse_args()
    main(dry_run_only=args.dry_run, output_dir=args.output_dir)
