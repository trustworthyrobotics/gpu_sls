"""Minimum-time Unitree Go2 barrel roll over a central obstacle.

The contact sequence and number of shooting intervals in each phase stay fixed.
The physical durations T_i are appended to the whole-body state and optimized
in the same way as the phase times in plan_drone_racing.

The reference starts and ends on opposite sides of a hurdle.  Smooth hard
constraints keep conservative trunk and foot envelopes outside the hurdle,
so the obstacle affects the optimized motion rather than only the rendering.
The post-roll phases preserve their pose targets without requiring contacts or
ground-reaction forces.

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
    "rolling_flight",
    "post_roll_flight",
    "terminal_posture",
    "posture_hold",
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
TIME_WEIGHT = 1e2
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
        # four-foot solve gives incorrect forces for flight.
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
    """Build hard timing, pose, obstacle, force, and airborne-end limits."""

    nj = config.n_joints
    waypoint_positions = reference["p"][PHASE_END_STEPS]
    waypoint_quaternions = reference["quat"][PHASE_END_STEPS]
    # The final waypoint keeps the original pose reference while its contact
    # mode remains fully airborne.
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
        # Keep all phase-end rate constraints unchanged except at the terminal
        # node: because this formulation intentionally ends while airborne, do
        # not require the base linear velocity to be near zero at t == N.
        waypoint_linear_speed_constraints = (
            jnp.abs(dp)[None, :] - WAYPOINT_LINEAR_SPEED_LIMITS[:, None]
        )
        linear_speed_active = active & (PHASE_END_STEPS != config.N)
        waypoint_linear_speed_constraints = jnp.where(
            linear_speed_active[:, None],
            waypoint_linear_speed_constraints,
            -jnp.ones_like(waypoint_linear_speed_constraints),
        )

        waypoint_angular_speed_constraints = (
            jnp.abs(omega)[None, :]
            - WAYPOINT_ANGULAR_SPEED_LIMITS[:, None]
        )
        waypoint_angular_speed_constraints = jnp.where(
            active[:, None],
            waypoint_angular_speed_constraints,
            -jnp.ones_like(waypoint_angular_speed_constraints),
        )

        waypoint_rate_constraints = jnp.concatenate(
            [
                waypoint_linear_speed_constraints,
                waypoint_angular_speed_constraints,
            ],
            axis=1,
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

        # The horizon terminates in midair, so the base may have nonzero
        # translational velocity.  Keep the terminal attitude-rate and joint
        # settling requirements, but do not impose a landing-speed condition.
        final_constraints = jnp.concatenate(
            [
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


# def make_phase_scaled_disturbance(n, magnitude=0.02):
#     """Disturb base position, scaled by the optimized phase timestep."""

#     def disturbance_at_state(x, t):
#         phase = phase_index(t)
#         dt = x[PHYSICAL_N + phase] / SEGMENT_LENGTHS[phase]
#         diagonal = jnp.zeros(n, dtype=x.dtype).at[:2].set(magnitude * dt)
#         diagonal = diagonal.at[DURATION_SLICE].set(0.0)
#         return jnp.diag(diagonal)

#     def disturbance(X):
#         return jax.vmap(disturbance_at_state)(X, jnp.arange(X.shape[0]))

#     disturbance.at_state = disturbance_at_state
#     return disturbance

# def make_phase_scaled_disturbance(
#     n,
#     position_magnitude=0.02,
#     velocity_magnitude=0.05,
#     joint_velocity_magnitude=0.05,
# ):
#     """
#     Disturb:
#         base x, y
#         base vx, vy
#         all joint velocities dq

#     Do not disturb:
#         base z
#         quaternion
#         joint angles
#         base vz
#         angular velocity
#         feet
#         GRFs
#         phase durations
#     """

#     nj = config.n_joints

#     # State layout:
#     #
#     # [0:3]               base position
#     # [3:7]               quaternion
#     # [7:7+nj]            joint angles
#     # [7+nj:13+nj]        vx, vy, vz, wx, wy, wz
#     # [13+nj:13+2*nj]     joint velocities dq
#     # ...

#     vel_start = 7 + nj
#     joint_vel_start = 13 + nj

#     def disturbance_at_state(x, t):
#         phase = phase_index(t)
#         dt = x[PHYSICAL_N + phase] / SEGMENT_LENGTHS[phase]

#         diagonal = jnp.zeros(n, dtype=x.dtype)

#         # Horizontal base position.
#         diagonal = diagonal.at[0].set(
#             position_magnitude * dt
#         )
#         diagonal = diagonal.at[1].set(
#             position_magnitude * dt
#         )

#         # Horizontal base velocity.
#         diagonal = diagonal.at[vel_start + 0].set(
#             velocity_magnitude * dt
#         )
#         diagonal = diagonal.at[vel_start + 1].set(
#             velocity_magnitude * dt
#         )

#         # All joint velocities.
#         # diagonal = diagonal.at[
#         #     joint_vel_start : joint_vel_start + nj
#         # ].set(
#         #     joint_velocity_magnitude * dt
#         # )

#         return jnp.diag(diagonal)

#     def disturbance(X):
#         return jax.vmap(disturbance_at_state)(
#             X,
#             jnp.arange(X.shape[0]),
#         )

#     disturbance.at_state = disturbance_at_state
#     return disturbance

def make_phase_scaled_disturbance(
    n,
    position_magnitude=0.02,
    quaternion_magnitude=0.01,
    joint_position_magnitude=0.02,
    linear_velocity_magnitude=0.05,
    angular_velocity_magnitude=0.05,
    joint_velocity_magnitude=0.05,
    foot_position_magnitude=0.02,
    grf_magnitude=5.0,
):
    """
    Disturb every physical state except the optimized phase durations.

    State layout:
        [0:3]                       base position
        [3:7]                       quaternion
        [7:7+nj]                    joint positions
        [7+nj:10+nj]                base linear velocity
        [10+nj:13+nj]               base angular velocity
        [13+nj:13+2*nj]             joint velocities
        [13+2*nj:13+2*nj+3*nc]      foot positions
        [GRF_START:GRF_STOP]         GRFs
        [DURATION_SLICE]             phase durations  <-- NO disturbance

    All disturbance magnitudes are scaled by the physical phase timestep:

        dt = T_phase / N_phase

    so that changing the optimized phase duration changes the discrete-time
    disturbance consistently.
    """

    nj = config.n_joints
    nc = config.n_contact

    # State indices.
    base_pos_start = 0
    quat_start = 3
    joint_pos_start = 7

    linear_vel_start = 7 + nj
    angular_vel_start = 10 + nj
    joint_vel_start = 13 + nj

    foot_start = 13 + 2 * nj
    foot_stop = foot_start + 3 * nc

    def disturbance_at_state(x, t):
        phase = phase_index(t)

        dt = (
            x[PHYSICAL_N + phase]
            / SEGMENT_LENGTHS[phase]
        )

        diagonal = jnp.zeros(n, dtype=x.dtype)

        # ---------------------------------------------------------
        # Base position: x, y, z
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            base_pos_start : base_pos_start + 3
        ].set(
            position_magnitude * dt
        )

        # ---------------------------------------------------------
        # Quaternion: qw, qx, qy, qz
        #
        # Keep this small because quaternion components are not
        # independent Euclidean coordinates.
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            quat_start : quat_start + 4
        ].set(
            quaternion_magnitude * dt
        )

        # ---------------------------------------------------------
        # Joint positions
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            joint_pos_start : joint_pos_start + nj
        ].set(
            joint_position_magnitude * dt
        )

        # ---------------------------------------------------------
        # Base linear velocity: vx, vy, vz
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            linear_vel_start : linear_vel_start + 3
        ].set(
            linear_velocity_magnitude * dt
        )

        # ---------------------------------------------------------
        # Base angular velocity: wx, wy, wz
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            angular_vel_start : angular_vel_start + 3
        ].set(
            angular_velocity_magnitude * dt
        )

        # ---------------------------------------------------------
        # Joint velocities
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            joint_vel_start : joint_vel_start + nj
        ].set(
            joint_velocity_magnitude * dt
        )

        # ---------------------------------------------------------
        # Foot positions
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            foot_start : foot_stop
        ].set(
            foot_position_magnitude * dt
        )

        # ---------------------------------------------------------
        # Ground reaction forces
        # ---------------------------------------------------------
        diagonal = diagonal.at[
            GRF_START : GRF_STOP
        ].set(
            grf_magnitude * dt
        )

        # ---------------------------------------------------------
        # Phase-duration states:
        #
        # diagonal[DURATION_SLICE] remains exactly zero.
        # ---------------------------------------------------------

        return jnp.diag(diagonal)

    def disturbance(X):
        return jax.vmap(
            disturbance_at_state,
            in_axes=(0, 0),
        )(
            X,
            jnp.arange(X.shape[0], dtype=jnp.int32),
        )

    disturbance.at_state = disturbance_at_state

    return disturbance

def get_trajectory_tubes(Phi_x):
    """Return the per-node, per-state SLS tube radius."""
    return jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)


def disturbance_response_to_state_history_feedback(
    Phi_x,
    Phi_u,
    physical_n=PHYSICAL_N,
):
    """Convert SLS disturbance feedback into state-error-history feedback.

    Tensor convention:

        Phi_x[j, k] = effect of disturbance w_k on state perturbation dx_j
        Phi_u[j, k] = effect of disturbance w_k on control perturbation du_j

    In this script the terminal Phi_x response row is clipped before this
    function is called, so the expected shapes are

        Phi_x : (N, N+1, nx, nx)
        Phi_u : (N, N+1, nu, nx)

    The final disturbance-history column is retained in the stored tensors so
    their convention is unchanged, but only columns 0,...,N-1 can participate
    in the N causal controls.  We therefore form the square causal operators

        Rx = Phi_x[:, :N]
        Ru = Phi_u[:, :N]

    and solve

        K_x Rx = Ru,

    i.e.

        K_x = Ru Rx^{-1}.

    The runtime controller is

        u_k = U_k + sum_{j=0}^k K_x[k,j] (x_j - X_j).

    Duration-state disturbance channels are zero by construction, so the
    inversion is performed only on the PHYSICAL_N physical-state subspace.
    """

    Phi_x_np = np.asarray(Phi_x)
    Phi_u_np = np.asarray(Phi_u)

    if Phi_x_np.ndim != 4 or Phi_u_np.ndim != 4:
        raise ValueError(
            "Expected rank-4 SLS response tensors; got "
            f"Phi_x={Phi_x_np.shape}, Phi_u={Phi_u_np.shape}."
        )

    N = Phi_u_np.shape[0]
    history_cols = Phi_u_np.shape[1]

    if Phi_x_np.shape[0] != N:
        raise ValueError(
            "Phi_x must have one response row per control after clipping; got "
            f"Phi_x={Phi_x_np.shape}, Phi_u={Phi_u_np.shape}."
        )

    if Phi_x_np.shape[1] != history_cols:
        raise ValueError(
            "Phi_x and Phi_u must use the same disturbance-history columns; got "
            f"Phi_x={Phi_x_np.shape}, Phi_u={Phi_u_np.shape}."
        )

    if history_cols < N:
        raise ValueError(
            f"Need at least N={N} disturbance-history columns; got {history_cols}."
        )

    if Phi_x_np.shape[2] < physical_n or Phi_x_np.shape[3] < physical_n:
        raise ValueError(
            f"Phi_x is too small for physical_n={physical_n}: {Phi_x_np.shape}."
        )

    if Phi_u_np.shape[3] < physical_n:
        raise ValueError(
            f"Phi_u is too small for physical_n={physical_n}: {Phi_u_np.shape}."
        )

    # Keep the requested (N, N+1, ...) tensors externally, but only the first
    # N disturbance columns can affect the N controls causally.
    Phi_x_causal = Phi_x_np[:, :N, :physical_n, :physical_n]
    Phi_u_causal = Phi_u_np[:, :N, :, :physical_n]

    nu = Phi_u_causal.shape[2]
    K_x = np.zeros((N, N, nu, physical_n), dtype=Phi_u_causal.dtype)

    # Solve K_x Phi_x = Phi_u blockwise.  With Phi(j,k) denoting the response
    # at time j to disturbance k, causality means Phi[j,k] = 0 for k > j.
    # For fixed control row k:
    #
    #   Phi_u[k,j] = sum_{s=j}^k K_x[k,s] Phi_x[s,j].
    #
    # Working backward in j makes all s>j terms available before solving the
    # right multiplication by Phi_x[j,j].
    for k in range(N):
        for j in range(k, -1, -1):
            rhs = Phi_u_causal[k, j].copy()

            if j < k:
                rhs -= np.einsum(
                    "smp,spq->mq",
                    K_x[k, j + 1 : k + 1],
                    Phi_x_causal[j + 1 : k + 1, j],
                )

            diagonal_block = Phi_x_causal[j, j]
            try:
                K_x[k, j] = np.linalg.solve(
                    diagonal_block.T,
                    rhs.T,
                ).T
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(
                    "State-history conversion failed because the physical "
                    f"Phi_x[{j},{j}] block is singular. Its shape is "
                    f"{diagonal_block.shape}."
                ) from exc

    # Reconstruction sanity check on the causal physical subspace.
    reconstructed = np.einsum(
        "ksmp,sjpq->kjmq",
        K_x,
        Phi_x_causal,
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - Phi_u_causal))
    )
    print(
        "State-history feedback reconstruction max error: "
        f"{reconstruction_error:.6e}"
    )

    return jnp.asarray(K_x)


def rollout_state_history_feedback(
    dynamics,
    X_nominal,
    U_nominal,
    K_x,
    parameter,
):
    """Roll out the nonlinear plant with causal state-perturbation history.

    At control step k the controller has observed dx_0,...,dx_k and applies

        u_k = U_nominal[k] + sum_{j=0}^k K_x[k,j] dx_j,

    with

        dx_j = x_j - X_nominal[j].

    The complete realized perturbation history is retained and returned.
    """

    N = U_nominal.shape[0]
    if K_x.shape[0] != N or K_x.shape[1] != N:
        raise ValueError(
            f"K_x horizon shape {K_x.shape[:2]} does not match N={N}."
        )

    x = X_nominal[0]
    error_history = jnp.zeros_like(X_nominal)
    error_history = error_history.at[0].set(x - X_nominal[0])

    X_feedback = [x]
    U_feedback = []
    feedback_corrections = []

    for k in range(N):
        observed_errors = error_history[: k + 1, :PHYSICAL_N]
        delta_u = jnp.einsum(
            "jmp,jp->m",
            K_x[k, : k + 1],
            observed_errors,
        )

        u = U_nominal[k] + delta_u
        x_next = dynamics(
            x,
            u,
            jnp.asarray(k, dtype=jnp.int32),
            parameter,
        )

        error_history = error_history.at[k + 1].set(
            x_next - X_nominal[k + 1]
        )

        X_feedback.append(x_next)
        U_feedback.append(u)
        feedback_corrections.append(delta_u)
        x = x_next

    return (
        jnp.stack(X_feedback),
        jnp.stack(U_feedback),
        error_history,
        jnp.stack(feedback_corrections),
    )


def rollout_zero_disturbance(dynamics, x0, U, parameter):
    """Roll out the exact nonlinear dynamics with the optimized controls.

    No disturbance is injected here.  This is therefore the open-loop nonlinear
    rollout implied by x0 and U, and its deviation from the multiple-shooting
    trajectory X measures nonlinear / shooting consistency rather than a
    disturbance response.
    """

    def step(x, inputs):
        u, t = inputs
        x_next = dynamics(x, u, t, parameter)
        return x_next, x_next

    _, successors = jax.lax.scan(
        step,
        x0,
        (U, jnp.arange(U.shape[0], dtype=jnp.int32)),
    )
    return jnp.concatenate([x0[None, :], successors], axis=0)


def state_dimension_names():
    """Human-readable labels for every augmented state dimension."""
    nj = config.n_joints
    nc = config.n_contact

    names = [
        "base_x", "base_y", "base_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
    ]
    names += [f"q_{i}" for i in range(nj)]
    names += [
        "v_x", "v_y", "v_z", "omega_x", "omega_y", "omega_z",
    ]
    names += [f"dq_{i}" for i in range(nj)]
    names += [f"foot_{i}_{axis}" for i in range(nc) for axis in ("x", "y", "z")]
    names += [f"grf_{i}_{axis}" for i in range(nc) for axis in ("x", "y", "z")]
    names += [f"duration_{name}" for name in PHASE_NAMES]

    if len(names) != config.n:
        return [f"state_{i}" for i in range(config.n)]
    return names


def align_tubes_to_states(Phi_x, num_state_nodes):
    """Align get_trajectory_tubes(Phi_x) with x_0, ..., x_N."""
    tubes = get_trajectory_tubes(Phi_x)

    if tubes.shape[0] == num_state_nodes:
        return tubes
    if tubes.shape[0] == num_state_nodes - 1:
        # The initial state is fixed, so its disturbance tube is exactly zero.
        return jnp.concatenate(
            [jnp.zeros((1, tubes.shape[1]), dtype=tubes.dtype), tubes], axis=0
        )

    raise ValueError(
        "Cannot align Phi_x tubes with the state trajectory: "
        f"tube nodes={tubes.shape[0]}, state nodes={num_state_nodes}."
    )


def plot_zero_rollout_vs_tubes(
    X_optimized,
    X_rollout,
    Phi_x,
    phase_times,
    output_path,
):
    """Plot |zero-disturbance nonlinear rollout - optimized X| vs tube radius.

    Every augmented state dimension is placed in its own subplot, while all
    subplots are saved into one PNG.
    """
    X_optimized_np = np.asarray(X_optimized)
    X_rollout_np = np.asarray(X_rollout)
    tubes_np = np.asarray(align_tubes_to_states(Phi_x, X_optimized_np.shape[0]))

    if X_rollout_np.shape != X_optimized_np.shape:
        raise ValueError(
            f"Rollout shape {X_rollout_np.shape} does not match optimized "
            f"trajectory shape {X_optimized_np.shape}."
        )
    if tubes_np.shape != X_optimized_np.shape:
        raise ValueError(
            f"Tube shape {tubes_np.shape} does not match state trajectory "
            f"shape {X_optimized_np.shape}."
        )

    deviation = np.abs(X_rollout_np - X_optimized_np)
    times = phase_node_times(phase_times)
    names = state_dimension_names()

    n_states = X_optimized_np.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_states / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 2.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        if i >= n_states:
            ax.axis("off")
            continue

        ax.plot(times, deviation[:, i], label="|rollout - optimized|")
        ax.plot(times, tubes_np[:, i], "--", label="tube radius")
        ax.set_title(f"{i}: {names[i]}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i % n_cols == 0:
            ax.set_ylabel("magnitude")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("time [s]")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Zero-disturbance nonlinear rollout deviation vs. SLS tube size",
        y=0.999,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.995))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return deviation, tubes_np

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
        eps_abs=6.0e-3,
        eps_rel=1.0e-3,
        rho_max=1.0e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=25.0,
        regularized_rho_update=False,
        num_phases=NUM_PHASES,
    )
    sls_config = SLSConfig(
        max_sls_iterations=2,
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
        lm_regularization=5.0e-2,
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
    Phi_x_full = result[8]
    Phi_u = result[9]

    # SLS returns one extra terminal state-response row.  For the feedback
    # controller, retain one Phi_x row per control while keeping all 51
    # disturbance-history columns:
    #
    #   (51, 51, 67, 67) -> (50, 51, 67, 67)
    Phi_x = Phi_x_full[: Phi_u.shape[0], ...]
    print(f"Clipped Phi_x: {Phi_x_full.shape} -> {Phi_x.shape}")
    print(f"Phi_u shape: {Phi_u.shape}")

    converged_admm = result[-1]
    X.block_until_ready()
    solve_time = timer() - start

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

    # Open-loop nonlinear rollout, retained as the original diagnostic.
    X_rollout = rollout_zero_disturbance(mpc.dynamics, X[0], U, parameter)
    X_rollout.block_until_ready()
    rollout_plot_path = output_dir / "quadruped_barrel_roll_zero_rollout_vs_tubes.png"
    rollout_deviation, trajectory_tubes = plot_zero_rollout_vs_tubes(
        X,
        X_rollout,
        Phi_x,
        phase_times,
        rollout_plot_path,
    )
    print(f"Saved zero-disturbance rollout/tube plot: {rollout_plot_path}")
    print(
        "Maximum zero-disturbance rollout deviation: "
        f"{float(np.max(rollout_deviation)):.6e}"
    )

    # Convert the SLS disturbance-feedback policy
    #
    #   du_k = sum_j Phi_u[k,j] w_j
    #
    # into a causal state-history policy
    #
    #   du_k = sum_s K_x[k,s] (x_{s+1} - X_{s+1}),
    #   K_x = Phi_u Phi_x^{-1}
    #
    # on the physical-state subspace.
    K_x_history = disturbance_response_to_state_history_feedback(Phi_x, Phi_u)
    (
        X_feedback_rollout,
        U_feedback_rollout,
        x_perturbation_history,
        feedback_corrections,
    ) = rollout_state_history_feedback(
        mpc.dynamics,
        X,
        U,
        K_x_history,
        parameter,
    )
    X_feedback_rollout.block_until_ready()

    max_feedback_state_error = float(
        jnp.max(jnp.abs(x_perturbation_history[:, :PHYSICAL_N]))
    )
    max_feedback_torque_correction = float(jnp.max(jnp.abs(feedback_corrections)))
    print(
        "Maximum state-history-feedback rollout perturbation: "
        f"{max_feedback_state_error:.6e}"
    )
    print(
        "Maximum state-history-feedback torque correction: "
        f"{max_feedback_torque_correction:.6e}"
    )

    result_path = output_dir / "quadruped_barrel_roll_obstacle_min_time.npz"
    np.savez(
        result_path,
        X=np.asarray(X),
        U=np.asarray(U),
        X_zero_disturbance_rollout=np.asarray(X_rollout),
        zero_rollout_deviation=np.asarray(rollout_deviation),
        trajectory_tubes=np.asarray(trajectory_tubes),
        Phi_x=np.asarray(Phi_x),
        Phi_x_full=np.asarray(Phi_x_full),
        Phi_u=np.asarray(Phi_u),
        K_x_history=np.asarray(K_x_history),
        X_state_feedback_rollout=np.asarray(X_feedback_rollout),
        U_state_feedback_rollout=np.asarray(U_feedback_rollout),
        x_perturbation_history=np.asarray(x_perturbation_history),
        feedback_corrections=np.asarray(feedback_corrections),
        phase_names=np.asarray(PHASE_NAMES),
        phase_end_steps=np.asarray(PHASE_END_STEPS),
        phase_times=phase_times,
        node_times=phase_node_times(phase_times),
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
