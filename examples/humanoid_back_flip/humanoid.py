"""Minimum-time, multi-phase backflip for the Unitree H1 humanoid.

The contact sequence and shooting intervals per phase stay fixed. Physical
phase durations are appended to the whole-body state and optimized, following
the quadruped barrel-roll example. Use ``--dry-run`` to validate the setup.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

DIR_PATH = Path(__file__).resolve().parent
sys.path.append(str(DIR_PATH.parent))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math

import config_h1 as config
import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
import mpx.utils.sim as sim_utils

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

PHASE_NAMES = (
    "stand", "crouch", "launch", "first_half", "second_half",
    "prelanding", "touchdown", "settle",
)
NOMINAL_DURATIONS = jnp.array(
    [0.10, 0.14, 0.12, 0.16, 0.16, 0.06, 0.10, 0.16]
)
PHASE_END_STEPS = jnp.array(
    [5, 12, 18, 26, 34, 37, 42, 50], dtype=jnp.int32
)
SEGMENT_LENGTHS = jnp.concatenate(
    [PHASE_END_STEPS[:1], PHASE_END_STEPS[1:] - PHASE_END_STEPS[:-1]]
)
NUM_PHASES = len(PHASE_NAMES)
if config.N != int(PHASE_END_STEPS[-1]):
    raise ValueError(f"Phase layout requires config.N=50; got {config.N}.")

MIN_DURATIONS = 0.15 * NOMINAL_DURATIONS
MAX_DURATIONS = 2.00 * NOMINAL_DURATIONS
# A terminal time cost that dominates the tracking terms drives every contact
# phase to its lower bound and produces an impulsive, visibly unnatural flip.
# This value still rewards shorter motions while leaving enough curvature from
# the motion cost to regularize the direct minimum-time solve.
TIME_WEIGHT = 3.0e3
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
DURATION_SLICE = slice(PHYSICAL_N, PHYSICAL_N + NUM_PHASES)
FOOT_START = 13 + 2 * config.n_joints
GRF_START = FOOT_START + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact
QVEL_START = 7 + config.n_joints
QVEL_STOP = 13 + 2 * config.n_joints

# Optimize nondimensional velocity and force coordinates.  Raw H1 joint rates
# (O(10) rad/s) and contact forces (O(100) N) otherwise dominate the SQP
# residual beside O(1) positions and quaternions.
VELOCITY_SCALE = jnp.concatenate([
    jnp.ones(3) * 3.0,
    jnp.ones(3) * 20.0,
    jnp.ones(config.n_joints) * 30.0,
])
GRF_SCALE = 500.0

JOINT_MIN = jnp.array([
    -0.43, -0.43, -1.57, -0.26, -0.87,
    -0.43, -0.43, -1.57, -0.26, -0.87, -2.35,
    -2.87, -0.34, -1.30, -1.25, -2.87, -3.11, -4.45, -1.25,
])
JOINT_MAX = jnp.array([
    0.43, 0.43, 1.57, 2.05, 0.52,
    0.43, 0.43, 1.57, 2.05, 0.52, 2.35,
    2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.30, 2.61,
])
TORQUE_LIMIT = jnp.array([
    200.0, 200.0, 200.0, 300.0, 40.0,
    200.0, 200.0, 200.0, 300.0, 40.0, 200.0,
    40.0, 40.0, 18.0, 18.0, 40.0, 40.0, 18.0, 18.0,
])

WAYPOINT_POSITION_HALF_WIDTHS = jnp.array([
    [0.06, 0.06, 0.06], [0.12, 0.08, 0.12],
    [0.20, 0.10, 0.25], [0.25, 0.12, 0.25],
    [0.30, 0.12, 0.30], [0.30, 0.12, 0.25],
    [0.20, 0.10, 0.20], [0.08, 0.06, 0.08],
])
WAYPOINT_ANGLE_TOLERANCES = jnp.deg2rad(
    jnp.array([8.0, 15.0, 25.0, 30.0, 35.0, 30.0, 20.0, 8.0])
)
WAYPOINT_LINEAR_SPEED_LIMITS = jnp.array(
    [0.4, 1.5, 4.0, 5.0, 5.0, 4.0, 2.0, 0.5]
)
WAYPOINT_ANGULAR_SPEED_LIMITS = jnp.array(
    [0.8, 5.0, 25.0, 30.0, 30.0, 20.0, 8.0, 0.8]
)
FINAL_LINEAR_SPEED = 0.5
FINAL_ANGULAR_SPEED = 0.8
FINAL_JOINT_SPEED = 3.0
FINAL_JOINT_POSITION_TOLERANCE = 0.25
MAX_JOINT_SPEED = 50.0
MIN_BASE_HEIGHT = 0.35
FRICTION_COEFFICIENT = 0.7
MAX_NORMAL_FORCE = 900.0
TANGENTIAL_FORCE_SMOOTHING = 1.0e-4
CONTACT_SOLVE_REGULARIZATION = 1.0e-3
CONTACT_POSITION_GAIN = 20.0
CONTACT_VELOCITY_GAIN = 20.0
QUATERNION_EPSILON = 1.0e-8
QUATERNION_NORM_WEIGHT = 1.0e4
DYNAMICS_TOLERANCE = 1.0e-2

# Joint-space feedback used only to initialize the controls from the analytic
# reference.  It is not the result of a separate optimization.
INITIAL_KP = jnp.array(
    [120.0] * 10 + [80.0] + [35.0] * 8
)
INITIAL_KD = jnp.array(
    [6.0] * 10 + [5.0] + [2.0] * 8
)


def _duration_logits(durations):
    """Map physical phase durations to unconstrained solver coordinates."""
    fraction = (durations - MIN_DURATIONS) / (MAX_DURATIONS - MIN_DURATIONS)
    fraction = jnp.clip(fraction, 1.0e-6, 1.0 - 1.0e-6)
    return jnp.log(fraction) - jnp.log1p(-fraction)


NOMINAL_DURATION_LOGITS = _duration_logits(NOMINAL_DURATIONS)


def phase_durations(x):
    """Decode strictly bounded physical durations from the solver state."""
    logits = x[DURATION_SLICE]
    return MIN_DURATIONS + (
        MAX_DURATIONS - MIN_DURATIONS
    ) * jax.nn.sigmoid(logits)


def safe_normalize_quaternion(quat):
    """Return a valid unit quaternion, including for a zero SQP iterate."""
    norm_squared = jnp.sum(quat * quat)
    normalized = quat / jnp.sqrt(jnp.maximum(norm_squared, QUATERNION_EPSILON))
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype)
    return jnp.where(norm_squared > QUATERNION_EPSILON, normalized, identity)


def phase_index(t):
    index = jnp.searchsorted(PHASE_END_STEPS, t, side="right")
    return jnp.clip(index, 0, NUM_PHASES - 1)


def project_contact_forces(grf, contact):
    """Project contact forces onto the unilateral Coulomb friction cones."""
    normal = jnp.clip(grf[:, 2], 0.0, MAX_NORMAL_FORCE)
    tangent = grf[:, :2]
    tangent_norm = jnp.sqrt(
        jnp.sum(tangent * tangent, axis=1)
        + TANGENTIAL_FORCE_SMOOTHING**2
    )
    scale = jnp.minimum(
        1.0,
        FRICTION_COEFFICIENT * normal / tangent_norm,
    )
    projected = jnp.concatenate(
        [tangent * scale[:, None], normal[:, None]], axis=1
    )
    return projected * contact[:, None]


def min_time_backflip_dynamics(model, mjx_model, contact_id, body_id):
    """Build H1 dynamics with dt_k = T_phase(k) / N_phase(k)."""
    nj = config.n_joints
    contact_body_ids = (body_id[0], body_id[0], body_id[1], body_id[1])

    def dynamics(x, u, t, parameter):
        phase = phase_index(t)
        dt = phase_durations(x)[phase] / SEGMENT_LENGTHS[phase]
        qpos = x[: nj + 7].at[3:7].set(
            safe_normalize_quaternion(x[3:7])
        )
        qvel = x[QVEL_START:QVEL_STOP] * VELOCITY_SCALE
        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        data = mjx.fwd_position(mjx_model, data)
        data = mjx.fwd_velocity(mjx_model, data)
        mass_factor = data.qLD
        bias_force = data.qfrc_bias
        contact = parameter[t, : config.n_contact]
        tau = jnp.concatenate([jnp.zeros(6, dtype=x.dtype), u])

        feet = [data.geom_xpos[geom_id] for geom_id in contact_id]
        jacobians = [
            mjx.jac(mjx_model, data, foot, ankle_id)[0]
            for foot, ankle_id in zip(feet, contact_body_ids)
        ]
        contact_jacobian = jnp.concatenate(jacobians, axis=1)
        current_feet = jnp.concatenate(feet)
        contact_mask = jnp.repeat(contact, 3)
        active_jacobian = contact_jacobian * contact_mask[None, :]
        velocity_violation = active_jacobian.T @ qvel
        target_feet = parameter[
            t, config.n_contact : config.n_contact + 3 * config.n_contact
        ]
        position_violation = contact_mask * (current_feet - target_feet)
        mass_inv_jacobian = jax.scipy.linalg.cho_solve(
            (mass_factor, False), active_jacobian
        )
        projected_mass = active_jacobian.T @ mass_inv_jacobian
        projected_mass = 0.5 * (projected_mass + projected_mass.T)
        projected_mass = (
            projected_mass + jnp.diag(1.0 - contact_mask)
            + CONTACT_SOLVE_REGULARIZATION
            * jnp.eye(projected_mass.shape[0], dtype=x.dtype)
        )
        rhs = (
            -active_jacobian.T
            @ jax.scipy.linalg.cho_solve((mass_factor, False), tau - bias_force)
            - CONTACT_VELOCITY_GAIN * velocity_violation
            - CONTACT_POSITION_GAIN * position_violation
        )
        raw_grf = jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(projected_mass), rhs
        ).reshape(config.n_contact, 3)
        grf = project_contact_forces(raw_grf, contact).reshape(-1)
        acceleration = jax.scipy.linalg.cho_solve(
            (mass_factor, False), tau - bias_force + active_jacobian @ grf
        )
        velocity_next = qvel + acceleration * dt
        physical_next = jnp.concatenate([
            qpos[:3] + velocity_next[:3] * dt,
            math.quat_integrate(qpos[3:7], velocity_next[3:6], dt),
            qpos[7:] + velocity_next[6:] * dt,
            velocity_next / VELOCITY_SCALE,
            current_feet,
            grf / GRF_SCALE,
        ])
        return jnp.concatenate([physical_next, x[DURATION_SLICE]])

    return dynamics


def min_time_backflip_cost(W, reference, x, u, t):
    nj = config.n_joints
    p, quat_raw = x[:3], x[3:7]
    quat = safe_normalize_quaternion(quat_raw)
    q = x[7 : 7 + nj]
    dp = x[7 + nj : 10 + nj] * VELOCITY_SCALE[:3]
    omega = x[10 + nj : 13 + nj] * VELOCITY_SCALE[3:6]
    dq = x[13 + nj : 13 + 2 * nj] * VELOCITY_SCALE[6:]
    feet = x[FOOT_START:GRF_START]
    grf = x[GRF_START:GRF_STOP] * GRF_SCALE
    qerr = math.quat_sub(quat, reference["quat"][t])
    contact_map = jnp.repeat(reference["contact"][t], 3)
    stage_cost = (
        (p - reference["p"][t]).T @ W["pos"] @ (p - reference["p"][t])
        + qerr.T @ W["rot"] @ qerr
        + (q - reference["q"][t]).T @ W["q"] @ (q - reference["q"][t])
        + (dp - reference["dp"][t]).T @ W["vel"] @ (dp - reference["dp"][t])
        + (omega - reference["omega"][t]).T @ W["omega"]
        @ (omega - reference["omega"][t])
        + (dq - reference["dq"][t]).T @ W["dq"]
        @ (dq - reference["dq"][t])
        + (contact_map * (feet - reference["foot"][t])).T
        @ W["contact"] @ (contact_map * (feet - reference["foot"][t]))
        + u.T @ W["tau"] @ u
        + (grf - reference["grf"][t]).T @ W["grf"]
        @ (grf - reference["grf"][t])
        + QUATERNION_NORM_WEIGHT
        * (jnp.sum(quat_raw * quat_raw) - 1.0) ** 2
    )
    time_cost = jnp.where(
        t == config.N, jnp.dot(W["time"], phase_durations(x)),
        jnp.asarray(0.0, dtype=x.dtype),
    )
    return 0.5 * stage_cost + time_cost


def _blend_pose(nodes, start, stop, q_start, q_stop):
    alpha = jnp.clip((nodes - start) / max(stop - start, 1), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return q_start + alpha[:, None] * (q_stop - q_start)


def build_backflip_reference():
    """Construct an N+1 reference with a complete backward pitch rotation."""
    nodes = jnp.arange(config.N + 1, dtype=jnp.float32)
    q0 = jnp.asarray(config.q0)
    crouch_q = q0.at[2].set(-1.05).at[3].set(1.90).at[4].set(-0.85)
    crouch_q = crouch_q.at[7].set(-1.05).at[8].set(1.90).at[9].set(-0.85)
    crouch_q = crouch_q.at[11].set(1.40).at[15].set(1.40)
    launch_q = q0.at[2].set(-0.25).at[3].set(0.50).at[4].set(-0.30)
    launch_q = launch_q.at[7].set(-0.25).at[8].set(0.50).at[9].set(-0.30)
    launch_q = launch_q.at[11].set(-1.80).at[15].set(-1.80)
    tuck_q = q0.at[2].set(-1.45).at[3].set(2.00).at[4].set(-0.55)
    tuck_q = tuck_q.at[7].set(-1.45).at[8].set(2.00).at[9].set(-0.55)
    tuck_q = tuck_q.at[10].set(0.25).at[14].set(1.80).at[18].set(1.80)
    landing_q = q0.at[2].set(-0.70).at[3].set(1.45).at[4].set(-0.78)
    landing_q = landing_q.at[7].set(-0.70).at[8].set(1.45).at[9].set(-0.78)
    landing_q = landing_q.at[11].set(0.80).at[15].set(0.80)

    q_ref = jnp.tile(q0, (config.N + 1, 1))
    segments = (
        (5, 12, q0, crouch_q), (12, 18, crouch_q, launch_q),
        (18, 26, launch_q, tuck_q), (26, 34, tuck_q, landing_q),
        (34, 37, landing_q, landing_q), (37, 42, landing_q, crouch_q),
        (42, 50, crouch_q, q0),
    )
    for start, stop, q_start, q_stop in segments:
        values = _blend_pose(nodes, start, stop, q_start, q_stop)
        q_ref = jnp.where(
            ((nodes >= start) & (nodes <= stop))[:, None], values, q_ref
        )

    stand_end, crouch_end, launch_end = 5, 12, 18
    flight_end, prelanding_end, touchdown_end = 34, 37, 42
    launch_duration = float(NOMINAL_DURATIONS[2])
    airborne_duration = float(sum(NOMINAL_DURATIONS[3:6]))
    takeoff_speed = 0.5 * 9.81 * airborne_duration
    standing_height = float(config.p0[2])
    crouch_height = standing_height - 0.5 * takeoff_speed * launch_duration
    p_ref = jnp.zeros((config.N + 1, 3)).at[:, 2].set(standing_height)
    alpha = jnp.clip(
        (nodes - stand_end) / (crouch_end - stand_end), 0.0, 1.0
    )
    alpha = alpha**2 * (3.0 - 2.0 * alpha)
    crouch_z = standing_height + alpha * (crouch_height - standing_height)
    active = (nodes >= stand_end) & (nodes <= crouch_end)
    p_ref = p_ref.at[:, 2].set(jnp.where(active, crouch_z, p_ref[:, 2]))
    launch_time = (nodes - crouch_end) * launch_duration / (launch_end - crouch_end)
    launch_z = crouch_height + 0.5 * (takeoff_speed / launch_duration) * launch_time**2
    active = (nodes > crouch_end) & (nodes <= launch_end)
    p_ref = p_ref.at[:, 2].set(jnp.where(active, launch_z, p_ref[:, 2]))
    air_time = (nodes - launch_end) * airborne_duration / (prelanding_end - launch_end)
    air_z = standing_height + takeoff_speed * air_time - 0.5 * 9.81 * air_time**2
    active = (nodes > launch_end) & (nodes <= prelanding_end)
    p_ref = p_ref.at[:, 2].set(jnp.where(active, air_z, p_ref[:, 2]))

    # Absorb the landing by returning to a crouch before standing.  Keeping
    # the pelvis at standing height while commanding deeply bent knees was
    # kinematically inconsistent with the fixed stance-foot references.
    alpha = jnp.clip(
        (nodes - prelanding_end) / (touchdown_end - prelanding_end),
        0.0,
        1.0,
    )
    alpha = alpha**2 * (3.0 - 2.0 * alpha)
    touchdown_z = standing_height + alpha * (
        crouch_height - standing_height
    )
    active = (nodes > prelanding_end) & (nodes <= touchdown_end)
    p_ref = p_ref.at[:, 2].set(jnp.where(active, touchdown_z, p_ref[:, 2]))
    alpha = jnp.clip(
        (nodes - touchdown_end) / (config.N - touchdown_end), 0.0, 1.0
    )
    alpha = alpha**2 * (3.0 - 2.0 * alpha)
    settle_z = crouch_height + alpha * (standing_height - crouch_height)
    active = nodes > touchdown_end
    p_ref = p_ref.at[:, 2].set(jnp.where(active, settle_z, p_ref[:, 2]))

    phase_ids = phase_index(jnp.arange(config.N))
    node_times = jnp.concatenate([
        jnp.asarray([0.0]),
        jnp.cumsum(NOMINAL_DURATIONS[phase_ids] / SEGMENT_LENGTHS[phase_ids]),
    ])
    dp_ref = jnp.zeros((config.N + 1, 3))
    dp_ref = dp_ref.at[1:-1].set(
        (p_ref[2:] - p_ref[:-2]) / (node_times[2:, None] - node_times[:-2, None])
    )
    dp_ref = dp_ref.at[0].set((p_ref[1] - p_ref[0]) / (node_times[1] - node_times[0]))

    # Accelerate the pitch rate while contact can create angular momentum,
    # preserve it in flight, then remove it during the grounded touchdown.  The
    # former reference jumped from zero to about -19 rad/s at takeoff and back
    # to zero one node before contact, making its own dynamics infeasible.
    launch_start_time = node_times[crouch_end]
    takeoff_time = node_times[launch_end]
    contact_time = node_times[prelanding_end]
    rotation_stop_time = node_times[touchdown_end]
    rotation_launch_time = takeoff_time - launch_start_time
    rotation_flight_time = contact_time - takeoff_time
    rotation_landing_time = rotation_stop_time - contact_time
    flight_pitch_rate = -2.0 * jnp.pi / (
        0.5 * rotation_launch_time
        + rotation_flight_time
        + 0.5 * rotation_landing_time
    )
    launch_elapsed = jnp.clip(
        node_times - launch_start_time, 0.0, rotation_launch_time
    )
    launch_angle = (
        0.5
        * flight_pitch_rate
        / rotation_launch_time
        * launch_elapsed**2
    )
    takeoff_angle = 0.5 * flight_pitch_rate * rotation_launch_time
    flight_elapsed = jnp.clip(
        node_times - takeoff_time, 0.0, rotation_flight_time
    )
    airborne_angle = takeoff_angle + flight_pitch_rate * flight_elapsed
    contact_angle = takeoff_angle + flight_pitch_rate * rotation_flight_time
    landing_elapsed = jnp.clip(
        node_times - contact_time, 0.0, rotation_landing_time
    )
    landing_angle = (
        contact_angle
        + flight_pitch_rate * landing_elapsed
        - 0.5
        * flight_pitch_rate
        / rotation_landing_time
        * landing_elapsed**2
    )
    angle = jnp.where(node_times < launch_start_time, 0.0, launch_angle)
    angle = jnp.where(node_times > takeoff_time, airborne_angle, angle)
    angle = jnp.where(node_times > contact_time, landing_angle, angle)
    angle = jnp.where(node_times >= rotation_stop_time, -2.0 * jnp.pi, angle)
    quat_ref = jnp.stack([
        jnp.cos(0.5 * angle), jnp.zeros_like(angle),
        jnp.sin(0.5 * angle), jnp.zeros_like(angle),
    ], axis=1)
    launch_rate = flight_pitch_rate * launch_elapsed / rotation_launch_time
    landing_rate = flight_pitch_rate * (
        1.0 - landing_elapsed / rotation_landing_time
    )
    pitch_rate = jnp.where(
        (node_times >= launch_start_time) & (node_times <= takeoff_time),
        launch_rate,
        0.0,
    )
    pitch_rate = jnp.where(
        (node_times > takeoff_time) & (node_times <= contact_time),
        flight_pitch_rate,
        pitch_rate,
    )
    pitch_rate = jnp.where(
        (node_times > contact_time) & (node_times < rotation_stop_time),
        landing_rate,
        pitch_rate,
    )
    omega_ref = jnp.zeros((config.N + 1, 3)).at[:, 1].set(pitch_rate)

    dq_ref = jnp.zeros((config.N + 1, config.n_joints))
    dq_ref = dq_ref.at[1:-1].set(
        (q_ref[2:] - q_ref[:-2])
        / (node_times[2:, None] - node_times[:-2, None])
    )

    contact = jnp.ones((config.N + 1, config.n_contact))
    contact = contact.at[launch_end:prelanding_end].set(0.0)
    foot_ref = jnp.tile(jnp.asarray(config.p_legs0), (config.N + 1, 1))
    grf_ref = jnp.zeros((config.N + 1, 3 * config.n_contact))
    grf_ref = grf_ref.at[:, 2::3].set(contact * (51.437 * 9.81 / config.n_contact))
    reference = {
        "p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref,
        "omega": omega_ref, "dq": dq_ref,
        "foot": foot_ref, "contact": contact,
        "grf": grf_ref,
    }
    return reference, jnp.concatenate([contact, foot_ref], axis=1)


def build_initial_guess(x0, reference):
    nj = config.n_joints
    X = jnp.tile(x0, (config.N + 1, 1))
    X = X.at[:, :3].set(reference["p"])
    X = X.at[:, 3:7].set(reference["quat"])
    X = X.at[:, 7 : 7 + nj].set(reference["q"])
    X = X.at[:, 7 + nj : 10 + nj].set(
        reference["dp"] / VELOCITY_SCALE[:3]
    )
    X = X.at[:, 10 + nj : 13 + nj].set(
        reference["omega"] / VELOCITY_SCALE[3:6]
    )
    X = X.at[:, 13 + nj : 13 + 2 * nj].set(
        reference["dq"] / VELOCITY_SCALE[6:]
    )
    X = X.at[:, FOOT_START:GRF_START].set(reference["foot"])
    X = X.at[:, GRF_START:GRF_STOP].set(reference["grf"] / GRF_SCALE)
    X = X.at[:, DURATION_SLICE].set(NOMINAL_DURATION_LOGITS)
    return X.at[0, :PHYSICAL_N].set(x0[:PHYSICAL_N])


def make_backflip_constraints(reference):
    nj = config.n_joints
    waypoint_positions = reference["p"][PHASE_END_STEPS]
    waypoint_quaternions = reference["quat"][PHASE_END_STEPS]
    terminal_joints = reference["q"][-1]
    cosine_limits = jnp.cos(0.5 * WAYPOINT_ANGLE_TOLERANCES)

    def constraints(x, u, t):
        durations = phase_durations(x)
        q = x[7 : 7 + nj]
        dp = x[7 + nj : 10 + nj] * VELOCITY_SCALE[:3]
        omega = x[10 + nj : 13 + nj] * VELOCITY_SCALE[3:6]
        dq = x[13 + nj : 13 + 2 * nj] * VELOCITY_SCALE[6:]
        grf = (
            x[GRF_START:GRF_STOP].reshape(config.n_contact, 3)
            * GRF_SCALE
        )
        duration_constraints = jnp.concatenate([
            MIN_DURATIONS - durations, durations - MAX_DURATIONS,
        ])
        torque_constraints = jnp.concatenate([
            u - TORQUE_LIMIT, -TORQUE_LIMIT - u,
        ])
        joint_constraints = jnp.concatenate([
            q - JOINT_MAX, JOINT_MIN - q, jnp.abs(dq) - MAX_JOINT_SPEED,
        ])

        active = t == PHASE_END_STEPS
        position_error = x[:3][None, :] - waypoint_positions
        position_constraints = jnp.concatenate([
            position_error - WAYPOINT_POSITION_HALF_WIDTHS,
            -position_error - WAYPOINT_POSITION_HALF_WIDTHS,
        ], axis=1)
        position_constraints = jnp.where(
            active[:, None], position_constraints,
            -jnp.ones_like(position_constraints),
        )
        normalized_quat = safe_normalize_quaternion(x[3:7])
        alignment = jnp.abs(waypoint_quaternions @ normalized_quat)
        orientation_constraints = jnp.where(
            active, cosine_limits - alignment, -jnp.ones_like(alignment)
        )
        rate_constraints = jnp.concatenate([
            jnp.abs(dp)[None, :] - WAYPOINT_LINEAR_SPEED_LIMITS[:, None],
            jnp.abs(omega)[None, :] - WAYPOINT_ANGULAR_SPEED_LIMITS[:, None],
        ], axis=1)
        rate_constraints = jnp.where(
            active[:, None], rate_constraints, -jnp.ones_like(rate_constraints)
        )
        # Disambiguate a full flip from rotating to pi and reversing.
        rotation_direction = jnp.where(
            t == PHASE_END_STEPS[3], omega[1] + 4.0, -1.0
        ).reshape(1)

        contact = reference["contact"][t]
        tangent_force = (
            jnp.sqrt(jnp.sum(grf[:, :2] ** 2, axis=1)
                     + TANGENTIAL_FORCE_SMOOTHING**2)
            - TANGENTIAL_FORCE_SMOOTHING
        )
        normal_force = grf[:, 2]
        force_constraints = jnp.stack([
            -normal_force,
            normal_force - MAX_NORMAL_FORCE,
            tangent_force - FRICTION_COEFFICIENT * normal_force,
        ], axis=1)
        force_constraints = jnp.where(
            contact[:, None] > 0.5, force_constraints,
            -jnp.ones_like(force_constraints),
        )
        final_constraints = jnp.concatenate([
            jnp.abs(dp) - FINAL_LINEAR_SPEED,
            jnp.abs(omega) - FINAL_ANGULAR_SPEED,
            jnp.abs(q - terminal_joints) - FINAL_JOINT_POSITION_TOLERANCE,
            jnp.abs(dq) - FINAL_JOINT_SPEED,
        ])
        final_constraints = jnp.where(
            t == config.N, final_constraints, -jnp.ones_like(final_constraints)
        )
        return jnp.concatenate([
            duration_constraints, torque_constraints, joint_constraints,
            position_constraints.reshape(-1), orientation_constraints,
            rate_constraints.reshape(-1), rotation_direction,
            force_constraints.reshape(-1), final_constraints,
            jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
        ])

    return constraints


def make_phase_scaled_disturbance(n, magnitude=0.02):
    def disturbance_at_state(x, t):
        phase = phase_index(t)
        dt = phase_durations(x)[phase] / SEGMENT_LENGTHS[phase]
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
    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.n = PHYSICAL_N + NUM_PHASES
    config.nu = config.n_joints
    config.m = config.nu
    config.initial_state = jnp.concatenate([base_state, NOMINAL_DURATION_LOGITS])
    config.dynamics = min_time_backflip_dynamics
    config.cost = min_time_backflip_cost
    config.hessian_approx = None
    config.reference_generator = unused_reference_generator
    config.max_torque = TORQUE_LIMIT
    config.min_torque = -TORQUE_LIMIT
    dtype = config.initial_state.dtype
    config.W = {
        "pos": jnp.diag(jnp.array([80.0, 150.0, 100.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([120.0, 800.0, 120.0], dtype=dtype)),
        "q": jnp.eye(config.n_joints, dtype=dtype) * 8.0,
        "vel": jnp.diag(jnp.array([2.0, 3.0, 2.0], dtype=dtype)),
        "omega": jnp.diag(jnp.array([2.0, 15.0, 2.0], dtype=dtype)),
        "dq": jnp.eye(config.n_joints, dtype=dtype) * 2.0e-1,
        "contact": jnp.eye(3 * config.n_contact, dtype=dtype) * 80.0,
        "tau": jnp.eye(config.n_joints, dtype=dtype) * 2.0e-3,
        "grf": jnp.eye(3 * config.n_contact, dtype=dtype) * 1.0e-4,
        "time": jnp.ones(NUM_PHASES, dtype=dtype) * TIME_WEIGHT,
    }


def solve_backflip(mpc, data, x0, reference, parameter):
    # This is a single direct minimum-time optimization.  The analytic motion
    # reference below is only an initial iterate, not an optimized warm start.
    X_guess = build_initial_guess(x0, reference)
    nj = config.n_joints
    q_guess = X_guess[:-1, 7 : 7 + nj]
    dq_guess = (
        X_guess[:-1, 13 + nj : 13 + 2 * nj] * VELOCITY_SCALE[6:]
    )
    U_guess = (
        INITIAL_KP * (reference["q"][1:] - q_guess)
        + INITIAL_KD * (reference["dq"][1:] - dq_guess)
    )
    U_guess = jnp.clip(U_guess, -TORQUE_LIMIT, TORQUE_LIMIT)
    predicted_next = jax.vmap(
        lambda x, u, t: mpc.dynamics(x, u, t, parameter)
    )(X_guess[:-1], U_guess, jnp.arange(config.N, dtype=jnp.int32))
    # Feet and GRFs are algebraic outputs stored in the legacy whole-body state.
    # Initialize those entries consistently without replacing physical states.
    X_guess = X_guess.at[1:, FOOT_START:GRF_STOP].set(
        predicted_next[:, FOOT_START:GRF_STOP]
    )
    return mpc._solve(
        reference, parameter, data.W, x0, X_guess, U_guess, data.V0,
        data.w, data.y, data.rho, data.rho_grad, mpc.obstacles,
        data.h_ct_ws, data.beta_ws, data.mu_ws, data.Phi_x_ws,
        data.Phi_u_ws, data.Phi_x_I_ws, data.Phi_u_I_ws,
        data.a, data.b, data.converged_admm,
    )


def phase_node_times(phase_times):
    dts = np.asarray(phase_times) / np.asarray(SEGMENT_LENGTHS)
    phases = np.searchsorted(
        np.asarray(PHASE_END_STEPS), np.arange(config.N), side="right"
    )
    return np.concatenate([[0.0], np.cumsum(dts[phases])])


def evaluate_constraint_violation(constraints, X, U):
    controls = jnp.concatenate([U, jnp.zeros((1, U.shape[1]))], axis=0)
    values = jax.vmap(constraints)(
        X, controls, jnp.arange(X.shape[0], dtype=jnp.int32)
    )
    positive = jnp.maximum(values, 0.0)
    flat_index = jnp.argmax(positive)
    return (
        float(jnp.max(positive)),
        int(flat_index // positive.shape[1]),
        int(flat_index % positive.shape[1]),
    )


def evaluate_dynamics_defect(dynamics, X, U, parameter):
    predicted = jax.vmap(lambda x, u, t: dynamics(x, u, t, parameter))(
        X[:-1], U, jnp.arange(U.shape[0], dtype=jnp.int32)
    )
    defect = jnp.abs(predicted - X[1:])
    flat_index = jnp.argmax(defect)
    return (
        float(jnp.max(defect)),
        int(flat_index // defect.shape[1]),
        int(flat_index % defect.shape[1]),
    )


def waypoint_diagnostics(reference, X):
    states = np.asarray(X)
    indices = np.asarray(PHASE_END_STEPS)
    positions = np.asarray(reference["p"])[indices]
    quaternions = np.asarray(reference["quat"])[indices].copy()
    actual = states[indices, 3:7].copy()
    actual /= np.linalg.norm(actual, axis=1, keepdims=True)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    alignment = np.abs(np.sum(actual * quaternions, axis=1))
    angles = 2.0 * np.arccos(np.clip(alignment, -1.0, 1.0))
    return np.abs(states[indices, :3] - positions), np.degrees(angles)


def dry_run(reference, parameter, constraints):
    x0 = jnp.asarray(config.initial_state)
    X0 = build_initial_guess(x0, reference)
    U0 = jnp.tile(config.u_ref, (config.N, 1))
    constraint_shape = jax.eval_shape(
        constraints, x0, U0[0], jnp.asarray(0)
    ).shape
    print("Humanoid backflip minimum-time setup is valid.")
    print(f"State/control dimensions: {config.n}/{config.nu}")
    print(f"Reference/parameter shapes: {reference['p'].shape}/{parameter.shape}")
    print(f"Initial guess shape: {X0.shape}")
    print(f"Constraint count per node: {constraint_shape[0]}")
    print("Optimization stages: 1 (direct minimum time; no optimized warm start)")
    print("Nominal phase times:")
    for name, duration, intervals in zip(
        PHASE_NAMES, NOMINAL_DURATIONS, SEGMENT_LENGTHS
    ):
        print(f"  {name:16s} {float(duration):.3f} s ({int(intervals)} intervals)")


def main(*, dry_run_only=False, output_dir=DIR_PATH):
    configure_problem()
    reference, parameter = build_backflip_reference()
    constraints = make_backflip_constraints(reference)
    if dry_run_only:
        dry_run(reference, parameter, constraints)
        return

    admm_config = ADMMConfig(
        eps_abs=5.0e-2, eps_rel=1.0e-3, rho_max=1.0e4,
        max_iterations=1500, rho_update_frequency=25, initial_rho=1.0,
        regularized_rho_update=False, num_phases=NUM_PHASES,
    )
    sls_config = SLSConfig(
        max_sls_iterations=1, sls_primal_tol=1.0e-2,
        enable_fastsls=False, initialize_nominal=False,
        max_initial_sqp_iterations=0, warm_start=False, rti=False,
        gradient_window=0,
    )
    sqp_config = SQPConfig(
        # Reusing ADMM duals between SQP linearizations is internal to this
        # single solve; no previously optimized trajectory is supplied.
        max_sqp_iterations=150, warm_start=True, feas_tol=1.0e-5,
        step_tol=1.0e-5, line_search=True, lm_regularization=1.0e-2,
    )
    mpc = mpc_wrapper.MPCWrapper(
        config, sls_config=sls_config, sqp_config=sqp_config,
        admm_config=admm_config, constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=config.initial_state.dtype),
        disturbance=make_phase_scaled_disturbance(config.n),
    )
    model_data = mujoco.MjData(mpc.model)
    model_data.qpos[:] = np.asarray(
        jnp.concatenate([config.p0, config.quat0, config.q0])
    )
    model_data.qvel[:] = 0.0
    mujoco.mj_forward(mpc.model, model_data)
    feet = jnp.asarray(sim_utils.geom_positions(model_data, mpc.contact_id_mj))
    x0 = (
        jnp.asarray(config.initial_state)
        .at[mpc.qpos_slice].set(jnp.asarray(model_data.qpos))
        .at[mpc.qvel_slice].set(
            jnp.asarray(model_data.qvel) / VELOCITY_SCALE
        )
        .at[mpc.foot_slice].set(feet)
        .at[DURATION_SLICE].set(NOMINAL_DURATION_LOGITS)
    )
    data = mpc.make_data()
    start = timer()
    result = solve_backflip(mpc, data, x0, reference, parameter)
    X, U = result[:2]
    converged_admm = result[-1]
    X.block_until_ready()
    solve_time = timer() - start
    if not bool(jnp.all(jnp.isfinite(X))) or not bool(jnp.all(jnp.isfinite(U))):
        raise FloatingPointError(
            "Backflip solver returned a non-finite trajectory; result not saved."
        )
    phase_times = np.asarray(phase_durations(X[0]))
    max_violation, violation_node, violation_index = (
        evaluate_constraint_violation(constraints, X, U)
    )
    max_defect, defect_node, defect_state = evaluate_dynamics_defect(
        mpc.dynamics, X, U, parameter
    )
    position_errors, angle_errors = waypoint_diagnostics(reference, X)
    print(f"Solve time: {solve_time:.3f} s")
    print("Optimized phase times:")
    for name, duration in zip(PHASE_NAMES, phase_times):
        print(f"  {name:16s} {duration:.6f} s")
    print(f"Computed minimum backflip time: {np.sum(phase_times):.6f} s")
    print(f"ADMM converged: {bool(np.asarray(converged_admm))}")
    print(
        f"Maximum constraint violation: {max_violation:.6e} at node "
        f"{violation_node}, constraint {violation_index}"
    )
    print(
        f"Maximum dynamics defect: {max_defect:.6e} at transition "
        f"{defect_node}, state {defect_state}"
    )
    print(f"Dynamics feasible: {max_defect <= DYNAMICS_TOLERANCE}")
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
    result_path = output_dir / "humanoid_backflip_min_time.npz"
    X_output = np.asarray(X).copy()
    duration_logits = X_output[:, DURATION_SLICE].copy()
    X_output[:, QVEL_START:QVEL_STOP] *= np.asarray(VELOCITY_SCALE)
    X_output[:, GRF_START:GRF_STOP] *= GRF_SCALE
    X_output[:, DURATION_SLICE] = np.asarray(
        jax.vmap(phase_durations)(X)
    )
    np.savez(
        result_path, X=X_output, U=np.asarray(U),
        phase_names=np.asarray(PHASE_NAMES),
        phase_end_steps=np.asarray(PHASE_END_STEPS), phase_times=phase_times,
        node_times=phase_node_times(phase_times),
        max_constraint_violation=np.asarray(max_violation),
        max_dynamics_defect=np.asarray(max_defect),
        waypoint_position_errors=position_errors,
        waypoint_orientation_errors_deg=angle_errors,
        phase_duration_logits=duration_logits,
        dynamics_feasible=np.asarray(max_defect <= DYNAMICS_TOLERANCE),
        optimization_stages=np.asarray(1),
        direct_min_time=np.asarray(True),
        warm_start_used=np.asarray(False),
        internal_state_scaling=np.asarray(True),
        velocity_scale=np.asarray(VELOCITY_SCALE),
        grf_scale=np.asarray(GRF_SCALE),
        reference_version=np.asarray(3),
    )
    print(f"Saved trajectory: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate dimensions and callbacks without solving.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DIR_PATH,
        help="Directory for the NPZ result.",
    )
    args = parser.parse_args()
    main(dry_run_only=args.dry_run, output_dir=args.output_dir)
