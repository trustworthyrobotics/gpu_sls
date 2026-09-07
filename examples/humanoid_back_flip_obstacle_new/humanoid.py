"""Direct minimum-time Unitree H1 backflip through a two-box corridor.

This version deliberately uses ONE optimization call.  There is no feasibility
solve, continuation, time ramp, compression pass, or final polishing solve.
Instead, the reference, hard waypoints, duration bounds, and initial guess are
constructed so the full minimum-time objective is active from SQP iteration 1.

Use ``--dry-run`` to validate dimensions and callbacks without compiling the
full solver.
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

compilation_cache_enabled = False
compilation_cache_enabled = os.environ.get(
    "JAX_ENABLE_COMPILATION_CACHE", "true"
).lower() not in {"0", "false", "no", "off"}
if compilation_cache_enabled:
    cache_dir = Path.home() / ".cache" / "jax_gpu_sls"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

CONTACT_POSITION_GAIN = 200.0
CONTACT_VELOCITY_GAIN = 30.0

PHASE_NAMES = (
    "crouch", "launch", "flight_early", "flight_late", "landing", "settle",
)

# Six phases are enough to express the task.  Removing the stand/prelanding/
# touchdown micro-phases reduces the number of optimized timing variables and
# avoids forcing the direct solve through a chain of tightly coupled phase-end
# equalities.
NOMINAL_DURATIONS = jnp.array([0.18, 0.16, 0.26, 0.26, 0.16, 0.20])
PHASE_END_STEPS = jnp.array([8, 16, 27, 38, 44, 50], dtype=jnp.int32)
SEGMENT_LENGTHS = jnp.concatenate(
    [PHASE_END_STEPS[:1], PHASE_END_STEPS[1:] - PHASE_END_STEPS[:-1]]
)
NUM_PHASES = len(PHASE_NAMES)
if config.N != int(PHASE_END_STEPS[-1]):
    raise ValueError(f"Phase layout requires config.N=50; got {config.N}.")

# Direct minimum-time needs meaningful lower bounds.  Starting from 15% of the
# nominal durations made the first SQP model strongly prefer collapsing time
# before it had corrected the state/control defects.
MIN_DURATIONS = 0.3 * NOMINAL_DURATIONS
MAX_DURATIONS = 1.75 * NOMINAL_DURATIONS
TIME_WEIGHT = 5e4
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
DURATION_SLICE = slice(PHYSICAL_N, PHYSICAL_N + NUM_PHASES)
FOOT_START = 13 + 2 * config.n_joints
GRF_START = FOOT_START + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact
QVEL_START = 7 + config.n_joints
QVEL_STOP = 13 + 2 * config.n_joints

# Keep velocity and force coordinates near unit scale for SQP/ADMM.
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

# Hard waypoints are deliberately sparse.  The running tracking objective
# shapes the rest of the backflip, while hard constraints only encode the
# topology of the motion: upside-down in flight, translated landing, and a
# settled terminal pose.
WAYPOINT_POSITION_ACTIVE = jnp.array([False, False, True, False, True, True])
WAYPOINT_ORIENTATION_ACTIVE = jnp.array([False, False, True, False, False, True])
WAYPOINT_RATE_ACTIVE = jnp.array([False, False, False, False, True, True])

WAYPOINT_POSITION_HALF_WIDTHS = jnp.array([
    [0.25, 0.15, 0.20],
    [0.25, 0.15, 0.20],
    [0.18, 0.10, 0.22],
    [0.25, 0.15, 0.25],
    [0.12, 0.08, 0.14],
    [0.08, 0.06, 0.08],
])
WAYPOINT_ANGLE_TOLERANCES = jnp.deg2rad(
    jnp.array([45.0, 45.0, 32.0, 45.0, 30.0, 8.0])
)
WAYPOINT_LINEAR_SPEED_LIMITS = jnp.array([
    4.0, 4.0, 4.0, 4.0, 1.4, 0.35
])
WAYPOINT_ANGULAR_SPEED_LIMITS = jnp.array([
    30.0, 30.0, 30.0, 30.0, 8.0, 0.8
])

FINAL_LINEAR_SPEED = 0.35
FINAL_ANGULAR_SPEED = 0.8
FINAL_ORIENTATION_TOLERANCE = jnp.deg2rad(5.0)
FINAL_JOINT_SPEED = 2.0
FINAL_JOINT_POSITION_TOLERANCE = 0.25
MAX_JOINT_SPEED = 50.0
MIN_BASE_HEIGHT = 0.43
FRICTION_COEFFICIENT = 0.7
MAX_NORMAL_FORCE = 900.0
TANGENTIAL_FORCE_SMOOTHING = 1.0e-4
CONTACT_SOLVE_REGULARIZATION = 1.0e-3
QUATERNION_EPSILON = 1.0e-8
QUATERNION_NORM_WEIGHT = 5.0e3
FINAL_TERMINAL_ORIENTATION_WEIGHT = 7.5e4

# Joint feedback is only used to construct the single solve's initial controls.
INITIAL_KP = jnp.array([100.0] * 10 + [70.0] + [30.0] * 8)
INITIAL_KD = jnp.array([5.0] * 10 + [4.0] + [1.5] * 8)
INITIAL_DEFECT_REPAIR_SWEEPS = 3
INITIAL_DEFECT_REPAIR_BLEND = 0.20

# The H1 faces +x, so a backflip travels in -x.  Two thin boxes at the crossing
# plane form a vertical corridor: a ground hurdle and a suspended obstacle.
# Sizes are full MuJoCo box dimensions.
BACKWARD_DISPLACEMENT = -0.80
LOWER_OBSTACLE_SIZE = jnp.array([0.08, 0.80, 0.22])
LOWER_OBSTACLE_CENTER = jnp.array([
    0.5 * BACKWARD_DISPLACEMENT, 0.0, 0.5 * LOWER_OBSTACLE_SIZE[2]
])
UPPER_OBSTACLE_SIZE = jnp.array([0.08, 0.80, 0.40])
UPPER_OBSTACLE_CENTER = jnp.array([
    0.5 * BACKWARD_DISPLACEMENT, 0.0, 2.55
])
OBSTACLE_CENTERS = jnp.stack([
    LOWER_OBSTACLE_CENTER, UPPER_OBSTACLE_CENTER
])
OBSTACLE_SIZES = jnp.stack([
    LOWER_OBSTACLE_SIZE, UPPER_OBSTACLE_SIZE
])
# Singular aliases retain compatibility with older result readers.
OBSTACLE_CENTER = LOWER_OBSTACLE_CENTER
OBSTACLE_SIZE = LOWER_OBSTACLE_SIZE
BASE_CLEARANCE_RADIUS = 0.20
JOINT_CLEARANCE_RADIUS = 0.055
FOOT_CLEARANCE_RADIUS = 0.040
HEAD_CLEARANCE_RADIUS = 0.12
HEAD_GEOM_LOCAL_Z = 0.42
SUPERELLIPSOID_POWER = 4
SUPERELLIPSOID_SCALE = 2.0 ** (1.0 / SUPERELLIPSOID_POWER)
OBSTACLE_TOLERANCE = 5.0e-3
FEASIBILITY_TOLERANCE = 5.0e-2
DYNAMICS_TOLERANCE = 3.0e-1


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

        # Velocity error of active contact points.
        velocity_violation = active_jacobian.T @ qvel

        # Desired world-frame foot locations for active contacts.
        target_feet = parameter[
            t,
            config.n_contact:
            config.n_contact + 3 * config.n_contact
        ]

        # Position error only matters for feet that are supposed to be in contact.
        position_violation = contact_mask * (
            current_feet - target_feet
        )
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
            @ jax.scipy.linalg.cho_solve(
                (mass_factor, False),
                tau - bias_force,
            )
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
            velocity_next / VELOCITY_SCALE, current_feet, grf / GRF_SCALE,
        ])
        return jnp.concatenate([physical_next, x[DURATION_SLICE]])

    return dynamics


def min_time_backflip_cost(W, reference, x, u, t, *, parameter):
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
    terminal_orientation_cost = jnp.where(
        t == config.N,
        W["terminal_orientation"] * jnp.dot(qerr, qerr),
        jnp.asarray(0.0, dtype=x.dtype),
    )
    time_cost = jnp.where(
        t == config.N, jnp.dot(W["time"], phase_durations(x)),
        jnp.asarray(0.0, dtype=x.dtype),
    )
    return 0.5 * (stage_cost + terminal_orientation_cost) + time_cost


def _blend_pose(nodes, start, stop, q_start, q_stop):
    alpha = jnp.clip((nodes - start) / max(stop - start, 1), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return q_start + alpha[:, None] * (q_stop - q_start)


def build_backflip_reference():
    """Construct a smooth six-phase reference for a direct minimum-time solve."""
    nodes = jnp.arange(config.N + 1, dtype=jnp.float32)
    q0 = jnp.asarray(config.q0)

    # Symmetric leg shapes avoid injecting unnecessary roll/yaw momentum.
    crouch_q = q0.at[2].set(-0.95).at[3].set(1.75).at[4].set(-0.75)
    crouch_q = crouch_q.at[7].set(-0.95).at[8].set(1.75).at[9].set(-0.75)
    crouch_q = crouch_q.at[11].set(1.10).at[15].set(1.10)

    launch_q = q0.at[2].set(-0.20).at[3].set(0.40).at[4].set(-0.25)
    launch_q = launch_q.at[7].set(-0.20).at[8].set(0.40).at[9].set(-0.25)
    launch_q = launch_q.at[11].set(-1.55).at[15].set(-1.55)

    tuck_q = q0.at[2].set(-1.30).at[3].set(1.90).at[4].set(-0.50)
    tuck_q = tuck_q.at[7].set(-1.30).at[8].set(1.90).at[9].set(-0.50)
    tuck_q = tuck_q.at[14].set(1.55).at[18].set(1.55)

    open_q = q0.at[2].set(-0.55).at[3].set(1.20).at[4].set(-0.65)
    open_q = open_q.at[7].set(-0.55).at[8].set(1.20).at[9].set(-0.65)
    open_q = open_q.at[11].set(0.55).at[15].set(0.55)

    landing_q = q0.at[2].set(-0.65).at[3].set(1.35).at[4].set(-0.70)
    landing_q = landing_q.at[7].set(-0.65).at[8].set(1.35).at[9].set(-0.70)

    # Pose schedule: q0 -> crouch -> launch -> tuck -> open -> landing -> q0.
    q_ref = jnp.tile(q0, (config.N + 1, 1))
    pose_segments = (
        (0, 8, q0, crouch_q),
        (8, 16, crouch_q, launch_q),
        (16, 27, launch_q, tuck_q),
        (27, 38, tuck_q, open_q),
        (38, 44, open_q, landing_q),
        (44, 50, landing_q, q0),
    )
    for start, stop, q_start, q_stop in pose_segments:
        values = _blend_pose(nodes, start, stop, q_start, q_stop)
        q_ref = jnp.where(
            ((nodes >= start) & (nodes <= stop))[:, None], values, q_ref
        )

    phase_ids = phase_index(jnp.arange(config.N))
    node_times = jnp.concatenate([
        jnp.asarray([0.0]),
        jnp.cumsum(NOMINAL_DURATIONS[phase_ids] / SEGMENT_LENGTHS[phase_ids]),
    ])

    crouch_end, takeoff, midflight, landing_start, touchdown, settle = (
        int(v) for v in PHASE_END_STEPS
    )
    del midflight, settle

    # Vertical COM reference: shallow crouch, smooth launch, then a ballistic arc.
    standing_height = float(config.p0[2])
    launch_duration = float(NOMINAL_DURATIONS[1])
    airborne_duration = float(NOMINAL_DURATIONS[2] + NOMINAL_DURATIONS[3])
    takeoff_speed = 0.5 * 9.81 * airborne_duration
    crouch_height = standing_height - 0.14

    p_ref = jnp.zeros((config.N + 1, 3)).at[:, 2].set(standing_height)
    alpha = jnp.clip(nodes / crouch_end, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    p_ref = p_ref.at[:, 2].set(
        jnp.where(nodes <= crouch_end,
                  standing_height + alpha * (crouch_height - standing_height),
                  p_ref[:, 2])
    )

    launch_elapsed = jnp.clip(
        node_times - node_times[crouch_end], 0.0, launch_duration
    )
    launch_s = launch_elapsed / launch_duration
    h00 = 2.0 * launch_s**3 - 3.0 * launch_s**2 + 1.0
    h10 = launch_s**3 - 2.0 * launch_s**2 + launch_s
    h01 = -2.0 * launch_s**3 + 3.0 * launch_s**2
    h11 = launch_s**3 - launch_s**2
    launch_z = (
        h00 * crouch_height
        + h10 * 0.0
        + h01 * standing_height
        + h11 * (takeoff_speed * launch_duration)
    )
    p_ref = p_ref.at[:, 2].set(
        jnp.where((nodes > crouch_end) & (nodes <= takeoff), launch_z, p_ref[:, 2])
    )

    air_elapsed = jnp.clip(
        node_times - node_times[takeoff], 0.0, airborne_duration
    )
    air_z = standing_height + takeoff_speed * air_elapsed - 0.5 * 9.81 * air_elapsed**2
    p_ref = p_ref.at[:, 2].set(
        jnp.where((nodes > takeoff) & (nodes <= landing_start), air_z, p_ref[:, 2])
    )

    # Horizontal reference is physically interpretable: accelerate during launch,
    # coast through flight, and brake during landing.
    launch_time_total = node_times[takeoff] - node_times[crouch_end]
    flight_time_total = node_times[landing_start] - node_times[takeoff]
    landing_time_total = node_times[touchdown] - node_times[landing_start]
    backward_speed = BACKWARD_DISPLACEMENT / (
        0.5 * launch_time_total + flight_time_total + 0.5 * landing_time_total
    )

    launch_t = jnp.clip(node_times - node_times[crouch_end], 0.0, launch_time_total)
    launch_x = 0.5 * backward_speed / launch_time_total * launch_t**2
    takeoff_x = 0.5 * backward_speed * launch_time_total
    flight_t = jnp.clip(node_times - node_times[takeoff], 0.0, flight_time_total)
    flight_x = takeoff_x + backward_speed * flight_t
    landing_x0 = takeoff_x + backward_speed * flight_time_total
    landing_t = jnp.clip(
        node_times - node_times[landing_start], 0.0, landing_time_total
    )
    landing_x = (
        landing_x0 + backward_speed * landing_t
        - 0.5 * backward_speed / landing_time_total * landing_t**2
    )
    x_ref = jnp.where(node_times <= node_times[takeoff], launch_x, flight_x)
    x_ref = jnp.where(node_times > node_times[landing_start], landing_x, x_ref)
    x_ref = jnp.where(node_times >= node_times[touchdown], BACKWARD_DISPLACEMENT, x_ref)
    p_ref = p_ref.at[:, 0].set(x_ref)

    # One continuous negative-pitch rotation.  Requiring the mid-flight state to
    # be near pi plus a negative pitch rate removes the quaternion 2*pi ambiguity.
    angle = jnp.zeros(config.N + 1)
    launch_angle = -0.20
    alpha = jnp.clip((nodes - crouch_end) / (takeoff - crouch_end), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    angle = jnp.where(
        (nodes > crouch_end) & (nodes <= takeoff), launch_angle * alpha, angle
    )
    alpha = jnp.clip((nodes - takeoff) / (landing_start - takeoff), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    angle = jnp.where(
        (nodes > takeoff) & (nodes <= landing_start),
        launch_angle + (-2.0 * jnp.pi - launch_angle) * alpha,
        angle,
    )
    angle = jnp.where(nodes > landing_start, -2.0 * jnp.pi, angle)
    quat_ref = jnp.stack([
        jnp.cos(0.5 * angle), jnp.zeros_like(angle),
        jnp.sin(0.5 * angle), jnp.zeros_like(angle),
    ], axis=1)

    dp_ref = jnp.zeros((config.N + 1, 3))
    dp_ref = dp_ref.at[1:-1].set(
        (p_ref[2:] - p_ref[:-2]) / (node_times[2:, None] - node_times[:-2, None])
    )
    dp_ref = dp_ref.at[0].set(
        (p_ref[1] - p_ref[0]) / (node_times[1] - node_times[0])
    )
    dp_ref = dp_ref.at[-1].set(jnp.zeros(3))

    omega_ref = jnp.zeros((config.N + 1, 3))
    omega_ref = omega_ref.at[1:-1, 1].set(
        (angle[2:] - angle[:-2]) / (node_times[2:] - node_times[:-2])
    )

    dq_ref = jnp.zeros((config.N + 1, config.n_joints))
    dq_ref = dq_ref.at[1:-1].set(
        (q_ref[2:] - q_ref[:-2]) / (node_times[2:, None] - node_times[:-2, None])
    )

    contact = jnp.ones((config.N + 1, config.n_contact))
    contact = contact.at[takeoff:landing_start].set(0.0)
    contact = contact.at[landing_start:touchdown].set(
        jnp.array([1.0, 0.0, 1.0, 0.0])
    )

    foot_ref = jnp.tile(jnp.asarray(config.p_legs0), (config.N + 1, 1))
    landing_foot_offset = jnp.tile(
        jnp.array([BACKWARD_DISPLACEMENT, 0.0, 0.0]), config.n_contact
    )
    foot_ref = foot_ref + jnp.where(
        (nodes >= landing_start)[:, None], landing_foot_offset[None, :], 0.0
    )

    grf_ref = jnp.zeros((config.N + 1, 3 * config.n_contact))
    grf_ref = grf_ref.at[:, 2::3].set(
        contact * (51.437 * 9.81 / config.n_contact)
    )

    reference = {
        "p": p_ref, "quat": quat_ref, "q": q_ref, "dp": dp_ref,
        "omega": omega_ref, "dq": dq_ref, "foot": foot_ref,
        "contact": contact, "grf": grf_ref,
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


def make_obstacle_clearance_constraints():
    """Build differentiable corridor constraints for the entire humanoid.

    Each box's x-z rectangle is inflated by a clearance radius around the
    pelvis, virtual head, every articulated joint, and every foot.  The H1
    asset has no separate head body, so the virtual head is the uppermost point
    of its main torso mesh.  Since both boxes span the humanoid in y, treating
    them as infinite along y is conservative.
    """

    model = mujoco.MjModel.from_xml_path(config.model_path)
    mjx_model = mjx.put_model(model)
    joint_ids = jnp.asarray(
        [
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ],
        dtype=jnp.int32,
    )
    torso_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    torso_geom_ids = np.flatnonzero(model.geom_bodyid == torso_body_id)
    if torso_body_id < 0 or torso_geom_ids.size == 0:
        raise ValueError("Could not locate the H1 torso geometry.")
    head_geom_id = int(
        torso_geom_ids[np.argmax(model.geom_aabb[torso_geom_ids, 5])]
    )
    head_local_offset = jnp.array([0.0, 0.0, HEAD_GEOM_LOCAL_Z])
    nj = config.n_joints
    feet_start = 13 + 2 * nj
    obstacle_xz = jnp.array([0, 2])

    def inflated_superellipse(points, clearance):
        radii = (
            0.5 * OBSTACLE_SIZES[:, obstacle_xz] + clearance
        ) * SUPERELLIPSOID_SCALE
        normalized = (
            points[:, obstacle_xz][None, :, :]
            - OBSTACLE_CENTERS[:, obstacle_xz][:, None, :]
        ) / radii[:, None, :]
        radial_power = jnp.sum(normalized**SUPERELLIPSOID_POWER, axis=-1)
        # Bounded residual with the same zero boundary and feasible set as
        # 1 - radial_power, avoiding extreme SQP Jacobians far from the box.
        return (2.0 / (1.0 + radial_power) - 1.0).reshape(-1)

    def constraints(x):
        qpos = x[: nj + 7].at[3:7].set(
            safe_normalize_quaternion(x[3:7])
        )
        kinematics = mjx.make_data(model).replace(qpos=qpos)
        kinematics = mjx.fwd_position(mjx_model, kinematics)
        joint_positions = kinematics.xanchor[joint_ids]
        head_position = (
            kinematics.geom_xpos[head_geom_id]
            + kinematics.geom_xmat[head_geom_id] @ head_local_offset
        )
        feet = x[feet_start : feet_start + 3 * config.n_contact].reshape(
            config.n_contact, 3
        )
        return jnp.concatenate([
            inflated_superellipse(x[:3][None, :], BASE_CLEARANCE_RADIUS),
            inflated_superellipse(
                head_position[None, :], HEAD_CLEARANCE_RADIUS
            ),
            inflated_superellipse(joint_positions, JOINT_CLEARANCE_RADIUS),
            inflated_superellipse(feet, FOOT_CLEARANCE_RADIUS),
        ])

    obstacle_count = int(OBSTACLE_CENTERS.shape[0])
    base_stop = obstacle_count
    head_stop = base_stop + obstacle_count
    joint_stop = head_stop + obstacle_count * len(joint_ids)
    constraints.base_slice = slice(0, base_stop)
    constraints.head_slice = slice(base_stop, head_stop)
    constraints.joint_slice = slice(head_stop, joint_stop)
    constraints.foot_slice = slice(joint_stop, None)
    constraints.joint_count = len(joint_ids)
    constraints.obstacle_count = obstacle_count
    constraints.count = obstacle_count * (
        2 + len(joint_ids) + config.n_contact
    )
    return constraints


def make_backflip_constraints(reference, obstacle_constraints):
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
            x[GRF_START:GRF_STOP].reshape(config.n_contact, 3) * GRF_SCALE
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

        phase_end_active = t == PHASE_END_STEPS
        position_active = phase_end_active & WAYPOINT_POSITION_ACTIVE
        orientation_active = phase_end_active & WAYPOINT_ORIENTATION_ACTIVE
        rate_active = phase_end_active & WAYPOINT_RATE_ACTIVE

        position_error = x[:3][None, :] - waypoint_positions
        position_constraints = jnp.concatenate([
            position_error - WAYPOINT_POSITION_HALF_WIDTHS,
            -position_error - WAYPOINT_POSITION_HALF_WIDTHS,
        ], axis=1)
        position_constraints = jnp.where(
            position_active[:, None], position_constraints,
            -jnp.ones_like(position_constraints),
        )

        normalized_quat = safe_normalize_quaternion(x[3:7])
        alignment = jnp.abs(waypoint_quaternions @ normalized_quat)
        orientation_constraints = jnp.where(
            orientation_active, cosine_limits - alignment,
            -jnp.ones_like(alignment),
        )

        rate_constraints = jnp.concatenate([
            jnp.abs(dp)[None, :] - WAYPOINT_LINEAR_SPEED_LIMITS[:, None],
            jnp.abs(omega)[None, :] - WAYPOINT_ANGULAR_SPEED_LIMITS[:, None],
        ], axis=1)
        rate_constraints = jnp.where(
            rate_active[:, None], rate_constraints, -jnp.ones_like(rate_constraints)
        )

        # Require continued negative pitch rotation at the upside-down waypoint.
        # This prevents a half-turn followed by reversal without hard-constraining
        # orientation at every intermediate phase boundary.
        rotation_direction = jnp.where(
            t == PHASE_END_STEPS[2], omega[1] + 2.0, -1.0
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
        terminal_orientation_error = math.quat_sub(
            normalized_quat, waypoint_quaternions[-1]
        )
        final_constraints = jnp.concatenate([
            # Scale the tight angular bound to O(1).  In raw radians, even a
            # multi-degree error is smaller than ADMM's absolute tolerance.
            # The squared norm bounds the total rotation, not just its three
            # axis-angle components independently.
            jnp.array([
                jnp.sum(terminal_orientation_error**2)
                / FINAL_ORIENTATION_TOLERANCE**2
                - 1.0
            ]),
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
            obstacle_constraints(x),
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
        "pos": jnp.diag(jnp.array([120.0, 180.0, 140.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([160.0, 1100.0, 160.0], dtype=dtype)),
        "q": jnp.eye(config.n_joints, dtype=dtype) * 10.0,
        "vel": jnp.diag(jnp.array([3.0, 4.0, 3.0], dtype=dtype)),
        "omega": jnp.diag(jnp.array([2.0, 22.0, 2.0], dtype=dtype)),
        "dq": jnp.eye(config.n_joints, dtype=dtype) * 2.0e-1,
        "contact": jnp.eye(3 * config.n_contact, dtype=dtype) * 50.0,
        "tau": jnp.eye(config.n_joints, dtype=dtype) * 2.0e-3,
        "grf": jnp.eye(3 * config.n_contact, dtype=dtype) * 1.0e-4,
        "time": jnp.ones(NUM_PHASES, dtype=dtype) * TIME_WEIGHT,
        "terminal_orientation": jnp.asarray(
            FINAL_TERMINAL_ORIENTATION_WEIGHT, dtype=dtype
        ),
    }


def build_direct_initial_guess(mpc, x0, reference, parameter):
    """Build a reference-shaped, dynamics-informed guess without solving an OCP."""
    X = build_initial_guess(x0, reference)
    nj = config.n_joints

    def controls_for(X_current):
        q = X_current[:-1, 7 : 7 + nj]
        dq = (
            X_current[:-1, 13 + nj : 13 + 2 * nj] * VELOCITY_SCALE[6:]
        )
        U = (
            INITIAL_KP * (reference["q"][1:] - q)
            + INITIAL_KD * (reference["dq"][1:] - dq)
        )
        return jnp.clip(U, -TORQUE_LIMIT, TORQUE_LIMIT)

    # A few fixed defect-repair sweeps make the starting point less hostile to
    # SQP while retaining the flip geometry.  This is initialization only: no
    # cost/constraint optimization is performed here.
    U = controls_for(X)
    for _ in range(INITIAL_DEFECT_REPAIR_SWEEPS):
        predicted = jax.vmap(
            lambda x, u, t: mpc.dynamics(x, u, t, parameter)
        )(X[:-1], U, jnp.arange(config.N, dtype=jnp.int32))

        # Blend only the physical dynamic state.  Keep phase-duration logits at
        # their nominal values so the first SQP model sees sane timesteps.
        repaired_physical = (
            (1.0 - INITIAL_DEFECT_REPAIR_BLEND) * X[1:, :PHYSICAL_N]
            + INITIAL_DEFECT_REPAIR_BLEND * predicted[:, :PHYSICAL_N]
        )
        X = X.at[1:, :PHYSICAL_N].set(repaired_physical)
        X = X.at[0, :PHYSICAL_N].set(x0[:PHYSICAL_N])

        # Feet and GRFs are algebraic outputs of the contact model; use the
        # dynamics values exactly instead of blending inconsistent references.
        X = X.at[1:, FOOT_START:GRF_STOP].set(
            predicted[:, FOOT_START:GRF_STOP]
        )
        U = controls_for(X)

    return X, U


def solve_backflip(mpc, data, x0, reference, parameter):
    """Run the full minimum-time problem exactly once."""
    X_guess, U_guess = build_direct_initial_guess(
        mpc, x0, reference, parameter
    )

    return mpc._solve(
        reference, parameter, data.W, x0, X_guess, U_guess,
        data.V0, data.w, data.y, data.rho, data.rho_grad,
        mpc.obstacles, data.h_ct_ws, data.beta_ws, data.mu_ws,
        data.Phi_x_ws, data.Phi_u_ws, data.Phi_x_I_ws, data.Phi_u_I_ws,
        data.a, data.b, data.converged_admm,
    )


def phase_node_times(phase_times):
    dts = np.asarray(phase_times) / np.asarray(SEGMENT_LENGTHS)
    phases = np.searchsorted(
        np.asarray(PHASE_END_STEPS), np.arange(config.N), side="right"
    )
    return np.concatenate([[0.0], np.cumsum(dts[phases])])


def refresh_algebraic_states(dynamics, X, U, parameter):
    """Recompute derived foot positions and GRFs from physical states."""
    predicted = jax.vmap(
        lambda x, u, t: dynamics(x, u, t, parameter)
    )(X[:-1], U, jnp.arange(U.shape[0], dtype=jnp.int32))
    return X.at[1:, FOOT_START:GRF_STOP].set(
        predicted[:, FOOT_START:GRF_STOP]
    )


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


def dry_run(reference, parameter, constraints, obstacle_constraints):
    x0 = jnp.asarray(config.initial_state)
    X0 = build_initial_guess(x0, reference)
    U0 = jnp.tile(config.u_ref, (config.N, 1))
    constraint_shape = jax.eval_shape(
        constraints, x0, U0[0], jnp.asarray(0)
    ).shape
    print("Humanoid obstacle-backflip minimum-time setup is valid.")
    print(f"State/control dimensions: {config.n}/{config.nu}")
    print(f"Reference/parameter shapes: {reference['p'].shape}/{parameter.shape}")
    print(f"Initial guess shape: {X0.shape}")
    print(f"Constraint count per node: {constraint_shape[0]}")
    print("Optimization stages: 1 (direct minimum-time solve)")
    print("Corridor obstacle centers/sizes:")
    for label, center, size in zip(
        ("lower", "upper"), OBSTACLE_CENTERS, OBSTACLE_SIZES
    ):
        print(f"  {label:5s}: {np.asarray(center)} / {np.asarray(size)} m")
    print(
        f"Clearance checks per node: {obstacle_constraints.obstacle_count} "
        "obstacles x (1 pelvis + 1 head + "
        f"{obstacle_constraints.joint_count} joint anchors + "
        f"{config.n_contact} feet) = {obstacle_constraints.count}"
    )
    print("Nominal phase times:")
    for name, duration, intervals in zip(
        PHASE_NAMES, NOMINAL_DURATIONS, SEGMENT_LENGTHS
    ):
        print(f"  {name:16s} {float(duration):.3f} s ({int(intervals)} intervals)")


def main(*, dry_run_only=False, output_dir=DIR_PATH):
    configure_problem()
    reference, parameter = build_backflip_reference()
    obstacle_constraints = make_obstacle_clearance_constraints()
    constraints = make_backflip_constraints(reference, obstacle_constraints)
    if dry_run_only:
        dry_run(reference, parameter, constraints, obstacle_constraints)
        return

    admm_config = ADMMConfig(
        eps_abs=1.0e-2, eps_rel=5.0e-3, rho_max=1.0e4,
        max_iterations=400, rho_update_frequency=25, initial_rho=1.0,
        regularized_rho_update=False, num_phases=NUM_PHASES,
        enable_slack=False,
        slack_weight=1e4,
    )
    sls_config = SLSConfig(
        max_sls_iterations=1, sls_primal_tol=1.0e-2,
        enable_fastsls=False, initialize_nominal=True,
        max_initial_sqp_iterations=100, warm_start=True, rti=False,
        gradient_window=0,
    )
    sqp_config = SQPConfig(
        max_sqp_iterations=0, warm_start=True, feas_tol=1.0e-5,
        step_tol=1.0e-5, line_search=True, lm_regularization=5.0e-2, # Just changed this 5 -> 1
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
    X = refresh_algebraic_states(mpc.dynamics, X, U, parameter)
    phase_times = np.asarray(phase_durations(X[0]))
    max_violation, violation_node, violation_index = (
        evaluate_constraint_violation(constraints, X, U)
    )
    max_defect, defect_node, defect_state = evaluate_dynamics_defect(
        mpc.dynamics, X, U, parameter
    )
    obstacle_values = jax.vmap(obstacle_constraints)(X)
    positive_obstacle_values = jnp.maximum(obstacle_values, 0.0)
    obstacle_flat_index = int(jnp.argmax(positive_obstacle_values))
    max_obstacle_violation = float(jnp.max(positive_obstacle_values))
    obstacle_node = obstacle_flat_index // obstacle_values.shape[1]
    obstacle_point = obstacle_flat_index % obstacle_values.shape[1]
    max_base_obstacle_violation = float(
        jnp.max(jnp.maximum(
            obstacle_values[:, obstacle_constraints.base_slice], 0.0
        ))
    )
    max_head_obstacle_violation = float(
        jnp.max(jnp.maximum(
            obstacle_values[:, obstacle_constraints.head_slice], 0.0
        ))
    )
    max_joint_obstacle_violation = float(
        jnp.max(jnp.maximum(
            obstacle_values[:, obstacle_constraints.joint_slice], 0.0
        ))
    )
    max_foot_obstacle_violation = float(
        jnp.max(jnp.maximum(
            obstacle_values[:, obstacle_constraints.foot_slice], 0.0
        ))
    )
    is_feasible = bool(
        np.isfinite(max_violation)
        and np.isfinite(max_defect)
        and max_violation <= FEASIBILITY_TOLERANCE
        and max_defect <= DYNAMICS_TOLERANCE
        and max_obstacle_violation <= OBSTACLE_TOLERANCE
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
    print(
        f"Maximum obstacle violation: {max_obstacle_violation:.6e} at node "
        f"{obstacle_node}, point {obstacle_point}"
    )
    print(
        "Pelvis/head/joint/foot obstacle violations: "
        f"{max_base_obstacle_violation:.6e} / "
        f"{max_head_obstacle_violation:.6e} / "
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
    result_path = output_dir / "humanoid_backflip_obstacle_min_time.npz"
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
        max_obstacle_violation=np.asarray(max_obstacle_violation),
        max_base_obstacle_violation=np.asarray(max_base_obstacle_violation),
        max_head_obstacle_violation=np.asarray(max_head_obstacle_violation),
        max_joint_obstacle_violation=np.asarray(max_joint_obstacle_violation),
        max_foot_obstacle_violation=np.asarray(max_foot_obstacle_violation),
        max_dynamics_defect=np.asarray(max_defect),
        waypoint_position_errors=position_errors,
        waypoint_orientation_errors_deg=angle_errors,
        phase_duration_logits=duration_logits,
        obstacle_center=np.asarray(OBSTACLE_CENTER),
        obstacle_size=np.asarray(OBSTACLE_SIZE),
        obstacle_centers=np.asarray(OBSTACLE_CENTERS),
        obstacle_sizes=np.asarray(OBSTACLE_SIZES),
        backward_displacement=np.asarray(BACKWARD_DISPLACEMENT),
        obstacle_point_count=np.asarray(obstacle_constraints.count),
        feasible=np.asarray(is_feasible),
        admm_converged=np.asarray(bool(np.asarray(converged_admm))),
        optimization_stages=np.asarray(1),
        direct_min_time=np.asarray(True),
        reference_version=np.asarray(8),
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
        "--dry-run", action="store_true",
        help="Validate dimensions and callbacks without solving.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DIR_PATH,
        help="Directory for the NPZ result.",
    )
    args = parser.parse_args()
    main(dry_run_only=args.dry_run, output_dir=args.output_dir)
