"""H1 backflip reference tracking with horizon-15 RTI using GPU-SLS/SQP.

This controller tracks the trajectory produced by the direct minimum-time
backflip solver (default: humanoid_backflip_obstacle_min_time.npz).

Key differences from the direct minimum-time solve:
  * MPC horizon is 15.
  * Phase durations are NOT decision variables.
  * The saved reference node times define the per-stage dt.
  * There are NO waypoint constraints and NO hard terminal-pose constraints.
  * The backflip is enforced only through the moving tracking objective.
  * One SQP step is taken per MPC update (RTI style).
  * The physical torque/joint/contact/friction/base-height/corridor constraints
    remain active.

The script performs an exact-model closed-loop rollout by applying the first
control from each horizon and shifting the primal warm start.

Example:
    python3 h1_backflip_rti_h15.py
    python3 h1_backflip_rti_h15.py --reference humanoid_backflip_obstacle_min_time.npz
    python3 h1_backflip_rti_h15.py --output humanoid_backflip_rti_h15.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
jax.config.update("jax_log_compiles", True)
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

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# -----------------------------------------------------------------------------
# RTI problem dimensions
# -----------------------------------------------------------------------------

TRACKING_N = 15

NJ = config.n_joints
NC = config.n_contact
PHYSICAL_N = 13 + 2 * NJ + 6 * NC

QPOS_STOP = 7 + NJ
QVEL_START = 7 + NJ
QVEL_STOP = 13 + 2 * NJ
FOOT_START = 13 + 2 * NJ
GRF_START = FOOT_START + 3 * NC
GRF_STOP = GRF_START + 3 * NC

# Keep exactly the same state scaling used by the direct minimum-time solve.
VELOCITY_SCALE = jnp.concatenate([
    jnp.ones(3) * 3.0,
    jnp.ones(3) * 20.0,
    jnp.ones(NJ) * 30.0,
])
GRF_SCALE = 500.0

CONTACT_POSITION_GAIN = 200.0
CONTACT_VELOCITY_GAIN = 30.0
CONTACT_SOLVE_REGULARIZATION = 1.0e-3
TANGENTIAL_FORCE_SMOOTHING = 1.0e-4
QUATERNION_EPSILON = 1.0e-8
QUATERNION_NORM_WEIGHT = 5.0e3

FRICTION_COEFFICIENT = 0.7
MAX_NORMAL_FORCE = 900.0
MAX_JOINT_SPEED = 50.0
MIN_BASE_HEIGHT = 0.43

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

# Corridor defaults used by the direct planner if the NPZ does not contain them.
DEFAULT_OBSTACLE_CENTERS = np.array([
    [-0.40, 0.0, 0.11],
    [-0.40, 0.0, 2.55],
], dtype=np.float32)
DEFAULT_OBSTACLE_SIZES = np.array([
    [0.08, 0.80, 0.22],
    [0.08, 0.80, 0.40],
], dtype=np.float32)

BASE_CLEARANCE_RADIUS = 0.20
JOINT_CLEARANCE_RADIUS = 0.055
FOOT_CLEARANCE_RADIUS = 0.040
HEAD_CLEARANCE_RADIUS = 0.12
HEAD_GEOM_LOCAL_Z = 0.42
SUPERELLIPSOID_POWER = 4
SUPERELLIPSOID_SCALE = 2.0 ** (1.0 / SUPERELLIPSOID_POWER)


def safe_normalize_quaternion(quat):
    norm_squared = jnp.sum(quat * quat)
    normalized = quat / jnp.sqrt(jnp.maximum(norm_squared, QUATERNION_EPSILON))
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype)
    return jnp.where(norm_squared > QUATERNION_EPSILON, normalized, identity)


# -----------------------------------------------------------------------------
# Reference loading / windowing
# -----------------------------------------------------------------------------

def _infer_contact_schedule(result, X_phys):
    if "contact" in result:
        contact = np.asarray(result["contact"], dtype=np.float32)
        if contact.shape == (X_phys.shape[0], NC):
            return contact

    # Preferred fallback for the six-phase direct backflip file.
    if "phase_end_steps" in result:
        ends = np.asarray(result["phase_end_steps"], dtype=int).reshape(-1)
        if ends.size >= 5:
            contact = np.ones((X_phys.shape[0], NC), dtype=np.float32)
            takeoff = int(ends[1])
            landing_start = int(ends[3])
            touchdown = int(ends[4])
            contact[takeoff:landing_start] = 0.0
            if NC == 4:
                contact[landing_start:touchdown] = np.array(
                    [1.0, 0.0, 1.0, 0.0], dtype=np.float32
                )
            return contact

    # Last-resort inference from nonzero GRF in the saved reference.
    grf = X_phys[:, GRF_START:GRF_STOP].reshape(-1, NC, 3)
    return (np.linalg.norm(grf, axis=2) > 1.0e-3).astype(np.float32)


def load_plan(filename: Path):
    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(
            f"Reference trajectory does not exist: {filename}\n"
            "Run the direct minimum-time backflip solver first."
        )

    with np.load(filename, allow_pickle=False) as result:
        if "X" not in result or "U" not in result:
            raise ValueError("Reference NPZ must contain X and U.")

        X_saved = np.asarray(result["X"], dtype=np.float32)
        U_saved = np.asarray(result["U"], dtype=np.float32)

        if X_saved.ndim != 2 or X_saved.shape[1] < PHYSICAL_N:
            raise ValueError(
                f"Expected X with at least {PHYSICAL_N} columns; got {X_saved.shape}."
            )
        if U_saved.ndim != 2 or U_saved.shape[1] != NJ:
            raise ValueError(f"Expected U shape (*, {NJ}); got {U_saved.shape}.")
        if X_saved.shape[0] != U_saved.shape[0] + 1:
            raise ValueError(
                "Reference must have one more state node than control intervals."
            )

        # Saved direct-min-time states are exported in physical qvel/GRF units.
        X_phys = X_saved[:, :PHYSICAL_N].copy()
        X_solver = X_phys.copy()
        X_solver[:, QVEL_START:QVEL_STOP] /= np.asarray(VELOCITY_SCALE)
        X_solver[:, GRF_START:GRF_STOP] /= GRF_SCALE

        # Normalize saved quaternions defensively.
        qnorm = np.linalg.norm(X_solver[:, 3:7], axis=1, keepdims=True)
        X_solver[:, 3:7] /= np.maximum(qnorm, 1.0e-8)
        X_phys[:, 3:7] = X_solver[:, 3:7]

        if "node_times" in result:
            node_times = np.asarray(result["node_times"], dtype=np.float32).reshape(-1)
        elif "phase_times" in result and "phase_end_steps" in result:
            phase_times = np.asarray(result["phase_times"], dtype=np.float32).reshape(-1)
            ends = np.asarray(result["phase_end_steps"], dtype=int).reshape(-1)
            lengths = np.concatenate([ends[:1], ends[1:] - ends[:-1]])
            phases = np.searchsorted(ends, np.arange(U_saved.shape[0]), side="right")
            dts = phase_times[phases] / lengths[phases]
            node_times = np.concatenate([[0.0], np.cumsum(dts)]).astype(np.float32)
        else:
            # This fallback is only for older fixed-dt references.
            node_times = (
                np.arange(X_saved.shape[0], dtype=np.float32) * float(config.dt)
            )

        if node_times.shape[0] != X_saved.shape[0]:
            raise ValueError(
                f"node_times has length {node_times.shape[0]}, "
                f"but X has {X_saved.shape[0]} nodes."
            )

        dt = np.diff(node_times)
        if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
            raise ValueError("Reference node_times must be finite and strictly increasing.")

        contact = _infer_contact_schedule(result, X_phys)

        obstacle_centers = np.asarray(
            result["obstacle_centers"]
            if "obstacle_centers" in result
            else DEFAULT_OBSTACLE_CENTERS,
            dtype=np.float32,
        )
        obstacle_sizes = np.asarray(
            result["obstacle_sizes"]
            if "obstacle_sizes" in result
            else DEFAULT_OBSTACLE_SIZES,
            dtype=np.float32,
        )

    # Padded control reference lets the terminal cost use the same dictionary shape.
    U_node = np.concatenate([U_saved, U_saved[-1:]], axis=0)

    reference = {
        "p": X_phys[:, :3],
        "quat": X_phys[:, 3:7],
        "q": X_phys[:, 7:7 + NJ],
        "dp": X_phys[:, 7 + NJ:10 + NJ],
        "omega": X_phys[:, 10 + NJ:13 + NJ],
        "dq": X_phys[:, 13 + NJ:13 + 2 * NJ],
        "foot": X_phys[:, FOOT_START:GRF_START],
        "grf": X_phys[:, GRF_START:GRF_STOP],
        "u": U_node,
        "contact": contact,
    }

    return {
        "X_solver": X_solver,
        "U": U_saved,
        "reference": reference,
        "node_times": node_times,
        "dt": dt,
        "obstacle_centers": obstacle_centers,
        "obstacle_sizes": obstacle_sizes,
    }


def take_padded(array, start, length):
    idx = np.minimum(start + np.arange(length), array.shape[0] - 1)
    return array[idx]


def make_window(plan, start):
    ref = {
        key: jnp.asarray(take_padded(value, start, TRACKING_N + 1))
        for key, value in plan["reference"].items()
    }

    # Dynamics uses parameter[t] for t=0,...,TRACKING_N-1.
    dt_window = take_padded(plan["dt"], start, TRACKING_N + 1)
    parameter = np.concatenate([
        np.asarray(ref["contact"]),
        np.asarray(ref["foot"]),
        dt_window[:, None],
    ], axis=1)

    X_guess = jnp.asarray(
        take_padded(plan["X_solver"], start, TRACKING_N + 1)
    )
    U_guess = jnp.asarray(
        take_padded(plan["U"], start, TRACKING_N)
    )

    return ref, jnp.asarray(parameter), X_guess, U_guess


# -----------------------------------------------------------------------------
# Fixed-time whole-body dynamics
# -----------------------------------------------------------------------------

def tracking_dynamics(model, mjx_model, contact_id, body_id):
    """Same contact dynamics as the direct solver, but dt comes from the reference."""
    contact_body_ids = (body_id[0], body_id[0], body_id[1], body_id[1])

    def dynamics(x, u, t, parameter):
        dt = parameter[t, NC + 3 * NC]

        qpos = x[:QPOS_STOP].at[3:7].set(
            safe_normalize_quaternion(x[3:7])
        )
        qvel = x[QVEL_START:QVEL_STOP] * VELOCITY_SCALE

        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        data = mjx.fwd_position(mjx_model, data)
        data = mjx.fwd_velocity(mjx_model, data)

        mass_factor = data.qLD
        bias_force = data.qfrc_bias

        contact = parameter[t, :NC]
        target_feet = parameter[t, NC:NC + 3 * NC]

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
        position_violation = contact_mask * (current_feet - target_feet)

        mass_inv_jacobian = jax.scipy.linalg.cho_solve(
            (mass_factor, False), active_jacobian
        )
        projected_mass = active_jacobian.T @ mass_inv_jacobian
        projected_mass = 0.5 * (projected_mass + projected_mass.T)
        projected_mass = (
            projected_mass
            + jnp.diag(1.0 - contact_mask)
            + CONTACT_SOLVE_REGULARIZATION
            * jnp.eye(projected_mass.shape[0], dtype=x.dtype)
        )

        rhs = (
            -active_jacobian.T
            @ jax.scipy.linalg.cho_solve(
                (mass_factor, False), tau - bias_force
            )
            - CONTACT_VELOCITY_GAIN * velocity_violation
            - CONTACT_POSITION_GAIN * position_violation
        )

        raw_grf = jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(projected_mass), rhs
        ).reshape(NC, 3)

        normal = jnp.clip(raw_grf[:, 2], 0.0, MAX_NORMAL_FORCE)
        tangent = raw_grf[:, :2]
        tangent_norm = jnp.sqrt(
            jnp.sum(tangent * tangent, axis=1)
            + TANGENTIAL_FORCE_SMOOTHING**2
        )
        tangent_scale = jnp.minimum(
            1.0, FRICTION_COEFFICIENT * normal / tangent_norm
        )
        grf_matrix = jnp.concatenate([
            tangent * tangent_scale[:, None],
            normal[:, None],
        ], axis=1)
        grf = (grf_matrix * contact[:, None]).reshape(-1)

        acceleration = jax.scipy.linalg.cho_solve(
            (mass_factor, False),
            tau - bias_force + active_jacobian @ grf,
        )

        velocity_next = qvel + acceleration * dt

        return jnp.concatenate([
            qpos[:3] + velocity_next[:3] * dt,
            math.quat_integrate(qpos[3:7], velocity_next[3:6], dt),
            qpos[7:] + velocity_next[6:] * dt,
            velocity_next / VELOCITY_SCALE,
            current_feet,
            grf / GRF_SCALE,
        ])

    return dynamics


# -----------------------------------------------------------------------------
# Tracking cost: no time objective, no waypoint objective, no hard terminal pose
# -----------------------------------------------------------------------------

def tracking_cost(W, reference, x, u, t):
    p = x[:3]
    quat_raw = x[3:7]
    quat = safe_normalize_quaternion(quat_raw)
    q = x[7:7 + NJ]

    dp = x[7 + NJ:10 + NJ] * VELOCITY_SCALE[:3]
    omega = x[10 + NJ:13 + NJ] * VELOCITY_SCALE[3:6]
    dq = x[13 + NJ:13 + 2 * NJ] * VELOCITY_SCALE[6:]

    feet = x[FOOT_START:GRF_START]
    grf = x[GRF_START:GRF_STOP] * GRF_SCALE

    qerr = math.quat_sub(quat, reference["quat"][t])
    contact_map = jnp.repeat(reference["contact"][t], 3)

    state_cost = (
        (p - reference["p"][t]).T @ W["pos"] @ (p - reference["p"][t])
        + qerr.T @ W["rot"] @ qerr
        + (q - reference["q"][t]).T @ W["q"] @ (q - reference["q"][t])
        + (dp - reference["dp"][t]).T @ W["vel"] @ (dp - reference["dp"][t])
        + (omega - reference["omega"][t]).T
        @ W["omega"] @ (omega - reference["omega"][t])
        + (dq - reference["dq"][t]).T @ W["dq"] @ (dq - reference["dq"][t])
        + (contact_map * (feet - reference["foot"][t])).T
        @ W["contact"]
        @ (contact_map * (feet - reference["foot"][t]))
        + (grf - reference["grf"][t]).T
        @ W["grf"]
        @ (grf - reference["grf"][t])
        + QUATERNION_NORM_WEIGHT
        * (jnp.sum(quat_raw * quat_raw) - 1.0) ** 2
    )

    # Track the nominal torque weakly as feed-forward, but do not penalize a
    # fictitious terminal control.
    uerr = u - reference["u"][t]
    control_cost = jnp.where(
        t < TRACKING_N,
        uerr.T @ W["tau"] @ uerr,
        jnp.asarray(0.0, dtype=x.dtype),
    )

    terminal_scale = jnp.where(
        t == TRACKING_N,
        W["terminal_scale"],
        jnp.asarray(1.0, dtype=x.dtype),
    )

    return 0.5 * (terminal_scale * state_cost + control_cost)


# -----------------------------------------------------------------------------
# Constraints: explicitly NO waypoint constraints
# -----------------------------------------------------------------------------

def make_obstacle_clearance_constraints(obstacle_centers, obstacle_sizes):
    obstacle_centers = jnp.asarray(obstacle_centers)
    obstacle_sizes = jnp.asarray(obstacle_sizes)

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
    obstacle_xz = jnp.array([0, 2])

    def inflated_superellipse(points, clearance):
        radii = (
            0.5 * obstacle_sizes[:, obstacle_xz] + clearance
        ) * SUPERELLIPSOID_SCALE
        normalized = (
            points[:, obstacle_xz][None, :, :]
            - obstacle_centers[:, obstacle_xz][:, None, :]
        ) / radii[:, None, :]
        radial_power = jnp.sum(
            normalized**SUPERELLIPSOID_POWER, axis=-1
        )
        return (2.0 / (1.0 + radial_power) - 1.0).reshape(-1)

    def constraints(x):
        qpos = x[:QPOS_STOP].at[3:7].set(
            safe_normalize_quaternion(x[3:7])
        )
        kinematics = mjx.make_data(model).replace(qpos=qpos)
        kinematics = mjx.fwd_position(mjx_model, kinematics)

        joint_positions = kinematics.xanchor[joint_ids]
        head_position = (
            kinematics.geom_xpos[head_geom_id]
            + kinematics.geom_xmat[head_geom_id] @ head_local_offset
        )
        feet = x[FOOT_START:GRF_START].reshape(NC, 3)

        return jnp.concatenate([
            inflated_superellipse(x[:3][None, :], BASE_CLEARANCE_RADIUS),
            inflated_superellipse(
                head_position[None, :], HEAD_CLEARANCE_RADIUS
            ),
            inflated_superellipse(
                joint_positions, JOINT_CLEARANCE_RADIUS
            ),
            inflated_superellipse(feet, FOOT_CLEARANCE_RADIUS),
        ])

    obstacle_count = int(obstacle_centers.shape[0])
    constraints.count = obstacle_count * (
        2 + len(joint_ids) + NC
    )
    return constraints


def make_tracking_constraints(obstacle_constraints):
    """Physical constraints only. There are no phase-end/waypoint constraints."""
    def constraints(x, u, t):
        del t

        q = x[7:7 + NJ]
        dq = x[13 + NJ:13 + 2 * NJ] * VELOCITY_SCALE[6:]
        grf = x[GRF_START:GRF_STOP].reshape(NC, 3) * GRF_SCALE

        torque_constraints = jnp.concatenate([
            u - TORQUE_LIMIT,
            -TORQUE_LIMIT - u,
        ])

        joint_constraints = jnp.concatenate([
            q - JOINT_MAX,
            JOINT_MIN - q,
            jnp.abs(dq) - MAX_JOINT_SPEED,
        ])

        # Apply the friction cone to every foot. During flight the dynamics
        # contact mask forces the corresponding GRF exactly to zero.
        tangent_force = (
            jnp.sqrt(
                jnp.sum(grf[:, :2] ** 2, axis=1)
                + TANGENTIAL_FORCE_SMOOTHING**2
            )
            - TANGENTIAL_FORCE_SMOOTHING
        )
        normal_force = grf[:, 2]
        force_constraints = jnp.stack([
            -normal_force,
            normal_force - MAX_NORMAL_FORCE,
            tangent_force - FRICTION_COEFFICIENT * normal_force,
        ], axis=1)

        return jnp.concatenate([
            torque_constraints,
            joint_constraints,
            force_constraints.reshape(-1),
            jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
            obstacle_constraints(x),
        ])

    return constraints


def make_zero_disturbance(n):
    def disturbance(X):
        return jnp.zeros((X.shape[0], n, n), dtype=X.dtype)
    return disturbance


def unused_reference_generator(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("This controller supplies an explicit moving reference.")


def configure_problem():
    # The controller state is only the 75-D physical state. The six duration
    # variables from the minimum-time planner are intentionally removed.
    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]

    config.N = TRACKING_N
    config.n = PHYSICAL_N
    config.nu = NJ
    config.m = NJ
    config.initial_state = base_state

    config.dynamics = tracking_dynamics
    config.cost = tracking_cost
    config.hessian_approx = None
    config.reference_generator = unused_reference_generator

    config.max_torque = TORQUE_LIMIT
    config.min_torque = -TORQUE_LIMIT

    dtype = base_state.dtype
    config.W = {
        # These are the soft tracking weights from the direct backflip problem.
        # Unlike the stock walking config, x/y position are intentionally nonzero
        # because there are no waypoint constraints anymore.
        "pos": jnp.diag(jnp.array([120.0, 180.0, 140.0], dtype=dtype)),
        "rot": jnp.diag(jnp.array([160.0, 1100.0, 160.0], dtype=dtype)),
        "q": jnp.eye(NJ, dtype=dtype) * 10.0,
        "vel": jnp.diag(jnp.array([3.0, 4.0, 3.0], dtype=dtype)),
        "omega": jnp.diag(jnp.array([2.0, 22.0, 2.0], dtype=dtype)),
        "dq": jnp.eye(NJ, dtype=dtype) * 2.0e-1,
        "contact": jnp.eye(3 * NC, dtype=dtype) * 50.0,
        "tau": jnp.eye(NJ, dtype=dtype) * 2.0e-3,
        "grf": jnp.eye(3 * NC, dtype=dtype) * 1.0e-4,
        "terminal_scale": jnp.asarray(8.0, dtype=dtype),
    }


def solve_one_rti_step(mpc, data, x0, reference, parameter, X_guess, U_guess):
    """Exactly one SQP update for the current 15-step MPC window."""
    return mpc._solve(
        reference,
        parameter,
        data.W,
        x0,
        X_guess,
        U_guess,
        data.V0,
        data.w,
        data.y,
        data.rho,
        data.rho_grad,
        mpc.obstacles,
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


def shift_primal_warm_start(X, U, x_next, X_reference, U_reference):
    """Shift the previous RTI solution one node and pad from the next reference."""
    X_shift = jnp.concatenate([X[1:], X_reference[-1:]], axis=0)
    X_shift = X_shift.at[0].set(x_next)

    U_shift = jnp.concatenate([U[1:], U_reference[-1:]], axis=0)
    return X_shift, U_shift


def physical_state_for_output(x):
    x = np.asarray(x).copy()
    x[QVEL_START:QVEL_STOP] *= np.asarray(VELOCITY_SCALE)
    x[GRF_START:GRF_STOP] *= GRF_SCALE
    return x


def state_tracking_errors(x_solver, ref):
    x = physical_state_for_output(x_solver)
    pos = np.linalg.norm(x[:3] - np.asarray(ref["p"]))
    q_actual = x[3:7] / max(np.linalg.norm(x[3:7]), 1.0e-8)
    q_ref = np.asarray(ref["quat"])
    q_ref = q_ref / max(np.linalg.norm(q_ref), 1.0e-8)
    quat_alignment = abs(float(np.dot(q_actual, q_ref)))
    angle = 2.0 * np.arccos(np.clip(quat_alignment, -1.0, 1.0))
    return pos, np.degrees(angle)


def main(reference_file: Path, output_file: Path):
    plan = load_plan(reference_file)
    configure_problem()

    obstacle_constraints = make_obstacle_clearance_constraints(
        plan["obstacle_centers"], plan["obstacle_sizes"]
    )
    constraints = make_tracking_constraints(obstacle_constraints)

    admm_config = ADMMConfig(
        eps_abs=1.0e-2,
        eps_rel=5.0e-3,
        rho_max=1.0e4,
        max_iterations=400,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        # There are no phase-duration variables in this controller.
        num_phases=1,
    )

    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        # Critical for RTI: do not run a block of "initial" SQP iterations
        # on every MPC solve.
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=True,
        gradient_window=0,
    )

    sqp_config = SQPConfig(
        # One SQP update per receding-horizon control step.
        max_sqp_iterations=1,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        line_search=False,
        lm_regularization=5.0e-2,
    )

    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=sls_config,
        sqp_config=sqp_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=config.initial_state.dtype),
        disturbance=make_zero_disturbance(config.n),
    )

    data = mpc.make_data()

    n_intervals = plan["U"].shape[0]

    # Start exactly at the reference initial state. Replace this with your
    # estimator state in hardware.
    x_current = jnp.asarray(plan["X_solver"][0])

    X_rollout = [physical_state_for_output(x_current)]
    U_rollout = []
    solve_times = []
    pos_errors = []
    angle_errors = []

    X_warm = None
    U_warm = None

    for k in range(n_intervals):
        reference, parameter, X_ref_guess, U_ref_guess = make_window(plan, k)

        if X_warm is None:
            X_guess = X_ref_guess.at[0].set(x_current)
            U_guess = U_ref_guess
        else:
            X_guess = X_warm.at[0].set(x_current)
            U_guess = U_warm

        start = timer()
        result = solve_one_rti_step(
            mpc,
            data,
            x_current,
            reference,
            parameter,
            X_guess,
            U_guess,
        )

        X_opt, U_opt = result[:2]
        X_opt.block_until_ready()
        elapsed = timer() - start

        if not bool(jnp.all(jnp.isfinite(X_opt))) or not bool(
            jnp.all(jnp.isfinite(U_opt))
        ):
            raise FloatingPointError(f"Non-finite RTI solution at MPC step {k}.")

        u_apply = U_opt[0]

        # Exact-model closed-loop propagation over this reference interval.
        x_next = mpc.dynamics(
            x_current,
            u_apply,
            jnp.asarray(0, dtype=jnp.int32),
            parameter,
        )
        x_next.block_until_ready()

        U_rollout.append(np.asarray(u_apply))
        X_rollout.append(physical_state_for_output(x_next))
        solve_times.append(elapsed)

        pos_err, ang_err = state_tracking_errors(
            x_next,
            {key: value[1] for key, value in reference.items()},
        )
        pos_errors.append(pos_err)
        angle_errors.append(ang_err)

        # Shift the primal solution. The next window's final reference is used
        # only to pad the last node/control; there are still no hard waypoints.
        next_reference, _, next_X_ref, next_U_ref = make_window(
            plan, min(k + 1, n_intervals - 1)
        )
        del next_reference
        X_warm, U_warm = shift_primal_warm_start(
            X_opt, U_opt, x_next, next_X_ref, next_U_ref
        )

        x_current = x_next

        print(
            f"RTI {k:02d}/{n_intervals - 1:02d} | "
            f"solve={1e3 * elapsed:8.3f} ms | "
            f"pos_err={pos_err:7.4f} m | "
            f"ori_err={ang_err:7.3f} deg"
        )

    X_rollout = np.asarray(X_rollout)
    U_rollout = np.asarray(U_rollout)
    solve_times = np.asarray(solve_times)
    pos_errors = np.asarray(pos_errors)
    angle_errors = np.asarray(angle_errors)

    output_file = output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        X=X_rollout,
        U=U_rollout,
        reference_X=np.asarray([
            physical_state_for_output(x) for x in plan["X_solver"]
        ]),
        reference_U=plan["U"],
        node_times=plan["node_times"],
        horizon=np.asarray(TRACKING_N),
        solve_times=solve_times,
        mean_solve_time=np.asarray(np.mean(solve_times)),
        max_solve_time=np.asarray(np.max(solve_times)),
        position_error=pos_errors,
        orientation_error_deg=angle_errors,
        max_position_error=np.asarray(np.max(pos_errors)),
        max_orientation_error_deg=np.asarray(np.max(angle_errors)),
        obstacle_centers=plan["obstacle_centers"],
        obstacle_sizes=plan["obstacle_sizes"],
        rti=np.asarray(True),
        waypoint_constraints=np.asarray(False),
    )

    print()
    print(f"Saved RTI rollout: {output_file}")
    print(f"Horizon: {TRACKING_N}")
    print("SQP updates per MPC step: 1")
    print("Waypoint constraints: disabled")
    print(f"Mean solve time: {1e3 * np.mean(solve_times):.3f} ms")
    print(f"Max solve time:  {1e3 * np.max(solve_times):.3f} ms")
    print(f"Max position tracking error: {np.max(pos_errors):.6f} m")
    print(
        "Max orientation tracking error: "
        f"{np.max(angle_errors):.6f} deg"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=SCRIPT_DIR / "humanoid_backflip_obstacle_min_time.npz",
        help="NPZ generated by the direct minimum-time backflip solver.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "humanoid_backflip_rti_h15.npz",
        help="Output NPZ for the closed-loop RTI rollout.",
    )
    args = parser.parse_args()
    main(args.reference, args.output)
