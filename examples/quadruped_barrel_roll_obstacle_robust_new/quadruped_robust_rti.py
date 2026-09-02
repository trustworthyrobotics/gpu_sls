"""Robust minimum-time RTI tracker with one online horizon-time variable.

The script loads an offline Unitree Go2 barrel-roll trajectory and uses its
state/control sequence as a path reference.  Unlike the fixed-timing tracker,
each RTI solve re-optimizes ONE scalar horizon duration T for the next
TRACK_HORIZON reference intervals.

At every MPC update:
  * the reference is the next H trajectory nodes;
  * one free scalar T is appended to the state;
  * T is constant across the prediction horizon;
  * every stage uses the uniform timestep dt = T / H;
  * the objective is trajectory tracking + TIME_WEIGHT * T;
  * the first optimized control is applied for dt = T / H;
  * the horizon shifts by one reference node and T is re-optimized again.

This therefore time-scales the preplanned trajectory online: RTI tries to
traverse the next H reference intervals as quickly as allowed by the dynamics,
robust tightenings, and path constraints.  The contact ordering remains tied
to the saved trajectory nodes; only the amount of physical time assigned to
the current short horizon is changed.

The same optimized dt is used in the plant dynamics and in E(x), so the robust
SLS tightening remains sensitive to the online timing decision.
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

import config_go2 as config
import mpc_utils as barrel_mpc_utils
import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# -----------------------------------------------------------------------------
# Online MPC dimensions / limits.
# -----------------------------------------------------------------------------
TRACK_HORIZON = 20
PLAN_N = 50
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact

# Exactly one free timing state.  It MUST remain the final state coordinate
# because gpu_sls treats the final `num_phases` state coordinates as free time
# variables in minimum-time mode.
T_INDEX = PHYSICAL_N
TRACK_N = PHYSICAL_N + 1

FOOT_START = 13 + 2 * config.n_joints
GRF_START = FOOT_START + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact

# T is the total physical time assigned to the complete short RTI horizon.
# These defaults correspond to dt in [0.3*config.dt, 2.0*config.dt].
NOMINAL_DT = float(config.dt)
MIN_DT = 0.30 * NOMINAL_DT
MAX_DT = 2.00 * NOMINAL_DT
MIN_HORIZON_TIME = TRACK_HORIZON * MIN_DT
MAX_HORIZON_TIME = TRACK_HORIZON * MAX_DT

# Tracking-vs-speed tradeoff.  The summed stage time term is exactly
# TIME_WEIGHT * T because each of H stages contributes T/H.
TIME_WEIGHT = 1.0e3

# First MPC update: solve the short-horizon NLP to convergence (or this cap).
# Every subsequent update uses exactly one SQP step, i.e. RTI.
INITIAL_FULL_SQP_STEPS = 100
RTI_SQP_STEPS = 1

# Same physical disturbance rates as the robust planner, scaled by optimized dt.
DIST_POSITION = 0.10
DIST_QUATERNION = 0.01
DIST_JOINT_POSITION = 0.02
DIST_LINEAR_VELOCITY = 0.10
DIST_ANGULAR_VELOCITY = 0.20
DIST_JOINT_VELOCITY = 0.05
DIST_FOOT_POSITION = 0.02
DIST_GRF = 5.0

JOINT_MIN = jnp.tile(
    jnp.array([-1.0472, -0.6632, -2.7210]), config.n_contact
)
JOINT_MAX = jnp.tile(
    jnp.array([1.0472, 2.9660, -0.8370]), config.n_contact
)
MAX_JOINT_SPEED = 25.0
MIN_BASE_HEIGHT = 0.12
FRICTION_COEFFICIENT = 0.5
MAX_NORMAL_FORCE = 250.0
TANGENTIAL_FORCE_SMOOTHING = 1.0e-4
CONTACT_SOLVE_REGULARIZATION = 1.0e-6

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


def tracking_dynamics(model, mjx_model, contact_id, body_id):
    """Whole-body dynamics using one optimized duration for the RTI horizon.

    State:
        x = [physical_state, T]

    parameter[t]:
        [FL_contact, FR_contact, RL_contact, RR_contact]

    The scalar T is constant under the augmented dynamics and every stage uses
        dt = T / TRACK_HORIZON.
    """

    n_joints = config.n_joints

    def dynamics(x, u, t, parameter):
        contact = parameter[t, : config.n_contact]
        T = x[T_INDEX]
        dt = T / TRACK_HORIZON

        qpos = x[: n_joints + 7]
        qvel = x[n_joints + 7 : 2 * n_joints + 13]
        data = mjx.make_data(model).replace(qpos=qpos, qvel=qvel)
        data = mjx.fwd_position(mjx_model, data)
        data = mjx.fwd_velocity(mjx_model, data)

        mass_factor = data.qLD
        bias_force = data.qfrc_bias
        tau = jnp.concatenate([jnp.zeros(6, dtype=x.dtype), u])

        feet = [data.geom_xpos[geom_id] for geom_id in contact_id[:4]]
        jacobians = [
            mjx.jac(mjx_model, data, foot, body)[0]
            for foot, body in zip(feet, body_id[:4])
        ]
        contact_jacobian = jnp.concatenate(jacobians, axis=1)
        current_feet = jnp.concatenate(feet)

        contact_mask = jnp.repeat(contact, 3)
        active_jacobian = contact_jacobian * contact_mask[None, :]
        velocity_violation = active_jacobian.T @ qvel
        baumgarte = -50.0 * velocity_violation

        mass_inv_jacobian = jax.scipy.linalg.cho_solve(
            (mass_factor, False), active_jacobian
        )
        projected_mass = active_jacobian.T @ mass_inv_jacobian
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
        quaternion_next = math.quat_integrate(qpos[3:7], velocity_next[3:6], dt)
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

        # T_{k+1} = T_k makes T one global scalar for this short horizon.
        return jnp.concatenate([physical_next, jnp.reshape(T, (1,))])

    return dynamics


def tracking_cost(W, reference, x, u, t, *, parameter):
    """Follow the saved path while minimizing the current horizon duration T."""

    nj = config.n_joints
    nc = config.n_contact

    p = x[:3]
    quat = x[3:7]
    q = x[7 : 7 + nj]
    dp = x[7 + nj : 10 + nj]
    omega = x[10 + nj : 13 + nj]
    dq = x[13 + nj : 13 + 2 * nj]
    feet = x[FOOT_START : FOOT_START + 3 * nc]
    grf = x[GRF_START:GRF_STOP]

    quaternion_error = math.quat_sub(quat, reference["quat"][t])

    tracking = 0.5 * (
        (p - reference["p"][t]).T @ W["pos"] @ (p - reference["p"][t])
        + quaternion_error.T @ W["rot"] @ quaternion_error
        + (q - reference["q"][t]).T @ W["q"] @ (q - reference["q"][t])
        + (dp - reference["dp"][t]).T @ W["vel"] @ (dp - reference["dp"][t])
        + (omega - reference["omega"][t]).T @ W["omega"] @ (omega - reference["omega"][t])
        + (dq - reference["dq"][t]).T @ W["dq"] @ (dq - reference["dq"][t])
        + (feet - reference["foot"][t]).T @ W["contact"] @ (feet - reference["foot"][t])
        + (grf - reference["grf"][t]).T @ W["grf"] @ (grf - reference["grf"][t])
        + (u - reference["tau"][t]).T @ W["tau"] @ (u - reference["tau"][t])
    )

    # H copies of T/H sum to exactly T.  No time cost is added at the terminal
    # node because there is no transition after it.
    dt = x[T_INDEX] / TRACK_HORIZON
    time_cost = jnp.where(
        t < config.N,
        W["time"] * dt,
        jnp.asarray(0.0, dtype=x.dtype),
    )
    return tracking + time_cost

def make_obstacle_clearance_constraints():
    """Same smooth obstacle envelope used by the offline planner."""

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

    def inflated_superellipse(points, clearance):
        radii = (
            (0.5 * OBSTACLE_SIZE[1:] + clearance) * SUPERELLIPSOID_SCALE
        )
        normalized = (points[:, 1:] - OBSTACLE_CENTER[1:]) / radii
        radial_power = jnp.sum(
            normalized**SUPERELLIPSOID_POWER, axis=-1
        )
        return 2.0 / (1.0 + radial_power) - 1.0

    def constraints(x):
        qpos = x[: config.n_joints + 7]
        data = mjx.make_data(model).replace(qpos=qpos)
        data = mjx.fwd_position(mjx_model, data)
        joint_positions = data.xanchor[joint_ids]
        feet = x[FOOT_START : FOOT_START + 3 * config.n_contact].reshape(
            config.n_contact, 3
        )

        return jnp.concatenate(
            [
                inflated_superellipse(
                    x[:3][None, :], BASE_CLEARANCE_RADIUS
                ),
                inflated_superellipse(
                    joint_positions, JOINT_CLEARANCE_RADIUS
                ),
                inflated_superellipse(feet, FOOT_CLEARANCE_RADIUS),
            ]
        )

    return constraints



def make_tracking_constraints(obstacle_constraints):
    """Physical path constraints plus lower/upper bounds on the single T."""

    nj = config.n_joints

    def constraints(x, u, t):
        del t
        q = x[7 : 7 + nj]
        dq = x[13 + nj : 13 + 2 * nj]
        grf = x[GRF_START:GRF_STOP].reshape(config.n_contact, 3)
        T = x[T_INDEX]

        time_constraints = jnp.asarray(
            [MIN_HORIZON_TIME - T, T - MAX_HORIZON_TIME], dtype=x.dtype
        )

        torque_constraints = jnp.concatenate(
            [u - config.max_torque, config.min_torque - u]
        )
        joint_constraints = jnp.concatenate(
            [q - JOINT_MAX, JOINT_MIN - q, jnp.abs(dq) - MAX_JOINT_SPEED]
        )

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
        ).reshape(-1)

        return jnp.concatenate(
            [
                time_constraints,
                torque_constraints,
                joint_constraints,
                contact_force_constraints,
                jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
                obstacle_constraints(x),
            ]
        )

    return constraints


def make_tracking_disturbance(n):
    """Robust disturbance model scaled by the same optimized dt = T/H."""

    nj = config.n_joints
    nc = config.n_contact
    linear_vel_start = 7 + nj
    angular_vel_start = 10 + nj
    joint_vel_start = 13 + nj
    foot_start = 13 + 2 * nj
    foot_stop = foot_start + 3 * nc

    def disturbance_at_state(x, t):
        del t
        dt = x[T_INDEX] / TRACK_HORIZON
        diagonal = jnp.zeros(n, dtype=x.dtype)
        diagonal = diagonal.at[0:3].set(DIST_POSITION * dt)
        diagonal = diagonal.at[3:7].set(DIST_QUATERNION * dt)
        diagonal = diagonal.at[7 : 7 + nj].set(DIST_JOINT_POSITION * dt)
        diagonal = diagonal.at[linear_vel_start : linear_vel_start + 3].set(
            DIST_LINEAR_VELOCITY * dt
        )
        diagonal = diagonal.at[angular_vel_start : angular_vel_start + 3].set(
            DIST_ANGULAR_VELOCITY * dt
        )
        diagonal = diagonal.at[joint_vel_start : joint_vel_start + nj].set(
            DIST_JOINT_VELOCITY * dt
        )
        diagonal = diagonal.at[foot_start:foot_stop].set(DIST_FOOT_POSITION * dt)
        diagonal = diagonal.at[GRF_START:GRF_STOP].set(DIST_GRF * dt)
        # T itself is an optimization variable, not a disturbance channel.
        return jnp.diag(diagonal)

    def disturbance(X):
        return jax.vmap(disturbance_at_state, in_axes=(0, 0))(
            X,
            jnp.arange(X.shape[0], dtype=jnp.int32),
        )

    disturbance.at_state = disturbance_at_state
    return disturbance


def configure_tracking_problem():
    """Configure H-stage RTI with exactly one free minimum-time state T."""

    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    T0 = jnp.asarray(TRACK_HORIZON * NOMINAL_DT, dtype=base_state.dtype)

    config.N = TRACK_HORIZON
    config.n = TRACK_N
    config.nu = config.n_joints
    config.m = config.n_joints
    config.initial_state = jnp.concatenate([base_state, jnp.reshape(T0, (1,))])
    config.dynamics = tracking_dynamics
    config.cost = tracking_cost
    config.hessian_approx = None

    def unused_reference_generator(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("RTI tracker supplies its reference window directly.")

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
        "time": jnp.asarray(TIME_WEIGHT, dtype=dtype),
    }

    return make_tracking_disturbance(config.n)

def load_plan(filename):
    filename = Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        required = ("X", "U", "node_times")
        missing = [name for name in required if name not in result]
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {missing}")

        X_full = np.asarray(result["X"], dtype=np.float64)
        U = np.asarray(result["U"], dtype=np.float64)
        node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)
        feasible = bool(np.asarray(result["feasible"])) if "feasible" in result else None

    if X_full.shape[0] != U.shape[0] + 1:
        raise ValueError(f"Expected X to have one more node than U; got {X_full.shape}, {U.shape}.")
    if node_times.size != X_full.shape[0]:
        raise ValueError("node_times must have one entry per state node.")
    if X_full.shape[1] < PHYSICAL_N:
        raise ValueError(
            f"Saved state has width {X_full.shape[1]}, smaller than physical Go2 state {PHYSICAL_N}."
        )
    if U.shape[1] != config.n_joints:
        raise ValueError(f"Expected {config.n_joints} controls; got {U.shape[1]}.")
    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("node_times must be strictly increasing.")
    if not np.all(np.isfinite(X_full)) or not np.all(np.isfinite(U)):
        raise ValueError("Saved trajectory contains non-finite values.")

    # Strip the appended minimum-time phase states from the online state.
    X = X_full[:, :PHYSICAL_N]
    return filename, X, U, node_times, feasible


def build_full_contact_schedule(plan_n):
    """Regenerate the exact fixed contact schedule used by the planner."""

    if plan_n != PLAN_N:
        raise ValueError(
            f"This tracker expects the planner's {PLAN_N}-interval contact schedule; got {plan_n}."
        )

    _, parameter = barrel_mpc_utils.reference_barrel_roll_min_time(
        PLAN_N,
        config.dt,
        config.n_joints,
        config.n_contact,
        config.p_legs0,
        config.q0,
        LATERAL_DISPLACEMENT,
    )
    parameter = np.asarray(parameter)
    if parameter.shape[0] < plan_n or parameter.shape[1] < config.n_contact:
        raise ValueError(f"Unexpected planner parameter shape: {parameter.shape}")
    return parameter[:plan_n, : config.n_contact]


def pad_window(array, start, length):
    """Slice [start:start+length] and repeat the last row past the plan end."""

    stop = min(start + length, array.shape[0])
    out = array[start:stop]
    if out.shape[0] == 0:
        out = array[-1:]
    if out.shape[0] < length:
        out = np.concatenate(
            [out, np.repeat(out[-1:], length - out.shape[0], axis=0)],
            axis=0,
        )
    return out



def make_reference_window(X_plan, U_plan, node_dts, start):
    """Next H path nodes plus an offline-time warm start for the single T."""

    X_phys = jnp.asarray(pad_window(X_plan, start, TRACK_HORIZON + 1))
    U = jnp.asarray(pad_window(U_plan, start, TRACK_HORIZON))

    # The saved timing is ONLY an initial guess for T.  It is not fixed data in
    # the online dynamics.  Sum the next H saved stage times to seed the solve.
    dt_guess = np.asarray(
        pad_window(node_dts[:, None], start, TRACK_HORIZON)
    ).reshape(-1)
    T_guess = float(np.sum(dt_guess))
    T_guess = float(np.clip(T_guess, MIN_HORIZON_TIME, MAX_HORIZON_TIME))
    T_nodes = jnp.full(
        (TRACK_HORIZON + 1, 1), T_guess, dtype=X_phys.dtype
    )
    X = jnp.concatenate([X_phys, T_nodes], axis=1)

    # The stage cost can be evaluated at the terminal node, so pad U once.
    U_nodes = jnp.concatenate([U, U[-1:]], axis=0)

    nj = config.n_joints
    return {
        "p": X_phys[:, :3],
        "quat": X_phys[:, 3:7],
        "q": X_phys[:, 7 : 7 + nj],
        "dp": X_phys[:, 7 + nj : 10 + nj],
        "omega": X_phys[:, 10 + nj : 13 + nj],
        "dq": X_phys[:, 13 + nj : 13 + 2 * nj],
        "foot": X_phys[:, FOOT_START:GRF_START],
        "grf": X_phys[:, GRF_START:GRF_STOP],
        "tau": U_nodes,
    }, X, U


def make_parameter_window(contact_schedule, start):
    """Only contact mode is fixed; timing is optimized through state T."""
    contact = pad_window(contact_schedule, start, TRACK_HORIZON)
    return jnp.asarray(contact)

def max_sls_tube_radius(Phi_x):
    """Maximum physical-state tube radius for one RTI solution."""
    tubes = jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)
    return jnp.max(tubes[..., :PHYSICAL_N])


def inject_unit_ball_disturbance(x, disturbance, key, scale=1.0):
    """Apply x+ = E(x)w with ||w||_2 <= scale to the physical plant state."""
    direction_key, radius_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (config.n,), dtype=x.dtype)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-12)
    # A random radius makes this a sample from inside, rather than only on, the
    # boundary of the modeled unit 2-ball.
    radius = jax.random.uniform(radius_key, (), dtype=x.dtype) ** (1.0 / PHYSICAL_N)
    w = scale * radius * direction
    E = disturbance.at_state(x, jnp.asarray(0, dtype=jnp.int32))
    return E @ w


def workspace_from_data(data):
    return (
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

def print_tree_signature(name, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)

    print(f"\n{name}: {treedef}")
    for i, x in enumerate(leaves):
        print(
            i,
            type(x),
            getattr(x, "shape", None),
            getattr(x, "dtype", None),
            getattr(x, "weak_type", None),
        )


def solve_rti_step(mpc, data, x0, reference, parameter, X_guess, U_guess, workspace):
    """One receding-horizon solve using the same private solver entry point."""

    if workspace is None:
        workspace = workspace_from_data(data)

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
        data.W,
        x0,
        X_guess,
        U_guess,
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



def shift_primal_guess(
    X_solution,
    U_solution,
    next_reference_X,
    next_reference_U,
    next_x0,
):
    """Shift RTI warm start and carry the latest optimized scalar T forward."""

    T_opt = X_solution[0, T_INDEX]
    terminal = next_reference_X[-1].at[T_INDEX].set(T_opt)
    X_guess = jnp.concatenate([X_solution[1:], terminal[None, :]], axis=0)
    U_guess = jnp.concatenate([U_solution[1:], next_reference_U[-1:]], axis=0)
    X_guess = X_guess.at[:, T_INDEX].set(T_opt)
    X_guess = X_guess.at[0, :PHYSICAL_N].set(next_x0[:PHYSICAL_N])
    return X_guess, U_guess

def state_tracking_error(x, x_ref):
    position_error = np.linalg.norm(x[:3] - x_ref[:3])
    velocity_start = 7 + config.n_joints
    velocity_stop = 13 + 2 * config.n_joints
    velocity_error = np.linalg.norm(
        x[velocity_start:velocity_stop] - x_ref[velocity_start:velocity_stop]
    )
    quat_a = x[3:7] / max(np.linalg.norm(x[3:7]), 1.0e-12)
    quat_b = x_ref[3:7] / max(np.linalg.norm(x_ref[3:7]), 1.0e-12)
    alignment = abs(float(np.dot(quat_a, quat_b)))
    angle_error = 2.0 * np.arccos(np.clip(alignment, -1.0, 1.0))
    return position_error, np.degrees(angle_error), velocity_error



def main(
    trajectory,
    output,
    max_steps=None,
    dry_run=False,
    inject_disturbance=False,
    disturbance_scale=1.0,
    seed=0,
):
    trajectory_path, X_plan, U_plan, node_times, feasible = load_plan(trajectory)
    plan_n = U_plan.shape[0]
    node_dts = np.diff(node_times)
    contact_schedule = build_full_contact_schedule(plan_n)

    disturbance = configure_tracking_problem()
    obstacle_constraints = make_obstacle_clearance_constraints()
    constraints = make_tracking_constraints(obstacle_constraints)

    # One free final state coordinate = one short-horizon duration T.
    admm_config = ADMMConfig(
        eps_abs=1e-1,
        eps_rel=1e-2,
        rho_max=1.0e6,
        max_iterations=400,
        rho_update_frequency=25,
        initial_rho=25.0,
        regularized_rho_update=False,
        num_phases=1,
    )
    # Use two solver wrappers with identical problem data:
    #   * full_mpc: first receding-horizon update, up to 100 SQP iterations
    #   * rti_mpc:  all later updates, exactly one SQP iteration
    #
    # Keeping the dimensions/problem definition identical lets the converged first
    # solution and SLS/ADMM workspace warm-start the RTI loop directly.
    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )
    full_sqp_config = SQPConfig(
        max_sqp_iterations=INITIAL_FULL_SQP_STEPS,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        line_search=False,
        lm_regularization=1.0e-2,
    )
    rti_sqp_config = SQPConfig(
        max_sqp_iterations=RTI_SQP_STEPS,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        line_search=False,
        lm_regularization=1.0e-2,
    )

    wrapper_kwargs = dict(
        sls_config=sls_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=config.initial_state.dtype),
        disturbance=disturbance,
    )
    full_mpc = mpc_wrapper.MPCWrapper(
        config, sqp_config=full_sqp_config, **wrapper_kwargs
    )
    rti_mpc = mpc_wrapper.MPCWrapper(
        config, sqp_config=rti_sqp_config, **wrapper_kwargs
    )
    full_data = full_mpc.make_data()
    rti_data = rti_mpc.make_data()

    first_reference, first_X_ref, first_U_ref = make_reference_window(
        X_plan, U_plan, node_dts, 0
    )
    first_parameter = make_parameter_window(contact_schedule, 0)

    if dry_run:
        constraint_shape = jax.eval_shape(
            constraints,
            first_X_ref[0],
            jnp.asarray(U_plan[0]),
            jnp.asarray(0, dtype=jnp.int32),
        ).shape
        dynamics_shape = jax.eval_shape(
            rti_mpc.dynamics,
            first_X_ref[0],
            jnp.asarray(U_plan[0]),
            jnp.asarray(0, dtype=jnp.int32),
            first_parameter,
        ).shape
        disturbance_shape = jax.eval_shape(
            disturbance.at_state,
            first_X_ref[0],
            jnp.asarray(0, dtype=jnp.int32),
        ).shape
        print("Single-T robust RTI setup is valid.")
        print(f"Plan: {trajectory_path}")
        print(f"Plan intervals: {plan_n}")
        print(f"Online horizon: {TRACK_HORIZON}")
        print(f"Physical/augmented state dimensions: {PHYSICAL_N}/{config.n}")
        print("Free online timing variables: 1")
        print(f"Initial T guess: {float(first_X_ref[0, T_INDEX]):.6f} s")
        print(f"Initial dt guess: {float(first_X_ref[0, T_INDEX]) / TRACK_HORIZON:.6f} s")
        print(
            f"Allowed T: {MIN_HORIZON_TIME:.6f} .. {MAX_HORIZON_TIME:.6f} s "
            f"(dt {MIN_DT:.6f} .. {MAX_DT:.6f} s)"
        )
        print(f"Reference state window: {first_X_ref.shape}")
        print(f"Reference control window: {first_U_ref.shape}")
        print(f"Parameter window: {first_parameter.shape}")
        print(f"Dynamics output shape: {dynamics_shape}")
        print(f"Disturbance matrix shape: {disturbance_shape}")
        print(f"Constraint count per node: {constraint_shape[0]}")
        print(f"Time weight: {TIME_WEIGHT:.3e}")
        return

    if feasible is False:
        print("WARNING: planner NPZ marks the offline trajectory as infeasible.")

    # n_steps = plan_n if max_steps is None else min(plan_n, max_steps)
    n_steps = 80
    x = first_X_ref[0]
    X_guess = first_X_ref.at[0].set(x)
    U_guess = first_U_ref
    workspace = workspace_from_data(full_data)

    rollout_X_aug = [np.asarray(x)]
    rollout_U = []
    solve_times = []
    position_errors = []
    orientation_errors_deg = []
    velocity_errors = []
    admm_flags = []
    max_tube_radii = []
    injected_disturbances = []
    optimized_horizon_times = []
    applied_dts = []

    def solve_full_step(x0, ref, param, Xg, Ug, ws):
        return solve_rti_step(
            full_mpc, full_data, x0, ref, param, Xg, Ug, ws
        )

    def solve_rti_update(x0, ref, param, Xg, Ug, ws):
        return solve_rti_step(
            rti_mpc, rti_data, x0, ref, param, Xg, Ug, ws
        )

    solve_full_step = jax.jit(solve_full_step)
    solve_rti_update = jax.jit(solve_rti_update)
    key = jax.random.PRNGKey(seed)

    for k in range(n_steps):
        reference, X_ref, U_ref = make_reference_window(
            X_plan, U_plan, node_dts, k
        )
        parameter = make_parameter_window(contact_schedule, k)

        # T is free in the minimum-time initial condition.  Use the warm-start T
        # only as the current linearization point, never as a fixed measurement.
        x = x.at[T_INDEX].set(X_guess[0, T_INDEX])
        X_guess = X_guess.at[0, :PHYSICAL_N].set(x[:PHYSICAL_N])

        # The first update is a genuine full SQP solve.  Once that trajectory is
        # converged, switch permanently to one-SQP-step RTI updates.
        solver = solve_full_step if k == 0 else solve_rti_update

        start = timer()
        result = solver(
            x,
            reference,
            parameter,
            X_guess,
            U_guess,
            workspace,
        )
        X_solution, U_solution = result[:2]
        X_solution.block_until_ready()
        elapsed = timer() - start

        workspace = result[2:]
        u = U_solution[0]
        T_opt = X_solution[0, T_INDEX]
        dt_opt = T_opt / TRACK_HORIZON

        optimized_horizon_times.append(float(np.asarray(T_opt)))
        applied_dts.append(float(np.asarray(dt_opt)))
        rollout_U.append(np.asarray(u))
        solve_times.append(elapsed)
        admm_flags.append(bool(np.asarray(result[-1])))
        max_tube_radii.append(float(max_sls_tube_radius(result[8])))

        # Apply only the first optimized stage.  On the next measurement the
        # horizon shifts one saved path node and T is optimized again.
        current_x = x.at[T_INDEX].set(T_opt)
        x = rti_mpc.dynamics(
            current_x,
            u,
            jnp.asarray(0, dtype=jnp.int32),
            parameter,
        )

        if inject_disturbance:
            key, subkey = jax.random.split(key)
            delta = inject_unit_ball_disturbance(
                current_x, disturbance, subkey, disturbance_scale
            )
            x = x + delta
            quat = x[3:7]
            x = x.at[3:7].set(
                quat / jnp.maximum(jnp.linalg.norm(quat), 1.0e-12)
            )
            # The free timing state is never physically disturbed.
            x = x.at[T_INDEX].set(T_opt)
            injected_disturbances.append(np.asarray(delta[:PHYSICAL_N]))
        else:
            injected_disturbances.append(np.zeros(PHYSICAL_N, dtype=np.float64))

        x.block_until_ready()
        rollout_X_aug.append(np.asarray(x))

        ref_next = X_plan[min(k + 1, plan_n)]
        pe, oe, ve = state_tracking_error(np.asarray(x)[:PHYSICAL_N], ref_next)
        position_errors.append(pe)
        orientation_errors_deg.append(oe)
        velocity_errors.append(ve)

        if k + 1 < n_steps:
            _, next_X_ref, next_U_ref = make_reference_window(
                X_plan, U_plan, node_dts, k + 1
            )
            X_guess, U_guess = shift_primal_guess(
                X_solution,
                U_solution,
                next_X_ref,
                next_U_ref,
                x,
            )

        solve_label = "FULL" if k == 0 else "RTI "
        print(
            f"{solve_label} {k:02d}: solve={elapsed * 1e3:8.3f} ms  "
            f"T={optimized_horizon_times[-1]:.5f} s  "
            f"dt={applied_dts[-1] * 1e3:7.3f} ms  "
            f"pos_err={pe:.4f} m  rot_err={oe:.2f} deg  "
            f"tube_max={max_tube_radii[-1]:.4e}  "
            f"ADMM={admm_flags[-1]}"
        )

    rollout_X_aug = np.asarray(rollout_X_aug)
    rollout_X = rollout_X_aug[:, :PHYSICAL_N]
    rollout_U = np.asarray(rollout_U)
    solve_times = np.asarray(solve_times)
    position_errors = np.asarray(position_errors)
    orientation_errors_deg = np.asarray(orientation_errors_deg)
    velocity_errors = np.asarray(velocity_errors)
    max_tube_radii = np.asarray(max_tube_radii)
    injected_disturbances = np.asarray(injected_disturbances)
    optimized_horizon_times = np.asarray(optimized_horizon_times)
    applied_dts = np.asarray(applied_dts)
    executed_node_times = np.concatenate([[0.0], np.cumsum(applied_dts)])

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        X_rollout=rollout_X,
        X_rollout_augmented=rollout_X_aug,
        U_rollout=rollout_U,
        X_reference=X_plan[: n_steps + 1],
        U_reference=U_plan[:n_steps],
        offline_node_times=node_times[: n_steps + 1],
        executed_node_times=executed_node_times,
        optimized_horizon_times=optimized_horizon_times,
        applied_dts=applied_dts,
        min_horizon_time=np.asarray(MIN_HORIZON_TIME),
        max_horizon_time=np.asarray(MAX_HORIZON_TIME),
        time_weight=np.asarray(TIME_WEIGHT),
        solve_times=solve_times,
        position_errors=position_errors,
        orientation_errors_deg=orientation_errors_deg,
        velocity_errors=velocity_errors,
        admm_converged=np.asarray(admm_flags),
        max_sls_tube_radius=max_tube_radii,
        injected_disturbances=injected_disturbances,
        disturbance_injected=np.asarray(inject_disturbance),
        disturbance_scale=np.asarray(disturbance_scale),
        disturbance_seed=np.asarray(seed),
        horizon=np.asarray(TRACK_HORIZON),
        source_trajectory=np.asarray(str(trajectory_path)),
    )

    steady = solve_times[3:] if solve_times.size > 3 else solve_times
    print(f"Saved robust RTI rollout: {output}")
    if steady.size:
        print(f"Mean steady-state RTI solve: {1e3 * steady.mean():.3f} ms")
    if applied_dts.size:
        print(f"Executed trajectory time: {executed_node_times[-1]:.6f} s")
        print(
            f"Applied dt range: {applied_dts.min():.6f} .. "
            f"{applied_dts.max():.6f} s"
        )
    if position_errors.size:
        print(f"Max position error: {position_errors.max():.6f} m")
        print(f"Max orientation error: {orientation_errors_deg.max():.3f} deg")
        print(f"Max SLS tube radius: {max_tube_radii.max():.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DIR_PATH / "quadruped_barrel_roll_obstacle_min_time.npz",
        help="Offline robust minimum-time NPZ whose path should be followed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIR_PATH / "quadruped_barrel_roll_robust_rti_rollout.npz",
        help="Closed-loop single-T robust RTI rollout NPZ.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optionally stop after this many receding-horizon updates.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate shapes/callbacks without compiling the RTI solve.",
    )
    parser.add_argument(
        "--inject-disturbance",
        action="store_true",
        help="Inject bounded plant disturbances drawn from the same E(x) model.",
    )
    parser.add_argument(
        "--disturbance-scale",
        type=float,
        default=1.0,
        help="Radius multiplier for optional injected unit-ball disturbances.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for optional disturbance injection.",
    )
    args = parser.parse_args()
    main(
        args.trajectory,
        args.output,
        args.max_steps,
        args.dry_run,
        args.inject_disturbance,
        args.disturbance_scale,
        args.seed,
    )
