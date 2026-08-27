"""RTI MPC tracker for the optimized Unitree Go2 barrel-roll trajectory.

The minimum-time problem is solved OFFLINE by the planner.  This script loads
that solution and tracks it ONLINE with a 20-stage receding horizon.  Each MPC
update performs one SQP step (RTI), warm-started from the previous solution.

Important differences from the offline minimum-time problem:
  * phase durations are NOT decision variables online;
  * phase-end waypoint equalities are NOT imposed online;
  * the optimized node_times/contact schedule are treated as fixed data;
  * torque, joint, contact-force, base-height, and obstacle constraints remain.

Typical use:
    python track_barrel_roll_rti.py
    python track_barrel_roll_rti.py quadruped_barrel_roll_obstacle_min_time.npz
    python track_barrel_roll_rti.py --dry-run
    python track_barrel_roll_rti.py --max-steps 20

The output NPZ contains the closed-loop rollout and per-step RTI solve times.
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
FOOT_START = 13 + 2 * config.n_joints
GRF_START = FOOT_START + 3 * config.n_contact
GRF_STOP = GRF_START + 3 * config.n_contact

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

# Same obstacle geometry/clearances as the minimum-time planner.
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
    """Build fixed-time whole-body dynamics used by the 20-stage tracker.

    parameter[t] = [FL_contact, FR_contact, RL_contact, RR_contact, dt].
    The dt at each prediction stage comes directly from the optimized node_times.
    """

    n_joints = config.n_joints

    def dynamics(x, u, t, parameter):
        contact = parameter[t, : config.n_contact]
        dt = parameter[t, config.n_contact]

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

        # Solve only active contact constraints, exactly as in the planner.
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
        quaternion_next = math.quat_integrate(
            qpos[3:7], velocity_next[3:6], dt
        )
        joints_next = (
            qpos[7 : 7 + n_joints]
            + velocity_next[6 : 6 + n_joints] * dt
        )

        return jnp.concatenate(
            [
                position_next,
                quaternion_next,
                joints_next,
                velocity_next,
                current_feet,
                grf,
            ]
        )

    return dynamics


def tracking_cost(W, reference, x, u, t):
    """Quadratic tracking cost around the saved optimized state/control path."""

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

    return 0.5 * (
        (p - reference["p"][t]).T @ W["pos"] @ (p - reference["p"][t])
        + quaternion_error.T @ W["rot"] @ quaternion_error
        + (q - reference["q"][t]).T @ W["q"] @ (q - reference["q"][t])
        + (dp - reference["dp"][t]).T
        @ W["vel"]
        @ (dp - reference["dp"][t])
        + (omega - reference["omega"][t]).T
        @ W["omega"]
        @ (omega - reference["omega"][t])
        + (dq - reference["dq"][t]).T
        @ W["dq"]
        @ (dq - reference["dq"][t])
        + (feet - reference["foot"][t]).T
        @ W["contact"]
        @ (feet - reference["foot"][t])
        + (grf - reference["grf"][t]).T
        @ W["grf"]
        @ (grf - reference["grf"][t])
        + (u - reference["tau"][t]).T
        @ W["tau"]
        @ (u - reference["tau"][t])
    )


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
    """Online path constraints; no phase-time or waypoint constraints."""

    nj = config.n_joints

    def constraints(x, u, t):
        del t
        q = x[7 : 7 + nj]
        dq = x[13 + nj : 13 + 2 * nj]
        grf = x[GRF_START:GRF_STOP].reshape(config.n_contact, 3)

        torque_constraints = jnp.concatenate(
            [u - config.max_torque, config.min_torque - u]
        )
        joint_constraints = jnp.concatenate(
            [q - JOINT_MAX, JOINT_MIN - q, jnp.abs(dq) - MAX_JOINT_SPEED]
        )

        # Enforce the cone for all feet. Inactive feet have zero GRF through
        # the dynamics, so zero remains feasible without a time-varying closure.
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
                torque_constraints,
                joint_constraints,
                contact_force_constraints,
                jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
                obstacle_constraints(x),
            ]
        )

    return constraints


def make_tracking_disturbance(n, dt_scale, magnitude=0.02):
    """Fixed small base-position disturbance model for the SLS tracker."""

    diagonal = jnp.zeros(n).at[:3].set(magnitude * dt_scale)
    E = jnp.diag(diagonal)

    def disturbance(X):
        return jnp.broadcast_to(E, (X.shape[0], n, n))

    def disturbance_at_state(x, t):
        del x, t
        return E

    disturbance.at_state = disturbance_at_state
    return disturbance


def configure_tracking_problem(dt_scale):
    """Mutate config_go2 from the 50-stage planner layout to 20-stage tracking."""

    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.N = TRACK_HORIZON
    config.n = PHYSICAL_N
    config.nu = config.n_joints
    config.m = config.n_joints
    config.initial_state = base_state
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
    }

    return make_tracking_disturbance(config.n, dt_scale)


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


def make_reference_window(X_plan, U_plan, start):
    X = jnp.asarray(pad_window(X_plan, start, TRACK_HORIZON + 1))
    U = jnp.asarray(pad_window(U_plan, start, TRACK_HORIZON))
    # The solver may evaluate the stage cost at the terminal node with a padded
    # control, so provide H+1 control references.
    U_nodes = jnp.concatenate([U, U[-1:]], axis=0)

    nj = config.n_joints
    return {
        "p": X[:, :3],
        "quat": X[:, 3:7],
        "q": X[:, 7 : 7 + nj],
        "dp": X[:, 7 + nj : 10 + nj],
        "omega": X[:, 10 + nj : 13 + nj],
        "dq": X[:, 13 + nj : 13 + 2 * nj],
        "foot": X[:, FOOT_START:GRF_START],
        "grf": X[:, GRF_START:GRF_STOP],
        "tau": U_nodes,
    }, X, U


def make_parameter_window(contact_schedule, node_dts, start):
    contact = pad_window(contact_schedule, start, TRACK_HORIZON)
    dt = pad_window(node_dts[:, None], start, TRACK_HORIZON)
    return jnp.asarray(np.concatenate([contact, dt], axis=1))


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


def shift_primal_guess(X_solution, U_solution, next_reference_X, next_reference_U, next_x0):
    """Standard RTI primal shift by one stage."""

    X_guess = jnp.concatenate([X_solution[1:], next_reference_X[-1:]], axis=0)
    U_guess = jnp.concatenate([U_solution[1:], next_reference_U[-1:]], axis=0)
    X_guess = X_guess.at[0].set(next_x0)
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


def main(trajectory, output, max_steps=None, dry_run=False):
    trajectory_path, X_plan, U_plan, node_times, feasible = load_plan(trajectory)
    plan_n = U_plan.shape[0]
    node_dts = np.diff(node_times)
    contact_schedule = build_full_contact_schedule(plan_n)

    disturbance = configure_tracking_problem(float(np.median(node_dts)))
    obstacle_constraints = make_obstacle_clearance_constraints()
    constraints = make_tracking_constraints(obstacle_constraints)

    admm_config = ADMMConfig(
        eps_abs=1.0e-1,
        eps_rel=1.0e-3,
        rho_max=1.0e6,
        max_iterations=400,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
    )
    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=True,
        gradient_window=0,
    )
    sqp_config = SQPConfig(
        max_sqp_iterations=1,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        # RTI applies one local SQP/QP step; a line-search globalization loop
        # would defeat the fixed-work-per-control-tick behavior.
        line_search=False,
        lm_regularization=1.0e-2,
    )

    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=sls_config,
        sqp_config=sqp_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=config.initial_state.dtype),
        disturbance=disturbance,
    )
    data = mpc.make_data()

    first_reference, first_X_ref, first_U_ref = make_reference_window(
        X_plan, U_plan, 0
    )
    first_parameter = make_parameter_window(
        contact_schedule, node_dts, 0
    )

    if dry_run:
        constraint_shape = jax.eval_shape(
            constraints,
            jnp.asarray(X_plan[0]),
            jnp.asarray(U_plan[0]),
            jnp.asarray(0, dtype=jnp.int32),
        ).shape
        dynamics_shape = jax.eval_shape(
            mpc.dynamics,
            jnp.asarray(X_plan[0]),
            jnp.asarray(U_plan[0]),
            jnp.asarray(0, dtype=jnp.int32),
            first_parameter,
        ).shape
        print("RTI barrel-roll tracker setup is valid.")
        print(f"Plan: {trajectory_path}")
        print(f"Plan intervals: {plan_n}")
        print(f"Online horizon: {TRACK_HORIZON}")
        print(f"Online state/control dimensions: {config.n}/{config.nu}")
        print(f"Reference state window: {first_X_ref.shape}")
        print(f"Reference control window: {first_U_ref.shape}")
        print(f"Parameter window: {first_parameter.shape}")
        print(f"Dynamics output shape: {dynamics_shape}")
        print(f"Constraint count per node: {constraint_shape[0]}")
        print(
            f"Optimized dt range: {node_dts.min():.6f} .. {node_dts.max():.6f} s"
        )
        return

    if feasible is False:
        print("WARNING: planner NPZ marks the offline trajectory as infeasible.")

    n_steps = plan_n if max_steps is None else min(plan_n, max_steps)
    x = jnp.asarray(X_plan[0])
    X_guess = first_X_ref.at[0].set(x)
    U_guess = first_U_ref
    # Keep a fixed workspace pytree structure from the first RTI call onward;
    # otherwise JAX would compile once for ws=None and again for the tuple.
    workspace = workspace_from_data(data)
    print_tree_signature("INITIAL WORKSPACE", workspace)

    rollout_X = [np.asarray(x)]
    rollout_U = []
    solve_times = []
    position_errors = []
    orientation_errors_deg = []
    velocity_errors = []
    admm_flags = []

    # JIT the complete RTI solve. The shapes remain fixed for every receding
    # horizon window, so only the first call should compile.
    def solve_step(x0, ref, param, Xg, Ug, ws):
        return solve_rti_step(
            mpc,
            data,
            x0,
            ref,
            param,
            Xg,
            Ug,
            ws,
        )

    for k in range(n_steps):
        reference, X_ref, U_ref = make_reference_window(X_plan, U_plan, k)
        parameter = make_parameter_window(contact_schedule, node_dts, k)

        start = timer()
        result = solve_step(
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

        # The solver returns the same 15 workspace arrays that it accepted,
        # immediately after X and U.
        workspace = result[2:]

        u = U_solution[0]
        rollout_U.append(np.asarray(u))
        if k > 2:
            solve_times.append(elapsed)
        admm_flags.append(bool(np.asarray(result[-1])))

        # Closed-loop plant step with the same nonlinear model/contact schedule.
        x = mpc.dynamics(
            x,
            u,
            jnp.asarray(0, dtype=jnp.int32),
            parameter,
        )
        x.block_until_ready()
        rollout_X.append(np.asarray(x))

        ref_next = X_plan[min(k + 1, plan_n)]
        pe, oe, ve = state_tracking_error(np.asarray(x), ref_next)
        position_errors.append(pe)
        orientation_errors_deg.append(oe)
        velocity_errors.append(ve)

        if k + 1 < n_steps:
            _, next_X_ref, next_U_ref = make_reference_window(
                X_plan, U_plan, k + 1
            )
            X_guess, U_guess = shift_primal_guess(
                X_solution,
                U_solution,
                next_X_ref,
                next_U_ref,
                x,
            )

        print(
            f"RTI {k:02d}: solve={elapsed * 1e3:8.3f} ms  "
            f"pos_err={pe:.4f} m  rot_err={oe:.2f} deg  "
            f"ADMM={admm_flags[-1]}"
        )

    rollout_X = np.asarray(rollout_X)
    rollout_U = np.asarray(rollout_U)
    solve_times = np.asarray(solve_times)
    position_errors = np.asarray(position_errors)
    orientation_errors_deg = np.asarray(orientation_errors_deg)
    velocity_errors = np.asarray(velocity_errors)

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        X_rollout=rollout_X,
        U_rollout=rollout_U,
        X_reference=X_plan[: n_steps + 1],
        U_reference=U_plan[:n_steps],
        node_times=node_times[: n_steps + 1],
        solve_times=solve_times,
        position_errors=position_errors,
        orientation_errors_deg=orientation_errors_deg,
        velocity_errors=velocity_errors,
        admm_converged=np.asarray(admm_flags),
        horizon=np.asarray(TRACK_HORIZON),
        source_trajectory=np.asarray(str(trajectory_path)),
    )

    # Exclude the first solve from the steady-state timing statistic because it
    # includes JAX compilation.
    steady = solve_times[1:] if solve_times.size > 1 else solve_times
    print(f"Saved RTI rollout: {output}")
    print(f"Mean steady-state RTI solve: {1e3 * steady.mean():.3f} ms")
    print(f"Max position error: {position_errors.max():.6f} m")
    print(f"Max orientation error: {orientation_errors_deg.max():.3f} deg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DIR_PATH / "quadruped_barrel_roll_obstacle_min_time.npz",
        help="Offline minimum-time NPZ to track.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DIR_PATH / "quadruped_barrel_roll_rti_rollout.npz",
        help="Closed-loop rollout NPZ.",
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
    args = parser.parse_args()
    main(args.trajectory, args.output, args.max_steps, args.dry_run)
