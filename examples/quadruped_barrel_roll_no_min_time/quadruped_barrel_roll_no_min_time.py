"""Fixed-time, multi-phase barrel-roll baseline for the Unitree Go2.

Every phase has a fixed number of shooting intervals and a fixed physical
duration.  Unlike quadruped_barrel_roll, this baseline has no duration states
and no minimum-time objective; it optimizes trajectory quality only.

Use --dry-run to validate callback shapes without compiling the full solver.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

# Configure paths and headless MuJoCo before importing JAX/MuJoCo.
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
import mpx.utils.objectives as mpc_objectives
import mpx.utils.sim as sim_utils
import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# Fixed phase schedule encoded by reference_barrel_roll_fixed_time.
PHASE_NAMES = (
    "stance",
    "lateral_launch",
    "flight",
    "prelanding",
    "touchdown",
    "settle",
)
FIXED_PHASE_DURATIONS = jnp.array([0.20, 0.20, 0.24, 0.06, 0.10, 0.20])
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

FEASIBILITY_TOLERANCE = 2.0e-2
DYNAMICS_TOLERANCE = 1.5e-1

# Original MPX state: qpos, qvel, foot positions, and GRFs.
PHYSICAL_N = 13 + 2 * config.n_joints + 6 * config.n_contact
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


def phase_index(t):
    index = jnp.searchsorted(PHASE_END_STEPS, t, side="right")
    return jnp.clip(index, 0, NUM_PHASES - 1)


def fixed_time_barrel_roll_dynamics(model, mjx_model, contact_id, body_id):
    """Build dynamics with a fixed timestep for every scheduled phase."""

    n_joints = config.n_joints

    def dynamics(x, u, t, parameter):
        phase = phase_index(t)
        dt = FIXED_PHASE_DURATIONS[phase] / SEGMENT_LENGTHS[phase]

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

        return physical_next

    return dynamics


def fixed_time_barrel_roll_cost(W, reference, x, u, t):
    """Whole-body tracking objective with no minimum-time term."""

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
    return 0.5 * stage_cost


def build_barrel_roll_reference():
    """Pad the existing N-sample barrel reference for the N+1-state SQP."""

    reference, parameter = barrel_mpc_utils.reference_barrel_roll_fixed_time(
        config.N,
        config.dt,
        config.n_joints,
        config.n_contact,
        config.p_legs0,
        config.q0,
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
    node_dts = FIXED_PHASE_DURATIONS[phase_ids] / SEGMENT_LENGTHS[phase_ids]
    joint_rates = (reference["q"][1:] - reference["q"][:-1]) / node_dts[:, None]
    joint_rates = jnp.concatenate(
        [joint_rates, jnp.zeros((1, nj), dtype=X.dtype)], axis=0
    )
    X = X.at[:, 13 + nj : 13 + 2 * nj].set(joint_rates)
    X = X.at[:, 13 + 2 * nj : 13 + 2 * nj + 3 * nc].set(
        reference["foot"]
    )
    X = X.at[:, GRF_START:GRF_STOP].set(reference["grf"])
    return X.at[0].set(x0)


def make_barrel_roll_constraints(reference):
    """Build hard pose, joint, contact-force, and landing limits."""

    nj = config.n_joints
    waypoint_positions = reference["p"][PHASE_END_STEPS]
    waypoint_quaternions = reference["quat"][PHASE_END_STEPS]
    terminal_joints = reference["q"][-1]
    cosine_limits = jnp.cos(0.5 * WAYPOINT_ANGLE_TOLERANCES)

    def constraints(x, u, t):
        q = x[7 : 7 + nj]
        dp = x[7 + nj : 10 + nj]
        omega = x[10 + nj : 13 + nj]
        dq = x[13 + nj : 13 + 2 * nj]
        grf = x[GRF_START:GRF_STOP].reshape(config.n_contact, 3)

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
                torque_constraints,
                joint_constraints,
                position_constraints.reshape(-1),
                orientation_constraints,
                waypoint_rate_constraints.reshape(-1),
                contact_force_constraints.reshape(-1),
                final_constraints,
                jnp.reshape(MIN_BASE_HEIGHT - x[2], (1,)),
            ]
        )

    return constraints


def make_fixed_time_disturbance(n, magnitude=0.02):
    """Disturb base position, scaled by the fixed phase timestep."""

    def disturbance_at_state(x, t):
        phase = phase_index(t)
        dt = FIXED_PHASE_DURATIONS[phase] / SEGMENT_LENGTHS[phase]
        diagonal = jnp.zeros(n, dtype=x.dtype).at[:3].set(magnitude * dt)
        return jnp.diag(diagonal)

    def disturbance(X):
        return jax.vmap(disturbance_at_state)(X, jnp.arange(X.shape[0]))

    disturbance.at_state = disturbance_at_state
    return disturbance


def unused_reference_generator(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("This one-shot experiment supplies its reference directly.")


def configure_problem():
    """Configure the Go2 problem with no free duration states."""

    base_state = jnp.asarray(config.initial_state)[:PHYSICAL_N]
    config.n = PHYSICAL_N
    config.nu = config.n_joints
    config.initial_state = base_state
    config.dynamics = fixed_time_barrel_roll_dynamics
    config.cost = fixed_time_barrel_roll_cost
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
    }


def solve_barrel_roll(mpc, data, x0, reference, parameter):
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


def fixed_node_times():
    dts = np.asarray(FIXED_PHASE_DURATIONS) / np.asarray(SEGMENT_LENGTHS)
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
    print("Barrel-roll fixed-time setup is valid.")
    print(f"State/control dimensions: {config.n}/{config.nu}")
    print(f"Reference/parameter shapes: {reference['p'].shape}/{parameter.shape}")
    print(f"Initial guess shape: {X0.shape}")
    print(f"Constraint count per node: {constraint_shape[0]}")
    print("Fixed phase times:")
    for name, duration, intervals in zip(
        PHASE_NAMES, FIXED_PHASE_DURATIONS, SEGMENT_LENGTHS
    ):
        print(
            f"  {name:16s} {float(duration):.3f} s "
            f"({int(intervals)} intervals)"
        )


def main(*, dry_run_only=False, output_dir=DIR_PATH):
    configure_problem()
    reference, parameter = build_barrel_roll_reference()
    constraints = make_barrel_roll_constraints(reference)

    if dry_run_only:
        dry_run(reference, parameter, constraints)
        return

    admm_config = ADMMConfig(
        eps_abs=5.0e-2,
        eps_rel=1.0e-3,
        rho_max=1.0e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        num_phases=0,
    )
    sls_config = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1.0e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=25,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )
    sqp_config = SQPConfig(
        max_sqp_iterations=3,
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
        disturbance=make_fixed_time_disturbance(config.n),
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
    )
    data = mpc.make_data()

    start = timer()
    result = solve_barrel_roll(mpc, data, x0, reference, parameter)
    X, U = result[:2]
    converged_admm = result[-1]
    X.block_until_ready()
    solve_time = timer() - start

    phase_times = np.asarray(FIXED_PHASE_DURATIONS)
    total_time = float(np.sum(phase_times))
    max_violation, violation_node, violation_index = (
        evaluate_constraint_violation(constraints, X, U)
    )
    max_dynamics_defect, defect_node, defect_state = evaluate_dynamics_defect(
        mpc.dynamics, X, U, parameter
    )
    position_errors, angle_errors = waypoint_diagnostics(reference, X)

    print(f"Solve time: {solve_time:.3f} s")
    print("Fixed phase times:")
    for name, duration in zip(PHASE_NAMES, phase_times):
        print(f"  {name:16s} {duration:.6f} s")
    print(f"Fixed barrel-roll time: {total_time:.6f} s")
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
    print("Phase-end waypoint errors:")
    for name, position_error, angle_error in zip(
        PHASE_NAMES, position_errors, angle_errors
    ):
        print(
            f"  {name:16s} position={position_error} "
            f"orientation={angle_error:.3f} deg"
        )

    if not np.isfinite(max_violation) or max_violation > FEASIBILITY_TOLERANCE:
        raise RuntimeError(
            "Refusing to overwrite the trajectory with an infeasible solution: "
            f"maximum constraint violation {max_violation:.6e} exceeds "
            f"{FEASIBILITY_TOLERANCE:.6e}."
        )
    if (
        not np.isfinite(max_dynamics_defect)
        or max_dynamics_defect > DYNAMICS_TOLERANCE
    ):
        raise RuntimeError(
            "Refusing to overwrite the trajectory with an inconsistent solution: "
            f"maximum dynamics defect {max_dynamics_defect:.6e} exceeds "
            f"{DYNAMICS_TOLERANCE:.6e}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "quadruped_barrel_roll_no_min_time.npz"
    np.savez(
        result_path,
        X=np.asarray(X),
        U=np.asarray(U),
        phase_names=np.asarray(PHASE_NAMES),
        phase_end_steps=np.asarray(PHASE_END_STEPS),
        phase_times=phase_times,
        total_time=np.asarray(total_time),
        node_times=fixed_node_times(),
        timing_mode=np.asarray("fixed"),
        fixed_phase_times=np.asarray(True),
        max_constraint_violation=np.asarray(max_violation),
        max_dynamics_defect=np.asarray(max_dynamics_defect),
        waypoint_position_errors=position_errors,
        waypoint_orientation_errors_deg=angle_errors,
        reference_version=np.asarray(3),
    )
    print(f"Saved feasible trajectory: {result_path}")


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
