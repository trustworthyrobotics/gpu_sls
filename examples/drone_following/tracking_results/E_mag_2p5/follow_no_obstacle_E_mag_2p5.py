from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import config
import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
)

config.update("jax_enable_x64", False)
DTYPE = jnp.float64 if config.x64_enabled else jnp.float32
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)


# ============================================================
# GPU-SLS receding-horizon tracker for the saved GPU-SLS trajectory
#
# Workflow:
#   1. Load the optimized minimum-time GPU-SLS/JAX trajectory.
#   2. Keep its optimized phase times fixed.
#   3. First MPC call: 50 nominal SQP + 50 SLS SQP iterations.
#   4. Rebuild once as a short warm-started SQP controller seeded by that solution.
#   5. At each trajectory node:
#        - set current measured state x_k
#        - set local X/U reference window
#        - set the optimized dt sequence
#        - solve
#        - apply only u_0
#        - roll the plant one step
#        - shift and repeat
#
# Online MPC state (13-D):
#   x = [px, py, pz, phi, theta, psi,
#        vx, vy, vz, p, q, r, dt]
#
# Online MPC control:
#   u = [thrust, tau_phi, tau_theta, tau_psi]
#
# Only dt is augmented into the MPC state.  Stage-active masks, terminal
# motion weighting, and offline control references are runtime parameters.
# The dt state is deterministic: its next value is copied from the saved
# offline dt schedule and is not reoptimized online.
# ============================================================


MASS = 1.0
GRAVITY = 9.81

JX = 0.02
JY = 0.02
JZ = 0.04

J_JAX = jnp.diag(jnp.array([JX, JY, JZ], dtype=DTYPE))
J_INV_JAX = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=DTYPE))

NX = 12
NU = 4

# 12 physical states + one deterministic dt schedule state.
DT_IDX = 12
TRACK_NX = 13

# First MPC solve: 50 nominal SQP iterations, then 50 SLS-enabled SQP iterations.
# Subsequent MPC solves: a few warm-started SQP iterations per control update.
# One iteration was not enough to re-establish dynamic feasibility during the
# aggressive pitch reversal near plan node 90.
GPU_SLS_INITIAL_NOMINAL_SQP_ITERS = 50
GPU_SLS_INITIAL_SLS_SQP_ITERS = 50
GPU_SLS_INITIAL_TOTAL_SQP_ITERS = (
    GPU_SLS_INITIAL_NOMINAL_SQP_ITERS
    + GPU_SLS_INITIAL_SLS_SQP_ITERS
)
GPU_SLS_RTI_SQP_ITERS = 1
GPU_SLS_ADMM_MAX_ITERS = 400


DEFAULT_MPC_HORIZON = 20
DEFAULT_PROGRESS_SEARCH_WINDOW = 15
DEFAULT_GOAL_TOL = 0.362
# Disturbances and nearest-point progress can require more controller updates
# than the nominal plan has intervals.  This also matches the CLI help text.
DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER = 2

# Disturbance-model settings
# DISTURBANCE_MAG controls the retained state-dependent vy uncertainty channel.
# E_MAG_MODEL controls the XY position uncertainty and the actual gate-directed
# rollout adversary. The z-position disturbance is identically zero.
DISTURBANCE_MAG = 2.5
E_MAG_MODEL = 0.0

T_HOVER = MASS * GRAVITY
T_MAX = 2.0 * T_HOVER
TAU_MAX = 10.0

# ZYX Euler angles are singular at pitch = +/- pi/2.  The saved trajectory
# stays inside this limit (its maximum pitch is about 84 degrees), so keeping a
# small margin does not alter the reference but prevents unbounded tan/sec
# terms in the dynamics and their Jacobians.
PITCH_SINGULARITY_MARGIN = np.deg2rad(5.0)
PITCH_MAX = np.pi / 2.0 - PITCH_SINGULARITY_MARGIN

# The QP uses soft constraints, so independently validate the control that is
# about to be sent to the simulated plant.
CONTROL_BOUND_TOL = 5e-2


# ------------------------------------------------------------
# Tracking weights
#
# These are intentionally much stronger than the offline
# trajectory-planning weights. The online problem is now a
# reference tracking problem, not a minimum-time problem.
# ------------------------------------------------------------
Q_POS = np.array([10.0, 10.0, 10.0])
Q_ANG = np.array([5.0, 5.0, 5.0])
Q_VEL = np.array([10.0, 10.0, 10.0])
Q_RATE = np.array([1.0, 1.0, 1.0])

R_U = np.array([0.20, 0.05, 0.05, 0.05])

TERMINAL_MULTIPLIER = 1.0


# ------------------------------------------------------------
# Same physical bounds as the offline planner
# ------------------------------------------------------------
X_MAX = np.array([
    15.0, 15.0, 15.0,
    np.pi / 2.0,
    PITCH_MAX,
    10.0 * np.pi,
    5.0, 5.0, 5.0,
    8.0, 8.0, 8.0,
], dtype=float)

X_MIN = -X_MAX
X_MIN[2] = -1.0

U_MIN = np.array(
    [0.0, -TAU_MAX, -TAU_MAX, -TAU_MAX],
    dtype=float,
)

U_MAX = np.array(
    [T_MAX, TAU_MAX, TAU_MAX, TAU_MAX],
    dtype=float,
)


# ============================================================
# State-dependent vy uncertainty field retained in the robust MPC model.
#
# The actual closed-loop plant adversary is defined below in the XY position
# subspace: it perturbs x/y toward the closest physical gate bar using
# E_MAG_MODEL, while the z-position disturbance is always zero.
# This spatial-scale function is still used by the robust vy channel and
# by the diagnostic background plot.
# ============================================================

@dataclass(frozen=True)
class DisturbanceFieldParams:
    x_min: float = -2.0
    x_max: float = 1.0
    y_min: float = -8.0
    y_max: float = 0.0
    kx: float = 1.0
    ky: float = 1.0


DISTURBANCE_FIELD = DisturbanceFieldParams()


def disturbance_spatial_scale(px, py, params=DISTURBANCE_FIELD):
    x_window = (
        0.5 * (1.0 + np.tanh(params.kx * (px - params.x_min)))
        * 0.5 * (1.0 - np.tanh(params.kx * (px - params.x_max)))
    )

    y_window = (
        0.5 * (1.0 + np.tanh(params.ky * (py - params.y_min)))
        * 0.5 * (1.0 - np.tanh(params.ky * (py - params.y_max)))
    )

    return x_window * y_window


# ============================================================
# NumPy dynamics used for the ACTUAL rollout / plant
# ============================================================

def rotation_matrix_np(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [
            cpsi * cth,
            cpsi * sth * sphi - spsi * cphi,
            cpsi * sth * cphi + spsi * sphi,
        ],
        [
            spsi * cth,
            spsi * sth * sphi + cpsi * cphi,
            spsi * sth * cphi - cpsi * sphi,
        ],
        [
            -sth,
            cth * sphi,
            cth * cphi,
        ],
    ], dtype=float)


def euler_angle_rates_matrix_np(phi, theta):
    theta_safe = np.clip(theta, -PITCH_MAX, PITCH_MAX)
    sphi, cphi = np.sin(phi), np.cos(phi)
    tth = np.tan(theta_safe)
    cth = np.cos(theta_safe)

    return np.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=float)


def quadrotor_xdot_np(x, u):
    phi, theta, psi = x[3], x[4], x[5]

    v = x[6:9]
    omega = x[9:12]

    thrust = u[0]
    tau = u[1:4]

    R = rotation_matrix_np(phi, theta, psi)
    E = euler_angle_rates_matrix_np(phi, theta)

    e3 = np.array([0.0, 0.0, 1.0])

    pos_dot = v
    v_dot = (R @ (thrust * e3)) / MASS - GRAVITY * e3
    euler_dot = E @ omega

    J_np = np.diag([JX, JY, JZ])
    J_inv_np = np.diag([1.0 / JX, 1.0 / JY, 1.0 / JZ])

    omega_dot = J_inv_np @ (
        tau - np.cross(omega, J_np @ omega)
    )

    return np.concatenate([
        pos_dot,
        euler_dot,
        v_dot,
        omega_dot,
    ])


def build_gate_bar_geometry_np(plan):
    """Build the exact inflated four-bar gate geometry used by the MPC.

    The drone center is treated as a point and each gate bar is inflated by
    ``drone_radius``.  This geometry is retained only to define the direction
    of the gate-directed rollout adversary and for visualization/diagnostics.
    """
    required = (
        "gate_centers",
        "gate_normals",
        "gate_axis_u",
        "gate_axis_v",
        "gate_opening_half_widths",
    )
    if not all(plan.get(k) is not None for k in required):
        raise ValueError(
            "Physical gate geometry is required for the gate-directed "
            "adversarial disturbance."
        )

    gate_centers = np.asarray(plan["gate_centers"], dtype=float)
    gate_normals = np.asarray(plan["gate_normals"], dtype=float)
    gate_axis_u = np.asarray(plan["gate_axis_u"], dtype=float)
    gate_axis_v = np.asarray(plan["gate_axis_v"], dtype=float)
    opening_half_widths = np.asarray(plan["gate_opening_half_widths"], dtype=float)

    num_gates = gate_centers.shape[0]
    hu = opening_half_widths[:, 0]
    hv = opening_half_widths[:, 1]
    t_bar = float(plan.get("gate_bar_thickness", 0.10))
    half_depth = 0.5 * float(plan.get("gate_depth", 0.10))
    radius = float(plan.get("drone_radius", 0.10))
    zeros = np.zeros(num_gates, dtype=float)

    top_center_local = np.stack([zeros, hv + 0.5 * t_bar, zeros], axis=1)
    bottom_center_local = np.stack([zeros, -(hv + 0.5 * t_bar), zeros], axis=1)
    left_center_local = np.stack([-(hu + 0.5 * t_bar), zeros, zeros], axis=1)
    right_center_local = np.stack([hu + 0.5 * t_bar, zeros, zeros], axis=1)

    top_bottom_half_sizes = np.stack(
        [
            hu + t_bar,
            0.5 * t_bar * np.ones_like(hu),
            half_depth * np.ones_like(hu),
        ],
        axis=1,
    ) + radius

    left_right_half_sizes = np.stack(
        [
            0.5 * t_bar * np.ones_like(hu),
            hv,
            half_depth * np.ones_like(hu),
        ],
        axis=1,
    ) + radius

    def local_to_world(local_center):
        return (
            gate_centers
            + local_center[:, 0:1] * gate_axis_u
            + local_center[:, 1:2] * gate_axis_v
            + local_center[:, 2:3] * gate_normals
        )

    top_centers = local_to_world(top_center_local)
    bottom_centers = local_to_world(bottom_center_local)
    left_centers = local_to_world(left_center_local)
    right_centers = local_to_world(right_center_local)

    # Shape conventions below are [gate, bar, xyz/local-axis].
    centers = np.stack(
        [top_centers, bottom_centers, left_centers, right_centers],
        axis=1,
    )
    half_sizes = np.stack(
        [
            top_bottom_half_sizes,
            top_bottom_half_sizes,
            left_right_half_sizes,
            left_right_half_sizes,
        ],
        axis=1,
    )

    axis_u = np.repeat(gate_axis_u[:, None, :], 4, axis=1)
    axis_v = np.repeat(gate_axis_v[:, None, :], 4, axis=1)
    normals = np.repeat(gate_normals[:, None, :], 4, axis=1)

    return {
        "gate_centers": gate_centers,
        "centers": centers,
        "half_sizes": half_sizes,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "normals": normals,
        "bar_names": ("top", "bottom", "left", "right"),
    }


def nearest_gate_bar_direction_np(pos, gate_bar_geometry, eps=1e-12):
    """Return a unit XYZ direction from ``pos`` toward the nearest gate bar.

    Distance is measured to the surface/volume of each *inflated oriented box*,
    not merely to its center.  Therefore the selected obstacle is the same
    physical gate-bar geometry used to aim the rollout adversary.
    """
    pos = np.asarray(pos, dtype=float).reshape(3)
    centers = gate_bar_geometry["centers"]
    half_sizes = gate_bar_geometry["half_sizes"]
    axis_u = gate_bar_geometry["axis_u"]
    axis_v = gate_bar_geometry["axis_v"]
    normals = gate_bar_geometry["normals"]

    delta = pos[None, None, :] - centers
    q_u = np.sum(delta * axis_u, axis=-1)
    q_v = np.sum(delta * axis_v, axis=-1)
    q_n = np.sum(delta * normals, axis=-1)
    q = np.stack([q_u, q_v, q_n], axis=-1)

    # Closest point on/in each inflated oriented box.
    q_closest = np.clip(q, -half_sizes, half_sizes)
    closest_world = (
        centers
        + q_closest[..., 0:1] * axis_u
        + q_closest[..., 1:2] * axis_v
        + q_closest[..., 2:3] * normals
    )

    toward = closest_world - pos[None, None, :]
    distances = np.linalg.norm(toward, axis=-1)
    gate_idx, bar_idx = np.unravel_index(
        int(np.argmin(distances)),
        distances.shape,
    )

    distance = float(distances[gate_idx, bar_idx])
    closest_point = closest_world[gate_idx, bar_idx].copy()

    if distance <= eps:
        # We are already touching/inside the inflated gate-bar volume, so there
        # is no unique direction toward it; return zero rather than manufacture one.
        direction = np.zeros(3, dtype=float)
    else:
        direction = toward[gate_idx, bar_idx] / distance

    return direction, distance, int(gate_idx), int(bar_idx), closest_point


def plant_step(
    x,
    u,
    dt,
    gate_bar_geometry,
    last_gate_idx,
):
    """One actual closed-loop step with an XY velocity disturbance.

    The disturbance acts on vx and vy and points AWAY from the most recently
    passed physical gate.  Let p_g be that gate center and p be the drone
    position.  The XY uncertainty direction is

        d_xy = p_xy - p_g,xy
        w_xy = d_xy / ||d_xy||_2,

    whenever d_xy is nonzero.  Therefore ||w_xy||_2 = 1 in the nondegenerate
    case and ||w_xy||_2 = 0 at the gate center, so ||w||_2 <= 1 always.

    The state-dependent velocity perturbation is

        delta_v_xy = dt * DISTURBANCE_MAG * spatial_scale(px, py) * w_xy.

    No direct position or z-velocity disturbance is applied.
    """
    last_gate_idx = 4
    x = np.asarray(x, dtype=float)
    x_next = x + dt * quadrotor_xdot_np(x, u)

    gate_centers = np.asarray(
        gate_bar_geometry["gate_centers"],
        dtype=float,
    )

    if gate_centers.ndim != 2 or gate_centers.shape[1] != 3:
        raise ValueError(
            "gate_bar_geometry['gate_centers'] must have shape (num_gates, 3)."
        )

    last_gate_idx = int(np.clip(last_gate_idx, 0, len(gate_centers) - 1))
    last_gate_center = gate_centers[last_gate_idx]

    # Direction from the last gate center to the current drone position.
    # This is the geometric direction AWAY from the gate in the XY plane.
    away_xy = x[:2] - last_gate_center[:2]
    away_norm = float(np.linalg.norm(away_xy))

    if away_norm > 1e-12:
        w_xy = away_xy / away_norm
    else:
        w_xy = np.zeros(2, dtype=float)

    # Numerical safeguard: enforce ||w||_2 <= 1 even under roundoff.
    w_norm = float(np.linalg.norm(w_xy))
    if w_norm > 1.0:
        w_xy = w_xy / w_norm

    disturbance_direction = np.array(
        [w_xy[0], w_xy[1], 0.0],
        dtype=float,
    )

    # No direct position disturbance.
    position_delta = np.zeros(3, dtype=float)

    # State-dependent disturbance magnitude shared by vx and vy through one
    # normalized 2-D uncertainty vector.  The TOTAL XY magnitude is therefore
    # bounded by dt * DISTURBANCE_MAG * spatial_scale, not sqrt(2) times it.
    spatial_scale = float(disturbance_spatial_scale(x[0], x[1]))
    disturbance_magnitude = dt * DISTURBANCE_MAG * spatial_scale
    velocity_delta_xy = disturbance_magnitude * w_xy

    x_next[6] += dt * DISTURBANCE_MAG * spatial_scale / 2 / (2**0.5)
    x_next[7] += dt * DISTURBANCE_MAG * spatial_scale / (2**0.5)

    gate_distance_xy = float(np.linalg.norm(away_xy))

    return (
        x_next,
        position_delta,
        disturbance_direction,
        gate_distance_xy,
        last_gate_idx,
        -1,
        last_gate_center.copy(),
    )


# ============================================================
# GPU-SLS online tracking model / cost / constraints
# ============================================================

def disturbance_spatial_scale_jax(px, py, params=DISTURBANCE_FIELD):
    x_window = (
        0.5 * (1.0 + jnp.tanh(params.kx * (px - params.x_min)))
        * 0.5 * (1.0 - jnp.tanh(params.kx * (px - params.x_max)))
    )
    y_window = (
        0.5 * (1.0 + jnp.tanh(params.ky * (py - params.y_min)))
        * 0.5 * (1.0 - jnp.tanh(params.ky * (py - params.y_max)))
    )
    return x_window * y_window


def rotation_matrix_jax(phi, theta, psi):
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth, cth * sphi, cth * cphi],
    ], dtype=DTYPE)


def euler_angle_rates_matrix_jax(phi, theta):
    theta_safe = jnp.clip(theta, -PITCH_MAX, PITCH_MAX)
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta_safe)
    cth = jnp.cos(theta_safe)
    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi, -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=DTYPE)


def quadrotor_xdot_jax(x, u):
    phi, theta, psi = x[3], x[4], x[5]
    v = x[6:9]
    omega = x[9:12]
    thrust = u[0]
    tau = u[1:4]

    R = rotation_matrix_jax(phi, theta, psi)
    E = euler_angle_rates_matrix_jax(phi, theta)
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    pos_dot = v
    v_dot = (R @ (thrust * e3)) / MASS - GRAVITY * e3
    euler_dot = E @ omega
    omega_dot = J_INV_JAX @ (tau - jnp.cross(omega, J_JAX @ omega))

    return jnp.concatenate([pos_dot, euler_dot, v_dot, omega_dot])


def tracking_dynamics(x, u, t, *, parameter):
    """13-D dynamics: 12 physical states plus deterministic stage dt."""
    dt_nodes, active_nodes, motion_nodes, uref_nodes = parameter

    # Current physical integration step comes directly from the augmented state.
    dt = x[DT_IDX]
    physical_next = (
        x[:NX]
        + dt * quadrotor_xdot_jax(x[:NX], u)
    )

    # dt is not optimized online.  Force the next node's dt to equal the saved
    # offline schedule value for that stage.
    next_idx = jnp.minimum(
        t + 1,
        dt_nodes.shape[0] - 1,
    )
    dt_next = dt_nodes[next_idx]

    return jnp.concatenate([
        physical_next,
        jnp.reshape(dt_next, (1,)).astype(x.dtype),
    ])


def make_tracking_cost(horizon):
    H = int(horizon)

    def cost(W, reference, x, u, t, *, parameter):
        """Strong local tracking cost with non-state schedule data in parameter."""
        dt_nodes, active_nodes, motion_nodes, uref_nodes = parameter

        idx = jnp.minimum(t, reference.shape[0] - 1)
        xref = reference[idx]

        # Only the 12 physical states are tracked.  The augmented dt component
        # is deterministic schedule bookkeeping and has no tracking cost.
        dpos = x[0:3] - xref[0:3]
        dang = x[3:6] - xref[3:6]
        dvel = x[6:9] - xref[6:9]
        drate = x[9:12] - xref[9:12]

        uref_idx = jnp.minimum(t, uref_nodes.shape[0] - 1)
        du = u - uref_nodes[uref_idx]

        active = active_nodes[
            jnp.minimum(t, active_nodes.shape[0] - 1)
        ]
        motion_weight = motion_nodes[
            jnp.minimum(t, motion_nodes.shape[0] - 1)
        ]

        terminal_scale = jnp.where(
            t == H,
            TERMINAL_MULTIPLIER,
            1.0,
        )
        control_scale = jnp.where(
            t == H,
            0.0,
            1.0,
        )

        angle_cost = (
            Q_ANG[0] * (1.0 - jnp.cos(dang[0]))
            + Q_ANG[1] * (1.0 - jnp.cos(dang[1]))
            + Q_ANG[2] * (1.0 - jnp.cos(dang[2]))
        )

        pose_cost = (
            Q_POS[0] * dpos[0] ** 2
            + Q_POS[1] * dpos[1] ** 2
            + Q_POS[2] * dpos[2] ** 2
            + angle_cost
        )

        motion_cost = (
            Q_VEL[0] * dvel[0] ** 2
            + Q_VEL[1] * dvel[1] ** 2
            + Q_VEL[2] * dvel[2] ** 2
            + Q_RATE[0] * drate[0] ** 2
            + Q_RATE[1] * drate[1] ** 2
            + Q_RATE[2] * drate[2] ** 2
        )

        control_cost = (
            R_U[0] * du[0] ** 2
            + R_U[1] * du[1] ** 2
            + R_U[2] * du[2] ** 2
            + R_U[3] * du[3] ** 2
        )

        state_cost = terminal_scale * (
            pose_cost + motion_weight * motion_cost
        )

        return active * (
            state_cost + control_scale * control_cost
        )

    return cost


def make_tracking_state_constraints():
    x_min = jnp.asarray(X_MIN, dtype=DTYPE)
    x_max = jnp.asarray(X_MAX, dtype=DTYPE)

    def constraints(x, u, t):
        physical = x[:NX]
        g = jnp.concatenate([
            physical - x_max,
            x_min - physical,
        ])
        # Like the previous tracker, do not constrain the already-measured x0.
        return jnp.where(t == 0, -jnp.ones_like(g), g)

    return constraints


def make_tracking_gate_frame_constraints(
    gate_centers,
    gate_normals,
    gate_axis_u,
    gate_axis_v,
    opening_half_widths,
    bar_thickness=0.10,
    gate_depth=0.10,
    drone_radius=0.10,
):
    """Same four oriented gate-bar obstacle model as the GPU-SLS planner."""
    gate_centers = jnp.asarray(gate_centers, dtype=DTYPE)
    gate_normals = jnp.asarray(gate_normals, dtype=DTYPE)
    gate_axis_u = jnp.asarray(gate_axis_u, dtype=DTYPE)
    gate_axis_v = jnp.asarray(gate_axis_v, dtype=DTYPE)
    opening_half_widths = jnp.asarray(opening_half_widths, dtype=DTYPE)

    num_gates = gate_centers.shape[0]
    hu = opening_half_widths[:, 0]
    hv = opening_half_widths[:, 1]
    t_bar = jnp.asarray(bar_thickness, dtype=DTYPE)
    half_depth = jnp.asarray(0.5 * gate_depth, dtype=DTYPE)
    radius = jnp.asarray(drone_radius, dtype=DTYPE)
    zeros = jnp.zeros(num_gates, dtype=DTYPE)

    top_center_local = jnp.stack([zeros, hv + 0.5 * t_bar, zeros], axis=1)
    bottom_center_local = jnp.stack([zeros, -(hv + 0.5 * t_bar), zeros], axis=1)
    left_center_local = jnp.stack([-(hu + 0.5 * t_bar), zeros, zeros], axis=1)
    right_center_local = jnp.stack([hu + 0.5 * t_bar, zeros, zeros], axis=1)

    top_bottom_half_sizes = jnp.stack([
        hu + t_bar,
        0.5 * t_bar * jnp.ones_like(hu),
        half_depth * jnp.ones_like(hu),
    ], axis=1) + radius

    left_right_half_sizes = jnp.stack([
        0.5 * t_bar * jnp.ones_like(hu),
        hv,
        half_depth * jnp.ones_like(hu),
    ], axis=1) + radius

    def local_to_world(local_center):
        return (
            gate_centers
            + local_center[:, 0:1] * gate_axis_u
            + local_center[:, 1:2] * gate_axis_v
            + local_center[:, 2:3] * gate_normals
        )

    top_centers = local_to_world(top_center_local)
    bottom_centers = local_to_world(bottom_center_local)
    left_centers = local_to_world(left_center_local)
    right_centers = local_to_world(right_center_local)

    def outside_box_constraint(pos, box_centers, half_sizes):
        delta = pos[None, :] - box_centers
        q_u = jnp.sum(delta * gate_axis_u, axis=1)
        q_v = jnp.sum(delta * gate_axis_v, axis=1)
        q_n = jnp.sum(delta * gate_normals, axis=1)
        q = jnp.stack([q_u, q_v, q_n], axis=1)
        outside_margin = jnp.max(jnp.abs(q) - half_sizes, axis=1)
        return -outside_margin

    def constraints(x, u, t):
        pos = x[:3]
        top = outside_box_constraint(pos, top_centers, top_bottom_half_sizes)
        bottom = outside_box_constraint(pos, bottom_centers, top_bottom_half_sizes)
        left = outside_box_constraint(pos, left_centers, left_right_half_sizes)
        right = outside_box_constraint(pos, right_centers, left_right_half_sizes)
        return jnp.stack([top, bottom, left, right], axis=1).reshape(-1)

    return constraints


def make_tracking_disturbance(n, disturbance_mag):
    """Robust 13-D disturbance model with dt read from the augmented state.

    E has shape (13, 13).  The first 12 rows/columns correspond to the physical
    state, while the dt disturbance row/column remains zero.
    """
    mag = jnp.asarray(
        disturbance_mag,
        dtype=DTYPE,
    )

    def disturbance_at_state(x_k, t):
        dt = x_k[DT_IDX]

        spatial_scale = disturbance_spatial_scale_jax(
            x_k[0],
            x_k[1],
        )

        E_k = jnp.zeros(
            (n, n),
            dtype=x_k.dtype,
        )

        # Retained state-dependent vy uncertainty channel.
        E_k = E_k.at[7, 7].set(
            dt * mag * spatial_scale
        )

        # XY position uncertainty matching the rollout adversary.
        # There is intentionally no z-position uncertainty term.
        E_k = E_k.at[0, 0].set(dt * E_MAG_MODEL)
        E_k = E_k.at[1, 1].set(dt * E_MAG_MODEL)

        # E_k[DT_IDX, :] and E_k[:, DT_IDX] remain zero: dt is deterministic.
        return E_k

    def disturbance(X):
        ts = jnp.arange(X.shape[0])
        return jax.vmap(
            disturbance_at_state,
            in_axes=(0, 0),
        )(X, ts)

    disturbance.at_state = disturbance_at_state
    return disturbance


def _has_gate_geometry(plan):
    required = (
        "gate_centers",
        "gate_normals",
        "gate_axis_u",
        "gate_axis_v",
        "gate_opening_half_widths",
    )
    return all(
        plan.get(k) is not None
        for k in required
    )


def build_gpu_sls_mpc(
    horizon,
    X_in,
    U_in,
    plan,
    disturbance_mag,
    *,
    solver_mode="rti",
):
    """Build either the one-time full initializer or the online RTI controller.

    solver_mode == "initial_full":
        50 nominal SQP iterations followed by 50 SLS-enabled SQP iterations.

    solver_mode == "rti":
        no nominal initialization and a small fixed number of SQP iterations
        per MPC update.
    """
    H = int(horizon)

    if solver_mode == "initial_full":
        initialize_nominal = True
        max_initial_sqp_iterations = GPU_SLS_INITIAL_NOMINAL_SQP_ITERS
        # gpu_sqp.sqp loops over max_initial_sqp_iterations +
        # max_sqp_iterations, so this value is the robust/SLS phase count only.
        max_sqp_iterations = GPU_SLS_INITIAL_SLS_SQP_ITERS
        rti = False
        line_search = True
    elif solver_mode == "rti":
        initialize_nominal = False
        max_initial_sqp_iterations = 0
        max_sqp_iterations = GPU_SLS_RTI_SQP_ITERS
        rti = False
        line_search = False
    else:
        raise ValueError(
            f"Unknown solver_mode={solver_mode!r}; expected 'initial_full' or 'rti'."
        )

    W = jnp.ones((16,), dtype=DTYPE)
    cfg = MPCConfig(
        n=TRACK_NX,
        nu=NU,
        N=H,
        W=W,
        u_ref=jnp.array([T_HOVER, 0.0, 0.0, 0.0], dtype=DTYPE),
    )

    # Online MPC constraints: physical-state bounds + control bounds only.
    # Gate/bar obstacle constraints are intentionally NOT included in the solve.
    # Gate geometry is retained elsewhere only to aim the rollout disturbance
    # and for visualization; it has no effect on the MPC optimization.
    constraints = [
        make_tracking_state_constraints(),
        make_control_box_constraints(
            jnp.asarray(U_MIN, dtype=DTYPE),
            jnp.asarray(U_MAX, dtype=DTYPE),
        ),
    ]

    constraints_all = combine_constraints(*constraints)

    # GenericMPC obstacle interface is intentionally empty.
    obstacles = jnp.zeros((0, 3), dtype=DTYPE)
    disturbance = make_tracking_disturbance(TRACK_NX, disturbance_mag)

    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e2,
        max_iterations=GPU_SLS_ADMM_MAX_ITERS,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        slack_weight=1e3,
        enable_slack=True,
        scale_max=2.0,
        # num_phases=0, # This is weird
    )

    q_diag = jnp.concatenate([
        jnp.ones(NX, dtype=DTYPE),
        jnp.asarray([1e-6], dtype=DTYPE),
    ])
    Q_single = jnp.diag(q_diag)
    Q_bar = jnp.broadcast_to(
        Q_single,
        (H + 1, TRACK_NX, TRACK_NX),
    )
    R_bar = jnp.broadcast_to(jnp.eye(NU, dtype=DTYPE), (H, NU, NU))

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=initialize_nominal,
        max_initial_sqp_iterations=max_initial_sqp_iterations,
        warm_start=True,
        rti=rti,
        gradient_window=5,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=max_sqp_iterations,
        warm_start=True,
        # Keep the one-time initializer running through its requested nominal
        # and robust phases.  Online solves execute their small fixed budget.
        feas_tol=1e-6,
        step_tol=10,
        line_search=line_search,
        lm_regularization=1e-2,
    )

    return GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=tracking_dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=make_tracking_cost(H),
        disturbance=disturbance,
        shift=1,
        X_in=jnp.asarray(X_in, dtype=DTYPE),
        U_in=jnp.asarray(U_in, dtype=DTYPE),
        Q_bar=Q_bar,
        R_bar=R_bar,
    )


# ============================================================
# Offline optimized trajectory loading
# ============================================================

def load_offline_plan(filename):
    filename = Path(filename).expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(f"Offline GPU-SLS plan not found: {filename}")

    with np.load(filename, allow_pickle=False) as data:
        required = ["X", "U", "phase_times", "waypoint_steps"]
        for key in required:
            if key not in data:
                raise ValueError(f"{filename} is missing required key '{key}'")

        X_full = np.asarray(data["X"], dtype=float)
        U_plan = np.asarray(data["U"], dtype=float)
        phase_times = np.asarray(data["phase_times"], dtype=float).reshape(-1)
        waypoint_steps = np.asarray(data["waypoint_steps"], dtype=int).reshape(-1)

        def optional_array(key):
            return np.asarray(data[key], dtype=float) if key in data else None

        waypoint_centers = optional_array("waypoint_centers")
        waypoint_half_widths = optional_array("waypoint_half_widths")
        waypoint_axis_u = optional_array("waypoint_axis_u")
        waypoint_axis_v = optional_array("waypoint_axis_v")
        waypoint_normals = optional_array("waypoint_normals")
        gate_centers = optional_array("gate_centers")
        gate_normals = optional_array("gate_normals")
        gate_axis_u = optional_array("gate_axis_u")
        gate_axis_v = optional_array("gate_axis_v")
        gate_opening_half_widths = optional_array("gate_opening_half_widths")

        gate_bar_thickness = float(np.asarray(data["gate_bar_thickness"])) if "gate_bar_thickness" in data else 0.10
        gate_depth = float(np.asarray(data["gate_depth"])) if "gate_depth" in data else 0.10
        drone_radius = float(np.asarray(data["drone_radius"])) if "drone_radius" in data else 0.10

    if X_full.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got {X_full.shape}")
    if X_full.shape[1] < NX:
        raise ValueError(f"Expected at least {NX} state columns, got {X_full.shape[1]}")

    X_plan = X_full[:, :NX]
    N = X_plan.shape[0] - 1

    if U_plan.shape != (N, NU):
        raise ValueError(f"U must have shape {(N, NU)}, got {U_plan.shape}")
    if waypoint_steps[-1] != N:
        raise ValueError("Final waypoint step must equal the number of control intervals.")

    segment_lengths = np.concatenate([
        waypoint_steps[:1],
        waypoint_steps[1:] - waypoint_steps[:-1],
    ]).astype(float)

    if len(phase_times) != len(segment_lengths):
        raise ValueError("phase_times and waypoint_steps imply different numbers of phases.")

    dt_plan = np.empty(N, dtype=float)
    for k in range(N):
        phase_idx = int(np.searchsorted(waypoint_steps, k, side="right"))
        dt_plan[k] = phase_times[phase_idx] / segment_lengths[phase_idx]

    return {
        "filename": filename,
        "X_plan": X_plan,
        "U_plan": U_plan,
        "phase_times": phase_times,
        "waypoint_steps": waypoint_steps,
        "waypoint_centers": waypoint_centers,
        "waypoint_half_widths": waypoint_half_widths,
        "waypoint_axis_u": waypoint_axis_u,
        "waypoint_axis_v": waypoint_axis_v,
        "waypoint_normals": waypoint_normals,
        "gate_centers": gate_centers,
        "gate_normals": gate_normals,
        "gate_axis_u": gate_axis_u,
        "gate_axis_v": gate_axis_v,
        "gate_opening_half_widths": gate_opening_half_widths,
        "gate_bar_thickness": gate_bar_thickness,
        "gate_depth": gate_depth,
        "drone_radius": drone_radius,
        "segment_lengths": segment_lengths,
        "dt_plan": dt_plan,
        "N": N,
    }


# ============================================================
# Reference horizon extraction
# ============================================================

def local_reference_window(
    X_plan,
    U_plan,
    dt_plan,
    k,
    horizon,
):
    """
    Extract a fixed-size local reference window without allowing padding
    beyond the final offline node to alter the preceding MPC behavior.

    Real trajectory intervals:
        active = 1
        dt     = optimized offline dt
        cost   = normal tracking cost

    Intervals beyond the final trajectory node:
        active = 0
        dt     = 0
        cost   = 0

    Because dt = 0 on padded intervals,

        x[j+1] = x[j]

    and because active = 0, those intervals exert no tracking pressure on
    the earlier controls.  The fixed GPU-SLS problem size is preserved.
    """
    H = int(horizon)
    N = len(U_plan)

    if N < 1:
        raise ValueError("The offline plan must contain at least one control interval.")

    # Node N has no outgoing control interval.  If progress matching reaches N
    # before the goal tolerance is satisfied, keep the MPC rooted at N - 1.
    # Otherwise its model receives an all-zero dt horizon while the plant still
    # advances using dt_plan[N - 1], destroying the warm-start consistency.
    k = int(np.clip(k, 0, N - 1))

    X_ref = np.empty(
        (H + 1, NX),
        dtype=float,
    )

    U_ref = np.empty(
        (H, NU),
        dtype=float,
    )

    dt_ref = np.zeros(
        H,
        dtype=float,
    )

    stage_active = np.zeros(
        H,
        dtype=float,
    )

    # State reference is still padded with the final state for a sensible
    # warm start.  Padded stage costs are masked out below.
    for j in range(H + 1):
        idx = min(
            k + j,
            N,
        )
        X_ref[j] = X_plan[idx]

    for j in range(H):
        idx = k + j

        if idx < N:
            U_ref[j] = U_plan[idx]
            dt_ref[j] = dt_plan[idx]
            stage_active[j] = 1.0
        else:
            # This is not a real part of the trajectory.  Keep a harmless
            # control reference for the warm start, but freeze dynamics and
            # completely remove the stage from the tracking objective.
            U_ref[j] = np.array(
                [T_HOVER, 0.0, 0.0, 0.0],
                dtype=float,
            )
            dt_ref[j] = 0.0
            stage_active[j] = 0.0

    # If the fixed MPC horizon extends beyond the saved trajectory, do not
    # ask the artificial terminal stage to match final velocity/rates.  It
    # still tracks final position/orientation, so the finish remains spatially
    # meaningful without imposing a stop merely because the file ends.
    terminal_motion_weight = (
        1.0 if (k + H) <= N else 0.0
    )

    return (
        X_ref,
        U_ref,
        dt_ref,
        stage_active,
        terminal_motion_weight,
    )


# ============================================================
# Solve one MPC step
# ============================================================

def build_augmented_local_reference(
    X_ref_np,
    U_ref_np,
    dt_ref_np,
    stage_active_np,
    terminal_motion_weight_np,
):
    """Build the 13-D reference trajectory and runtime stage parameters."""
    H = len(U_ref_np)

    # dt is both (i) embedded as state 12 and (ii) retained in parameter so the
    # dynamics can deterministically assign x_{k+1}[DT_IDX] = dt_nodes[k+1].
    dt_nodes = np.zeros(
        H + 1,
        dtype=float,
    )
    dt_nodes[:H] = dt_ref_np
    dt_nodes[H] = 0.0

    active_nodes = np.zeros(
        H + 1,
        dtype=float,
    )
    active_nodes[:H] = stage_active_np
    active_nodes[H] = 1.0

    motion_nodes = np.zeros(
        H + 1,
        dtype=float,
    )
    motion_nodes[:H] = stage_active_np
    motion_nodes[H] = float(
        terminal_motion_weight_np
    )

    uref_nodes = np.empty(
        (H + 1, NU),
        dtype=float,
    )
    uref_nodes[:H] = U_ref_np

    if H > 0:
        uref_nodes[H] = U_ref_np[-1]
    else:
        uref_nodes[H] = np.array(
            [T_HOVER, 0.0, 0.0, 0.0],
            dtype=float,
        )

    X_aug_ref = np.concatenate([
        np.asarray(X_ref_np, dtype=float),
        dt_nodes[:, None],
    ], axis=1)

    parameter = (
        jnp.asarray(dt_nodes, dtype=DTYPE),
        jnp.asarray(active_nodes, dtype=DTYPE),
        jnp.asarray(motion_nodes, dtype=DTYPE),
        jnp.asarray(uref_nodes, dtype=DTYPE),
    )

    return X_aug_ref, parameter


def compute_tracking_objective_np(X_pred, U_pred, X_ref, U_ref, stage_active, terminal_motion_weight):
    H = len(U_ref)
    total = 0.0
    for j in range(H):
        if stage_active[j] <= 0.0:
            continue
        x = X_pred[j, :NX]
        xr = X_ref[j]
        dpos = x[:3] - xr[:3]
        dang = x[3:6] - xr[3:6]
        dvel = x[6:9] - xr[6:9]
        drate = x[9:12] - xr[9:12]
        du = U_pred[j] - U_ref[j]
        total += float(np.dot(Q_POS, dpos ** 2))
        total += float(np.sum(Q_ANG * (1.0 - np.cos(dang))))
        total += float(np.dot(Q_VEL, dvel ** 2))
        total += float(np.dot(Q_RATE, drate ** 2))
        total += float(np.dot(R_U, du ** 2))

    # A diagnostic terminal term matching the old tracker semantics.
    x = X_pred[-1, :NX]
    xr = X_ref[-1]
    dpos = x[:3] - xr[:3]
    dang = x[3:6] - xr[3:6]
    dvel = x[6:9] - xr[6:9]
    drate = x[9:12] - xr[9:12]
    total += TERMINAL_MULTIPLIER * (
        float(np.dot(Q_POS, dpos ** 2))
        + float(np.sum(Q_ANG * (1.0 - np.cos(dang))))
        + float(terminal_motion_weight) * (
            float(np.dot(Q_VEL, dvel ** 2)) + float(np.dot(Q_RATE, drate ** 2))
        )
    )
    return total


def solve_mpc_step(
    controller,
    x_current,
    X_ref_np,
    U_ref_np,
    dt_ref_np,
    stage_active_np,
    terminal_motion_weight_np,
    reset_warm_start=False,
):
    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(NX)

    if not np.all(np.isfinite(x_current)):
        raise ValueError(
            "Measured state contains NaN or Inf."
        )

    X_aug_ref, parameter = build_augmented_local_reference(
        X_ref_np=X_ref_np,
        U_ref_np=U_ref_np,
        dt_ref_np=dt_ref_np,
        stage_active_np=stage_active_np,
        terminal_motion_weight_np=terminal_motion_weight_np,
    )

    # Measured physical state replaces the reference physical state at node 0.
    # The node-0 dt remains the saved dt schedule value.
    x0_aug = X_aug_ref[0].copy()
    x0_aug[:NX] = x_current

    if reset_warm_start:
        X_reset = X_aug_ref.copy()
        X_reset[0, :NX] = x_current

        controller.reset(
            jnp.asarray(
                X_reset,
                dtype=DTYPE,
            ),
            jnp.asarray(
                U_ref_np,
                dtype=DTYPE,
            ),
        )

    tic = time.perf_counter()

    result = controller.run(
        x0=jnp.asarray(
            x0_aug,
            dtype=DTYPE,
        ),
        reference=jnp.asarray(
            X_aug_ref,
            dtype=DTYPE,
        ),
        parameter=parameter,
    )

    if len(result) != 7:
        raise RuntimeError(
            "Unexpected GenericMPC.run return length: "
            f"{len(result)}"
        )

    (
        u0,
        X_pred,
        U_pred,
        V_pred,
        backoffs,
        Phi_x,
        Phi_u,
    ) = result

    X_pred = jax.block_until_ready(X_pred)
    U_pred = jax.block_until_ready(U_pred)
    solve_time = time.perf_counter() - tic

    # --------------------------------------------------------
    # Terminal robust SLS tube diagnostic.
    #
    # Phi_x[k, j] maps the j-th unit-2-norm disturbance w_j
    # into the state deviation at node k.  Therefore, for each
    # state coordinate i, an exact coordinate-wise terminal
    # half-width is
    #
    #     r_i = sum_j || Phi_x[N, j, i, :] ||_2.
    #
    # We print the XYZ half-widths and the Euclidean radius of
    # the resulting XYZ bounding box.  The latter is a safe
    # scalar bound on terminal position deviation.
    # --------------------------------------------------------
    Phi_x_np = np.asarray(
        jax.block_until_ready(Phi_x),
        dtype=float,
    )
    if not np.all(np.isfinite(Phi_x_np)):
        raise RuntimeError("SLS response Phi_x contains NaN or Inf.")

    # Coordinate-wise robust tube half-width at EVERY prediction node.
    #
    # Phi_x[k, j, i, :] maps the j-th unit-2-norm disturbance into state
    # coordinate i at prediction node k. Summing the row 2-norms over all
    # disturbance times gives the exact coordinate-wise Minkowski-sum
    # half-width used for plotting an axis-aligned tube around X_pred.
    tube_halfwidth = np.sum(
        np.linalg.norm(Phi_x_np, axis=-1),
        axis=1,
    )  # [prediction_node, state]
    tube_xyz_halfwidth = tube_halfwidth[:, :3]

    terminal_xyz_halfwidth = tube_xyz_halfwidth[-1]
    terminal_xyz_radius_bound = float(
        np.linalg.norm(terminal_xyz_halfwidth)
    )

    print(
        "Terminal SLS tube: "
        f"xyz_halfwidth=[{terminal_xyz_halfwidth[0]:.6f}, "
        f"{terminal_xyz_halfwidth[1]:.6f}, "
        f"{terminal_xyz_halfwidth[2]:.6f}] m  "
        f"xyz_radius_bound={terminal_xyz_radius_bound:.6f} m"
    )

    X_pred_np = np.asarray(
        X_pred,
        dtype=float,
    )
    U_pred_np = np.asarray(
        U_pred,
        dtype=float,
    )

    if not np.all(np.isfinite(X_pred_np)):
        raise RuntimeError("MPC produced NaN or Inf in its predicted state trajectory.")
    if not np.all(np.isfinite(U_pred_np)):
        raise RuntimeError("MPC produced NaN or Inf in its predicted control trajectory.")

    u0_np = U_pred_np[0]
    if np.any(u0_np < U_MIN - CONTROL_BOUND_TOL) or np.any(
        u0_np > U_MAX + CONTROL_BOUND_TOL
    ):
        raise RuntimeError(
            "MPC rejected an out-of-bounds first control: "
            f"u0={u0_np}, bounds=[{U_MIN}, {U_MAX}]."
        )

    objective = compute_tracking_objective_np(
        X_pred_np,
        U_pred_np,
        X_ref_np,
        U_ref_np,
        stage_active_np,
        terminal_motion_weight_np,
    )

    if not np.isfinite(objective):
        raise RuntimeError("MPC produced a nonfinite tracking objective.")

    # Return the full nominal prediction and its XYZ robust tube so the online
    # rollout can save both for later trajectory/tube visualization.
    return (
        X_pred_np,
        U_pred_np,
        tube_xyz_halfwidth,
        objective,
        solve_time,
    )


# ============================================================
# Warm-start horizon shift

# ============================================================
# Warm-start horizon shift
# ============================================================

def shift_warm_start(
    X_sol,
    U_sol,
):
    X_guess = np.empty_like(X_sol)
    U_guess = np.empty_like(U_sol)

    X_guess[:-1] = X_sol[1:]
    X_guess[-1] = X_sol[-1]

    U_guess[:-1] = U_sol[1:]
    U_guess[-1] = U_sol[-1]

    return (
        X_guess,
        U_guess,
    )


# ============================================================
# Monotonic nearest-point progress
# ============================================================

def find_monotonic_progress_index(
    x_current,
    X_plan,
    previous_progress_idx,
    search_window=DEFAULT_PROGRESS_SEARCH_WINDOW,
):
    """
    Find the nearest trajectory position while allowing progress
    to move only forward.

    Search set:

        i in [previous_progress_idx,
              previous_progress_idx + search_window]

    clipped to the end of the offline plan.

    Therefore:

        progress_{k+1} >= progress_k

    by construction.

    Only XYZ position is used for progress matching. The full
    state is still tracked by the GPU-SLS MPC cost.
    """
    N = X_plan.shape[0] - 1

    start = int(
        np.clip(
            previous_progress_idx,
            0,
            N,
        )
    )

    end = int(
        min(
            start + int(search_window),
            N,
        )
    )

    candidates = X_plan[
        start:end + 1,
        :3,
    ]

    distances = np.linalg.norm(
        candidates
        - np.asarray(x_current[:3])[None, :],
        axis=1,
    )

    relative_idx = int(
        np.argmin(distances)
    )

    progress_idx = (
        start + relative_idx
    )

    nearest_distance = float(
        distances[relative_idx]
    )

    return (
        progress_idx,
        nearest_distance,
    )


# ============================================================
# Closed-loop MPC rollout
# ============================================================

def run_tracking(
    plan,
    horizon=DEFAULT_MPC_HORIZON,
    disturbance_mag=DISTURBANCE_MAG,
    initial_offset=None,
    progress_search_window=DEFAULT_PROGRESS_SEARCH_WINDOW,
    goal_tol=DEFAULT_GOAL_TOL,
    max_control_steps=None,
):
    """Closed-loop GPU-SLS tracking with an XY velocity adversary.

    At each real plant step, the disturbance points in the XY direction away
    from the most recently passed physical gate, uses a normalized uncertainty
    direction with ||w||_2 <= 1, and applies

        delta_v_xy = dt * DISTURBANCE_MAG * spatial_scale(px, py) * w_xy.

    No direct position or z-velocity disturbance is applied. Gate geometry is
    used only to define the away-from-gate direction. The rollout does not
    perform collision detection and never terminates because of gates.
    """
    X_plan = plan["X_plan"]
    U_plan = plan["U_plan"]
    dt_plan = plan["dt_plan"]
    N = plan["N"]

    gate_bar_geometry = build_gate_bar_geometry_np(plan)

    # Each physical gate has a pre- and post-gate waypoint.  A gate counts as
    # "passed" once monotonic trajectory progress reaches its post-gate step.
    waypoint_steps = np.asarray(plan["waypoint_steps"], dtype=int).reshape(-1)
    post_gate_steps = waypoint_steps[1::2]
    num_gates = len(gate_bar_geometry["gate_centers"])
    if len(post_gate_steps) != num_gates:
        raise ValueError(
            "Expected exactly two waypoint steps (pre/post) per physical gate."
        )

    if progress_search_window < 1:
        raise ValueError(
            "progress_search_window must be >= 1"
        )

    if max_control_steps is None:
        max_control_steps = (
            DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER
            * N
        )

    max_control_steps = int(max_control_steps)
    if max_control_steps < 1:
        raise ValueError(
            "max_control_steps must be >= 1"
        )

    controller = None
    initial_full_solve_time = np.nan

    X_closed_loop = np.zeros(
        (max_control_steps + 1, NX),
        dtype=float,
    )
    U_closed_loop = np.zeros(
        (max_control_steps, NU),
        dtype=float,
    )
    objective_history = np.full(
        max_control_steps,
        np.nan,
        dtype=float,
    )
    solve_times = np.full(
        max_control_steps,
        np.nan,
        dtype=float,
    )
    fallback_used = np.zeros(
        max_control_steps,
        dtype=bool,
    )

    # Save the online nominal MPC prediction and robust XYZ tube at every
    # controller step. Entries remain NaN when that step falls back because no
    # accepted MPC solution exists for that frame.
    X_mpc_nominal_history = np.full(
        (max_control_steps, horizon + 1, TRACK_NX),
        np.nan,
        dtype=float,
    )
    U_mpc_nominal_history = np.full(
        (max_control_steps, horizon, NU),
        np.nan,
        dtype=float,
    )
    tube_xyz_halfwidth_history = np.full(
        (max_control_steps, horizon + 1, 3),
        np.nan,
        dtype=float,
    )

    # Actual discrete position perturbation applied at each rollout step.
    disturbance_xyz_history = np.zeros(
        (max_control_steps, 3),
        dtype=float,
    )
    disturbance_direction_history = np.zeros(
        (max_control_steps, 3),
        dtype=float,
    )
    closest_obstacle_distance_history = np.full(
        max_control_steps,
        np.nan,
        dtype=float,
    )
    closest_gate_index_history = np.full(
        max_control_steps,
        -1,
        dtype=int,
    )
    closest_bar_index_history = np.full(
        max_control_steps,
        -1,
        dtype=int,
    )
    closest_obstacle_point_history = np.full(
        (max_control_steps, 3),
        np.nan,
        dtype=float,
    )

    dt_applied_history = np.zeros(
        max_control_steps,
        dtype=float,
    )
    progress_idx_history = np.zeros(
        max_control_steps + 1,
        dtype=int,
    )
    nearest_distance_history = np.full(
        max_control_steps + 1,
        np.nan,
        dtype=float,
    )

    x_current = X_plan[0].copy()
    if initial_offset is not None:
        x_current += np.asarray(
            initial_offset,
            dtype=float,
        )
    X_closed_loop[0] = x_current

    progress_idx = 0
    (
        progress_idx,
        nearest_distance,
    ) = find_monotonic_progress_index(
        x_current=x_current,
        X_plan=X_plan,
        previous_progress_idx=progress_idx,
        search_window=progress_search_window,
    )
    progress_idx_history[0] = progress_idx
    nearest_distance_history[0] = nearest_distance


    print(
        "\n================ GPU-SLS MPC TRACKING ==============="
    )
    print(f"Offline GPU-SLS plan: {plan['filename']}")
    print(f"Offline plan intervals: {N}")
    print(f"MPC horizon: {horizon}")
    print("Progress mode: monotonic nearest point")
    print(f"Progress search window: {progress_search_window} nodes")
    print(f"Goal tolerance: {goal_tol:.3f} m")
    print(f"Max controller steps: {max_control_steps}")
    print(
        "Plant adversary: XY position disturbance toward closest gate bar (z delta = 0)"
    )
    print(
        f"E_MAG_MODEL: {E_MAG_MODEL:.6f}  "
        "(delta_p = dt * E_MAG_MODEL * [w_x, w_y, 0], ||w_xy||_2 <= 1)"
    )
    print(
        f"Robust-model state-dependent vy magnitude: {disturbance_mag}"
    )
    print("Gate-bar collision detection: DISABLED")
    print(
        "First MPC solve: "
        f"{GPU_SLS_INITIAL_NOMINAL_SQP_ITERS} nominal SQP + "
        f"{GPU_SLS_INITIAL_SLS_SQP_ITERS} SLS SQP iterations"
    )
    print(
        f"Subsequent MPC solves: RTI, {GPU_SLS_RTI_SQP_ITERS} SQP iterations per step"
    )
    print("=====================================================")

    num_control_steps = 0
    reached_goal = False
    previous_reference_progress = progress_idx

    for control_step in range(max_control_steps):

        (
            progress_idx,
            nearest_distance,
        ) = find_monotonic_progress_index(
            x_current=x_current,
            X_plan=X_plan,
            previous_progress_idx=progress_idx,
            search_window=progress_search_window,
        )

        progress_advance = (
            progress_idx
            - previous_reference_progress
        )

        final_pos_error_before = float(
            np.linalg.norm(
                x_current[:3]
                - X_plan[-1, :3]
            )
        )
        if (
            progress_idx >= N
            and final_pos_error_before <= goal_tol
        ):
            reached_goal = True
            break

        # There is no interval N.  Keep optimizing the last real interval if
        # nearest-point matching reaches the final node before the vehicle is
        # actually within the goal tolerance.
        mpc_progress_idx = min(progress_idx, N - 1)

        (
            X_ref,
            U_ref,
            dt_ref,
            stage_active,
            terminal_motion_weight,
        ) = local_reference_window(
            X_plan=X_plan,
            U_plan=U_plan,
            dt_plan=dt_plan,
            k=mpc_progress_idx,
            horizon=horizon,
        )

        dt_apply = float(
            dt_plan[mpc_progress_idx]
        )

        reset_warm_start = (
            control_step > 0
            and (
                mpc_progress_idx
                - min(previous_reference_progress, N - 1)
            ) != 1
        )

        X_aug_ref, _stage_parameter = build_augmented_local_reference(
            X_ref_np=X_ref,
            U_ref_np=U_ref,
            dt_ref_np=dt_ref,
            stage_active_np=stage_active,
            terminal_motion_weight_np=terminal_motion_weight,
        )

        if controller is None:
            X_init = X_aug_ref.copy()
            X_init[0, :NX] = x_current

            controller = build_gpu_sls_mpc(
                horizon=horizon,
                X_in=X_init,
                U_in=U_ref,
                plan=plan,
                disturbance_mag=disturbance_mag,
                solver_mode="initial_full",
            )
            reset_warm_start = False
            print(
                "\nRunning one-time full GPU-SLS initialization: "
                f"{GPU_SLS_INITIAL_NOMINAL_SQP_ITERS} nominal SQP + "
                f"{GPU_SLS_INITIAL_SLS_SQP_ITERS} SLS SQP iterations..."
            )

        try:
            (
                X_mpc,
                U_mpc,
                tube_xyz_halfwidth,
                objective,
                solve_time,
            ) = solve_mpc_step(
                controller=controller,
                x_current=x_current,
                X_ref_np=X_ref,
                U_ref_np=U_ref,
                dt_ref_np=dt_ref,
                stage_active_np=stage_active,
                terminal_motion_weight_np=terminal_motion_weight,
                reset_warm_start=reset_warm_start,
            )

            # Remove harmless line-search roundoff at an active box bound.  Any
            # material violation was rejected in solve_mpc_step above.
            u_apply = np.clip(U_mpc[0], U_MIN, U_MAX)
            objective_history[control_step] = objective
            solve_times[control_step] = solve_time
            X_mpc_nominal_history[control_step] = X_mpc
            U_mpc_nominal_history[control_step] = U_mpc
            tube_xyz_halfwidth_history[control_step] = tube_xyz_halfwidth

            if control_step == 0:
                initial_full_solve_time = solve_time

                # Seed RTI directly with the fully optimized 13-D horizon.
                X_seed = X_mpc.copy()

                controller = build_gpu_sls_mpc(
                    horizon=horizon,
                    X_in=X_seed,
                    U_in=U_mpc,
                    plan=plan,
                    disturbance_mag=disturbance_mag,
                    solver_mode="rti",
                )

                print(
                    f"Initial full solve completed in {1000.0 * solve_time:.3f} ms. "
                    "Switching permanently to "
                    f"{GPU_SLS_RTI_SQP_ITERS}-SQP RTI updates."
                )

        except RuntimeError as error:
            print(
                f"\nMPC solve failed at controller step "
                f"{control_step}; using offline feedforward "
                f"control at progress node {mpc_progress_idx}."
            )
            print(error)
            u_apply = U_ref[0].copy()
            fallback_used[control_step] = True

            # controller.run() updates its internal warm start before the
            # validation above.  Reset it so a rejected trajectory cannot
            # poison every subsequent RTI solve.
            X_reset = X_aug_ref.copy()
            X_reset[0, :NX] = x_current
            controller.reset(
                jnp.asarray(X_reset, dtype=DTYPE),
                jnp.asarray(U_ref, dtype=DTYPE),
            )

            if control_step == 0:
                # The full initializer failed. Continue the experiment in RTI mode
                # from the current reference warm start rather than rerunning the
                # expensive initialization on the next controller tick.
                X_seed = X_aug_ref.copy()
                X_seed[0, :NX] = x_current

                controller = build_gpu_sls_mpc(
                    horizon=horizon,
                    X_in=X_seed,
                    U_in=U_ref,
                    plan=plan,
                    disturbance_mag=disturbance_mag,
                    solver_mode="rti",
                )
                print(
                    "Initial full solve failed; continuing with "
                    f"{GPU_SLS_RTI_SQP_ITERS}-SQP RTI mode."
                )

        previous_reference_progress = progress_idx
        U_closed_loop[control_step] = u_apply
        dt_applied_history[control_step] = dt_apply

        # ----------------------------------------------------
        # Apply the state-dependent XY VELOCITY disturbance in the direction
        # pointing away from the most recently passed physical gate.
        #
        # Before Gate 1 has been passed, use Gate 1 as the reference gate.
        # Thereafter use the most recent gate whose post-gate waypoint has
        # been reached by monotonic trajectory progress.
        # ----------------------------------------------------
        last_gate_idx = int(
            np.searchsorted(
                post_gate_steps,
                progress_idx,
                side="right",
            ) - 1
        )
        last_gate_idx = int(np.clip(last_gate_idx, 0, num_gates - 1))

        (
            x_current,
            disturbance_xyz,
            disturbance_direction,
            closest_obstacle_distance,
            closest_gate_idx,
            closest_bar_idx,
            closest_obstacle_point,
        ) = plant_step(
            x=x_current,
            u=u_apply,
            dt=dt_apply,
            gate_bar_geometry=gate_bar_geometry,
            last_gate_idx=last_gate_idx,
        )

        if not np.all(np.isfinite(x_current)):
            raise FloatingPointError(
                "Plant state became nonfinite after controller step "
                f"{control_step}; u={u_apply}, dt={dt_apply}."
            )

        disturbance_xyz_history[control_step] = disturbance_xyz
        disturbance_direction_history[control_step] = disturbance_direction
        closest_obstacle_distance_history[control_step] = closest_obstacle_distance
        closest_gate_index_history[control_step] = closest_gate_idx
        closest_bar_index_history[control_step] = closest_bar_idx
        closest_obstacle_point_history[control_step] = closest_obstacle_point

        num_control_steps = control_step + 1
        X_closed_loop[num_control_steps] = x_current

        (
            next_progress_idx,
            next_nearest_distance,
        ) = find_monotonic_progress_index(
            x_current=x_current,
            X_plan=X_plan,
            previous_progress_idx=progress_idx,
            search_window=progress_search_window,
        )
        progress_idx = next_progress_idx
        progress_idx_history[num_control_steps] = progress_idx
        nearest_distance_history[num_control_steps] = next_nearest_distance

        reference_pos_error = float(
            np.linalg.norm(
                x_current[:3]
                - X_plan[progress_idx, :3]
            )
        )
        final_pos_error = float(
            np.linalg.norm(
                x_current[:3]
                - X_plan[-1, :3]
            )
        )

        if np.isfinite(solve_times[control_step]):
            solve_string = (
                f"{1000.0 * solve_times[control_step]:9.3f} ms"
            )
        else:
            solve_string = " fallback"

        print(
            f"MPC {control_step:03d}: "
            f"progress={progress_idx:03d}/{N:03d}  "
            f"advance={progress_advance:+d}  "
            f"nearest_err={reference_pos_error:.4f} m  "
            f"final_err={final_pos_error:.4f} m  "
            f"solve={solve_string}  "
            f"dxyz=[{disturbance_xyz[0]:+.4f},"
            f"{disturbance_xyz[1]:+.4f},"
            f"{disturbance_xyz[2]:+.4f}]  "
            f"away_from=G{closest_gate_idx + 1}  "
            f"obs_dist={closest_obstacle_distance:.4f} m  "
            f"|u|={np.linalg.norm(u_apply):.3f}"
        )

        if (
            progress_idx >= N
            and final_pos_error <= goal_tol
        ):
            reached_goal = True
            break


    X_closed_loop = X_closed_loop[:num_control_steps + 1]
    U_closed_loop = U_closed_loop[:num_control_steps]
    objective_history = objective_history[:num_control_steps]
    # Preserve the original behavior of excluding the first JIT/compile solve
    # from timing statistics.
    solve_times = solve_times[1:num_control_steps]
    fallback_used = fallback_used[:num_control_steps]
    X_mpc_nominal_history = X_mpc_nominal_history[:num_control_steps]
    U_mpc_nominal_history = U_mpc_nominal_history[:num_control_steps]
    tube_xyz_halfwidth_history = tube_xyz_halfwidth_history[:num_control_steps]
    disturbance_xyz_history = disturbance_xyz_history[:num_control_steps]
    disturbance_direction_history = disturbance_direction_history[:num_control_steps]
    closest_obstacle_distance_history = closest_obstacle_distance_history[:num_control_steps]
    closest_gate_index_history = closest_gate_index_history[:num_control_steps]
    closest_bar_index_history = closest_bar_index_history[:num_control_steps]
    closest_obstacle_point_history = closest_obstacle_point_history[:num_control_steps]
    dt_applied_history = dt_applied_history[:num_control_steps]
    progress_idx_history = progress_idx_history[:num_control_steps + 1]
    nearest_distance_history = nearest_distance_history[:num_control_steps + 1]

    X_progress_reference = X_plan[
        progress_idx_history
    ]
    pos_errors = np.linalg.norm(
        X_closed_loop[:, :3]
        - X_progress_reference[:, :3],
        axis=1,
    )
    state_errors = np.linalg.norm(
        X_closed_loop
        - X_progress_reference,
        axis=1,
    )
    final_pos_error = float(
        np.linalg.norm(
            X_closed_loop[-1, :3]
            - X_plan[-1, :3]
        )
    )

    valid_solve_times = solve_times[
        np.isfinite(solve_times)
    ]
    rti_solve_times = solve_times[1:][
        np.isfinite(solve_times[1:])
    ]

    rollout_duration = float(np.sum(dt_applied_history))

    print(
        "\n================ TRACKING RESULT ====================="
    )
    print(f"Reached final goal: {reached_goal}")
    print(f"Rollout duration [s]: {rollout_duration:.6f}")
    print(f"Controller steps executed: {num_control_steps}")
    print(
        f"Final progress index: "
        f"{progress_idx_history[-1]}/{N}"
    )
    print(
        f"Final position error [m]: "
        f"{final_pos_error:.6f}"
    )
    print(
        f"Max progress-matched position error [m]: "
        f"{np.max(pos_errors):.6f}"
    )
    print(
        f"Mean progress-matched position error [m]: "
        f"{np.mean(pos_errors):.6f}"
    )

    if np.isfinite(initial_full_solve_time):
        print(
            "Initial full GPU-SLS solve [ms]: "
            f"{1000.0 * initial_full_solve_time:.3f}"
        )

    if len(rti_solve_times) > 0:
        print(
            "Mean GPU-SLS RTI solve [ms]: "
            f"{1000.0 * np.mean(rti_solve_times):.3f}"
        )
        print(
            "Max GPU-SLS RTI solve [ms]:  "
            f"{1000.0 * np.max(rti_solve_times):.3f}"
        )
    elif len(valid_solve_times) > 0:
        print(
            "Only the initialization solve was completed; no RTI timing samples yet."
        )

    print(
        f"Fallback controls used: "
        f"{np.sum(fallback_used)}"
    )
    if not reached_goal:
        print(
            "WARNING: maximum controller steps reached "
            "before satisfying the final goal tolerance."
        )

    print(
        "======================================================"
    )

    return {
        "X_closed_loop": X_closed_loop,
        "U_closed_loop": U_closed_loop,
        "objective_history": objective_history,
        "solve_times": solve_times,
        "initial_full_solve_time": (
            float(initial_full_solve_time)
            if np.isfinite(initial_full_solve_time)
            else np.nan
        ),
        "fallback_used": fallback_used,
        "X_mpc_nominal_history": X_mpc_nominal_history,
        "U_mpc_nominal_history": U_mpc_nominal_history,
        "tube_xyz_halfwidth_history": tube_xyz_halfwidth_history,
        "pos_errors": pos_errors,
        "state_errors": state_errors,
        "disturbance_xyz_history": disturbance_xyz_history,
        "disturbance_direction_history": disturbance_direction_history,
        "closest_obstacle_distance_history": closest_obstacle_distance_history,
        "closest_gate_index_history": closest_gate_index_history,
        "closest_bar_index_history": closest_bar_index_history,
        "closest_obstacle_point_history": closest_obstacle_point_history,
        "dt_applied_history": dt_applied_history,
        "progress_idx_history": progress_idx_history,
        "nearest_distance_history": nearest_distance_history,
        "X_progress_reference": X_progress_reference,
        "reached_goal": reached_goal,
        "num_control_steps": num_control_steps,
    }


# ============================================================
# Visualization
# ============================================================

def plot_tracking_top_down(
    X_plan,
    X_closed_loop,
    waypoint_centers=None,
    waypoint_half_widths=None,
    waypoint_axis_u=None,
    waypoint_normals=None,
    gate_centers=None,
    gate_normals=None,
    gate_axis_u=None,
    gate_opening_half_widths=None,
    gate_bar_thickness=0.10,
    gate_depth=0.10,
    disturbance_mag=0.0,
    save_path="multiphase_trajectory_tracking_topdown.png",
):
    """Plot the closed-loop tracker using the planner/animation geometry.

    Physical gates are rendered as oriented top-down footprints using the gate
    local u/n axes. Waypoint boxes are rendered in their oriented local u/n
    frames rather than as axis-aligned XY rectangles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle, Polygon

    X_plan = np.asarray(X_plan, dtype=float)
    X_closed_loop = np.asarray(X_closed_loop, dtype=float)

    x_min = -6.0
    x_max = 5.0
    y_min = -7.5
    y_max = 8.0

    fig, ax = plt.subplots(figsize=(9, 8))

    # ==================================================
    # Disturbance-field background -- same presentation
    # as the planning/animation renderer.
    # ==================================================
    display_mag = (
        float(disturbance_mag)
        if disturbance_mag > 0.0
        else 2.5
    )

    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    XX, YY = np.meshgrid(xs, ys)

    field_values = (
        display_mag
        * disturbance_spatial_scale(XX, YY)
    )

    magma = plt.get_cmap("magma")
    cmap = LinearSegmentedColormap.from_list(
        "white_to_magma",
        [
            (0.00, "white"),
            (0.08, magma(0.18)),
            (0.35, magma(0.40)),
            (0.65, magma(0.68)),
            (1.00, magma(1.00)),
        ],
    )

    field = ax.contourf(
        XX,
        YY,
        field_values,
        levels=np.linspace(0.0, display_mag, 100),
        cmap=cmap,
        vmin=0.0,
        vmax=display_mag,
        alpha=0.80,
        zorder=0,
    )

    cbar = fig.colorbar(field, ax=ax, pad=0.02)
    cbar.set_label(r"Disturbance magnitude on $v_y$")

    # ==================================================
    # Normalize optional geometry arrays.
    # ==================================================
    if waypoint_centers is not None:
        waypoint_centers = np.asarray(waypoint_centers, dtype=float)
    if waypoint_half_widths is not None:
        waypoint_half_widths = np.asarray(waypoint_half_widths, dtype=float)
    if waypoint_axis_u is not None:
        waypoint_axis_u = np.asarray(waypoint_axis_u, dtype=float)
    if waypoint_normals is not None:
        waypoint_normals = np.asarray(waypoint_normals, dtype=float)

    if gate_centers is not None:
        gate_centers = np.asarray(gate_centers, dtype=float)
    if gate_normals is not None:
        gate_normals = np.asarray(gate_normals, dtype=float)
    if gate_axis_u is not None:
        gate_axis_u = np.asarray(gate_axis_u, dtype=float)
    if gate_opening_half_widths is not None:
        gate_opening_half_widths = np.asarray(
            gate_opening_half_widths,
            dtype=float,
        )

    # ==================================================
    # Physical gates -- same oriented top-down footprint
    # as the animation/planning renderer.
    # ==================================================
    gate_handle = None

    if (
        gate_centers is not None
        and gate_normals is not None
        and gate_axis_u is not None
        and gate_opening_half_widths is not None
        and len(gate_centers) > 0
    ):
        for i, center in enumerate(gate_centers):
            u_xy = np.asarray(gate_axis_u[i, :2], dtype=float)
            n_xy = np.asarray(gate_normals[i, :2], dtype=float)

            u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
            n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

            hu_open = float(gate_opening_half_widths[i, 0])
            half_outer_width = hu_open + float(gate_bar_thickness)
            half_depth = 0.5 * float(gate_depth)

            c = np.asarray(center[:2], dtype=float)
            corners = np.array([
                c - half_outer_width * u_xy - half_depth * n_xy,
                c + half_outer_width * u_xy - half_depth * n_xy,
                c + half_outer_width * u_xy + half_depth * n_xy,
                c - half_outer_width * u_xy + half_depth * n_xy,
            ])

            gate_patch = Polygon(
                corners,
                closed=True,
                facecolor="none",
                edgecolor="black",
                linewidth=2.5,
                zorder=7,
            )
            ax.add_patch(gate_patch)

            if gate_handle is None:
                gate_handle = gate_patch

            ax.scatter(
                center[0],
                center[1],
                marker="x",
                s=45,
                color="black",
                zorder=8,
            )

            ax.annotate(
                f"G{i + 1}",
                (center[0], center[1]),
                xytext=(5, -14),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=9,
            )

    # ==================================================
    # Transition waypoints.
    # ==================================================
    if waypoint_centers is not None and len(waypoint_centers) > 0:
        ax.scatter(
            waypoint_centers[:, 0],
            waypoint_centers[:, 1],
            marker="s",
            s=90,
            label="Waypoints",
            zorder=7,
        )

        for i, waypoint in enumerate(waypoint_centers):
            ax.annotate(
                str(i + 1),
                (waypoint[0], waypoint[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                zorder=8,
            )

    # ==================================================
    # Oriented waypoint boxes in the local [u, n] plane.
    # Falls back to the old axis-aligned rectangles only
    # when orientation arrays are unavailable.
    # ==================================================
    if (
        waypoint_centers is not None
        and waypoint_half_widths is not None
        and len(waypoint_centers) > 0
        and waypoint_half_widths.shape[0] == waypoint_centers.shape[0]
    ):
        if (
            waypoint_axis_u is not None
            and waypoint_normals is not None
            and waypoint_axis_u.shape[0] == waypoint_centers.shape[0]
            and waypoint_normals.shape[0] == waypoint_centers.shape[0]
        ):
            for center, hw, u_axis, n_axis in zip(
                waypoint_centers,
                waypoint_half_widths,
                waypoint_axis_u,
                waypoint_normals,
            ):
                c = np.asarray(center[:2], dtype=float)
                u_xy = np.asarray(u_axis[:2], dtype=float)
                n_xy = np.asarray(n_axis[:2], dtype=float)

                u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
                n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

                hu = float(hw[0])
                hn = float(hw[2])

                corners = np.array([
                    c - hu * u_xy - hn * n_xy,
                    c + hu * u_xy - hn * n_xy,
                    c + hu * u_xy + hn * n_xy,
                    c - hu * u_xy + hn * n_xy,
                ])

                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        fill=False,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=6,
                    )
                )
        else:
            for center, hw in zip(
                waypoint_centers,
                waypoint_half_widths,
            ):
                ax.add_patch(
                    Rectangle(
                        (
                            center[0] - hw[0],
                            center[1] - hw[1],
                        ),
                        2.0 * hw[0],
                        2.0 * hw[1],
                        fill=False,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=6,
                    )
                )

    # ==================================================
    # Offline and closed-loop trajectories.
    # ==================================================
    ax.plot(
        X_plan[:, 0],
        X_plan[:, 1],
        "--",
        linewidth=1.5,
        label="Offline optimized path",
        zorder=3,
    )

    ax.plot(
        X_closed_loop[:, 0],
        X_closed_loop[:, 1],
        "-o",
        markersize=2,
        linewidth=2.5,
        label="Closed-loop GPU-SLS MPC",
        zorder=5,
    )

    ax.scatter(
        X_closed_loop[0, 0],
        X_closed_loop[0, 1],
        marker="x",
        s=100,
        linewidths=3,
        label="Start",
        zorder=9,
    )

    final_error = float(
        np.linalg.norm(
            X_closed_loop[-1, :3]
            - X_plan[-1, :3]
        )
    )

    ax.set_title(
        "GPU-SLS Receding-Horizon Tracking\n"
        f"Final position error: {final_error:.3f} m"
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if gate_handle is not None:
        handles.append(gate_handle)
        labels.append("Physical gate")

    ax.legend(
        handles,
        labels,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved top-down tracking plot to: {save_path}")


def plot_tracking_error(
    pos_errors,
    solve_times,
    save_path="multiphase_trajectory_tracking_error.png",
):
    import matplotlib.pyplot as plt

    steps = np.arange(
        len(pos_errors)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        steps,
        pos_errors,
        linewidth=2.0,
    )

    ax.set_xlabel("Controller step")
    ax.set_ylabel("Position tracking error [m]")
    ax.set_title(
        "GPU-SLS MPC Position Tracking Error"
    )
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved tracking-error plot to: {save_path}"
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track the saved GPU-SLS/JAX minimum-time trajectory "
            "using online receding-horizon GPU-SLS MPC."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="multiphase_trajectory_dist.npz",
        help=(
            "Offline GPU-SLS trajectory NPZ "
            "(default: multiphase_trajectory.npz)"
        ),
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_MPC_HORIZON,
        help=(
            f"MPC prediction horizon "
            f"(default: {DEFAULT_MPC_HORIZON})"
        ),
    )

    parser.add_argument(
        "--disturbance-mag",
        type=float,
        default=DISTURBANCE_MAG,
        help=(
            "Magnitude of the state-dependent vy uncertainty channel kept "
            "inside the robust MPC model. The actual rollout adversary uses "
            "E_MAG_MODEL in XYZ and points toward the closest gate bar. "
            f"Default vy-model magnitude is {DISTURBANCE_MAG}."
        ),
    )

    parser.add_argument(
        "--initial-x-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--initial-y-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--initial-z-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--progress-search-window",
        type=int,
        default=DEFAULT_PROGRESS_SEARCH_WINDOW,
        help=(
            "Number of OFFLINE trajectory nodes ahead of the current "
            "progress index searched for the nearest point. "
            f"Default is {DEFAULT_PROGRESS_SEARCH_WINDOW}."
        ),
    )

    parser.add_argument(
        "--goal-tol",
        type=float,
        default=DEFAULT_GOAL_TOL,
        help=(
            "Final XYZ position tolerance required to stop after "
            "reaching the final progress node. "
            f"Default is {DEFAULT_GOAL_TOL} m."
        ),
    )

    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=None,
        help=(
            "Maximum number of online MPC updates. "
            "Default is 3 times the number of offline intervals."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="multiphase_trajectory_tracking.npz",
        help="Output NPZ filename.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.horizon < 1:
        raise ValueError(
            "--horizon must be at least 1"
        )

    plan = load_offline_plan(
        args.input
    )

    initial_offset = np.zeros(
        NX,
        dtype=float,
    )

    initial_offset[0] = (
        args.initial_x_offset
    )
    initial_offset[1] = (
        args.initial_y_offset
    )
    initial_offset[2] = (
        args.initial_z_offset
    )

    result = run_tracking(
        plan=plan,
        horizon=args.horizon,
        disturbance_mag=args.disturbance_mag,
        initial_offset=initial_offset,
        progress_search_window=args.progress_search_window,
        goal_tol=args.goal_tol,
        max_control_steps=args.max_control_steps,
    )

    output_path = Path(
        args.output
    )

    np.savez(
        output_path,
        X_closed_loop=result["X_closed_loop"],
        U_closed_loop=result["U_closed_loop"],
        X_plan=plan["X_plan"],
        U_plan=plan["U_plan"],
        dt_plan=plan["dt_plan"],
        phase_times=plan["phase_times"],
        waypoint_steps=plan["waypoint_steps"],
        waypoint_centers=(
            plan["waypoint_centers"]
            if plan["waypoint_centers"] is not None
            else np.zeros((0, 3))
        ),
        waypoint_half_widths=(
            plan["waypoint_half_widths"]
            if plan["waypoint_half_widths"] is not None
            else np.zeros((0, 3))
        ),
        pos_errors=result["pos_errors"],
        state_errors=result["state_errors"],
        solve_times=result["solve_times"],
        initial_full_solve_time=np.array(
            result["initial_full_solve_time"]
        ),
        initial_nominal_sqp_iterations=np.array(
            GPU_SLS_INITIAL_NOMINAL_SQP_ITERS
        ),
        initial_sls_sqp_iterations=np.array(
            GPU_SLS_INITIAL_SLS_SQP_ITERS
        ),
        rti_sqp_iterations=np.array(
            GPU_SLS_RTI_SQP_ITERS
        ),
        objective_history=result["objective_history"],
        fallback_used=result["fallback_used"],
        # Per-controller-step nominal MPC predictions and coordinate-wise
        # robust XYZ tube half-widths. Shapes are respectively
        # [steps, H+1, 13], [steps, H, 4], and [steps, H+1, 3].
        X_mpc_nominal_history=result["X_mpc_nominal_history"],
        U_mpc_nominal_history=result["U_mpc_nominal_history"],
        tube_xyz_halfwidth_history=result["tube_xyz_halfwidth_history"],
        mpc_horizon=np.array(args.horizon),
        disturbance_mag=np.array(
            args.disturbance_mag
        ),
        e_mag_model=np.array(E_MAG_MODEL),
        disturbance_xyz_history=result[
            "disturbance_xyz_history"
        ],
        disturbance_direction_history=result[
            "disturbance_direction_history"
        ],
        closest_obstacle_distance_history=result[
            "closest_obstacle_distance_history"
        ],
        closest_gate_index_history=result[
            "closest_gate_index_history"
        ],
        closest_bar_index_history=result[
            "closest_bar_index_history"
        ],
        closest_obstacle_point_history=result[
            "closest_obstacle_point_history"
        ],
        dt_applied_history=result[
            "dt_applied_history"
        ],
        progress_idx_history=result[
            "progress_idx_history"
        ],
        nearest_distance_history=result[
            "nearest_distance_history"
        ],
        X_progress_reference=result[
            "X_progress_reference"
        ],
        reached_goal=np.array(
            result["reached_goal"]
        ),
        num_control_steps=np.array(
            result["num_control_steps"]
        ),
        progress_search_window=np.array(
            args.progress_search_window
        ),
        goal_tol=np.array(
            args.goal_tol
        ),
    )

    print(
        f"Saved closed-loop tracking result to: {output_path}"
    )

    plot_tracking_top_down(
        X_plan=plan["X_plan"],
        X_closed_loop=result["X_closed_loop"],
        waypoint_centers=plan["waypoint_centers"],
        waypoint_half_widths=plan["waypoint_half_widths"],
        waypoint_axis_u=plan["waypoint_axis_u"],
        waypoint_normals=plan["waypoint_normals"],
        gate_centers=plan["gate_centers"],
        gate_normals=plan["gate_normals"],
        gate_axis_u=plan["gate_axis_u"],
        gate_opening_half_widths=plan["gate_opening_half_widths"],
        gate_bar_thickness=plan["gate_bar_thickness"],
        gate_depth=plan["gate_depth"],
        disturbance_mag=args.disturbance_mag,
        save_path="multiphase_trajectory_tracking_topdown.png",
    )

    plot_tracking_error(
        pos_errors=result["pos_errors"],
        solve_times=result["solve_times"],
        save_path="multiphase_trajectory_tracking_error.png",
    )


if __name__ == "__main__":
    main()