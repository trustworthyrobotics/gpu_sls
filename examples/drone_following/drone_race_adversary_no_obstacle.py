from __future__ import annotations

import argparse
import time
from pathlib import Path

import casadi as ca
import numpy as np


# ============================================================
# FATROP receding-horizon trajectory tracker matched to the GPU-SLS experiment
#
# Workflow:
#   1. Load the optimized minimum-time FATROP trajectory.
#   2. Keep its optimized phase times fixed.
#   3. Build ONE reusable NOMINAL FATROP MPC problem.
#   4. At each trajectory node:
#        - set current measured state x_k
#        - set local X/U reference window
#        - set the optimized dt sequence
#        - solve
#        - apply only u_0
#        - roll the plant one step
#        - shift and repeat
#
# Online MPC state:
#   x = [px, py, pz, phi, theta, psi,
#        vx, vy, vz, p, q, r]
#
# Online MPC control:
#   u = [thrust, tau_phi, tau_theta, tau_psi]
#
# The phase-time states from the offline planner are NOT optimized
# online. Their optimized values are converted into a fixed dt_k
# schedule for tracking.
# ============================================================


MASS = 1.0
GRAVITY = 9.81

JX = 0.02
JY = 0.02
JZ = 0.04

J = ca.diag(ca.DM([JX, JY, JZ]))
J_INV = ca.diag(ca.DM([1.0 / JX, 1.0 / JY, 1.0 / JZ]))

NX = 12
NU = 4

DEFAULT_MPC_HORIZON = 20
DEFAULT_PROGRESS_SEARCH_WINDOW = 15
DEFAULT_GOAL_TOL = 0.362
DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER = 3

# Rollout duration semantics:
#   - if the final goal is reached, use the elapsed time at goal satisfaction;
#   - otherwise, use the elapsed time at the closest approach to the final goal.
# Kept for output-file compatibility with older analysis scripts.
ROLLOUT_DURATION_SEARCH_START_STEP = 0

# Disturbance settings matched to the GPU-SLS tracking experiment.
#
# The ACTUAL rollout receives two disturbance channels:
#   1. an adversarial XY position perturbation of magnitude E_MAG_MODEL,
#      directed toward the closest inflated physical gate bar after projecting
#      the closest-bar direction into XY and renormalizing it;
#   2. the same state-dependent +vy disturbance used in the GPU-SLS rollout.
#
# FATROP itself remains a NOMINAL MPC baseline: these disturbances are not
# inserted into its prediction model or constraints.
DISTURBANCE_MAG = 15.0
E_MAG_MODEL = 0.00

# Backward-compatible name used by a few existing output fields/call sites.
DISTURBANCE_K = E_MAG_MODEL

# Spatial window for the retained state-dependent vy disturbance channel.
DISTURBANCE_X_MIN = -2.0
DISTURBANCE_X_MAX = 1.0
DISTURBANCE_Y_MIN = -8.0
DISTURBANCE_Y_MAX = 0.0
DISTURBANCE_KX = 1.0
DISTURBANCE_KY = 1.0

# Constraint semantics:
#   - state bounds are imposed on predicted nodes j > 0 inside the MPC;
#   - the measured x0 is not clipped/constrained;
#   - gate/obstacle constraints are NOT imposed by the MPC;
#   - rollout is NOT terminated on gate/obstacle collisions;
#   - applied control is independently validated with CONTROL_BOUND_TOL.
ENABLE_PHYSICAL_STATE_BOUNDS = True
CHECK_ROLLOUT_STATE_BOUNDS = False
CONSTRAINT_TOL = 1.0e-2
CONTROL_BOUND_TOL = 5.0e-2

T_HOVER = MASS * GRAVITY
T_MAX = 2.0 * T_HOVER
TAU_MAX = 10.0

# Match the GPU-SLS Euler-angle singularity treatment.  The saved trajectory
# remains within this limit; clipping prevents tan(theta) and 1/cos(theta)
# from becoming unbounded in the model and actual rollout.
PITCH_SINGULARITY_MARGIN = np.deg2rad(5.0)
PITCH_MAX = np.pi / 2.0 - PITCH_SINGULARITY_MARGIN


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
# Disturbance field shared with the GPU-SLS tracking experiment
# ============================================================


def disturbance_spatial_scale(px, py):
    """Same smooth XY spatial window used by the GPU-SLS tracker."""
    x_window = (
        0.5 * (1.0 + np.tanh(DISTURBANCE_KX * (px - DISTURBANCE_X_MIN)))
        * 0.5 * (1.0 - np.tanh(DISTURBANCE_KX * (px - DISTURBANCE_X_MAX)))
    )
    y_window = (
        0.5 * (1.0 + np.tanh(DISTURBANCE_KY * (py - DISTURBANCE_Y_MIN)))
        * 0.5 * (1.0 - np.tanh(DISTURBANCE_KY * (py - DISTURBANCE_Y_MAX)))
    )
    return x_window * y_window


# ============================================================
# Physical gate geometry, collision constraints, and adversary
# ============================================================


def build_gate_bar_geometry_np(
    gate_centers,
    gate_normals,
    gate_axis_u,
    gate_axis_v,
    opening_half_widths,
    bar_thickness=0.10,
    gate_depth=0.10,
    drone_radius=0.10,
):
    """Build the same four inflated oriented gate bars as the planner."""
    gate_centers = np.asarray(gate_centers, dtype=float)
    gate_normals = np.asarray(gate_normals, dtype=float)
    gate_axis_u = np.asarray(gate_axis_u, dtype=float)
    gate_axis_v = np.asarray(gate_axis_v, dtype=float)
    opening_half_widths = np.asarray(opening_half_widths, dtype=float)

    num_gates = gate_centers.shape[0]
    hu = opening_half_widths[:, 0]
    hv = opening_half_widths[:, 1]

    t_bar = float(bar_thickness)
    half_depth = 0.5 * float(gate_depth)
    radius = float(drone_radius)
    zeros = np.zeros(num_gates, dtype=float)

    top_center_local = np.stack([
        zeros,
        hv + 0.5 * t_bar,
        zeros,
    ], axis=1)
    bottom_center_local = np.stack([
        zeros,
        -(hv + 0.5 * t_bar),
        zeros,
    ], axis=1)
    left_center_local = np.stack([
        -(hu + 0.5 * t_bar),
        zeros,
        zeros,
    ], axis=1)
    right_center_local = np.stack([
        hu + 0.5 * t_bar,
        zeros,
        zeros,
    ], axis=1)

    top_bottom_half_sizes = np.stack([
        hu + t_bar,
        0.5 * t_bar * np.ones_like(hu),
        half_depth * np.ones_like(hu),
    ], axis=1)
    left_right_half_sizes = np.stack([
        0.5 * t_bar * np.ones_like(hu),
        hv,
        half_depth * np.ones_like(hu),
    ], axis=1)

    # Inflate every bar by the drone radius so x[:3] is the drone center.
    top_bottom_half_sizes = top_bottom_half_sizes + radius
    left_right_half_sizes = left_right_half_sizes + radius

    def local_to_world(local_center):
        return (
            gate_centers
            + local_center[:, 0:1] * gate_axis_u
            + local_center[:, 1:2] * gate_axis_v
            + local_center[:, 2:3] * gate_normals
        )

    return {
        "top_centers": local_to_world(top_center_local),
        "bottom_centers": local_to_world(bottom_center_local),
        "left_centers": local_to_world(left_center_local),
        "right_centers": local_to_world(right_center_local),
        "top_bottom_half_sizes": top_bottom_half_sizes,
        "left_right_half_sizes": left_right_half_sizes,
    }


def oriented_coordinates_ca(pos, center, axis_u, axis_v, normal):
    """Return q=[u^T delta, v^T delta, n^T delta] in CasADi."""
    delta = pos - ca.DM(np.asarray(center, dtype=float).reshape(3, 1))
    u = ca.DM(np.asarray(axis_u, dtype=float).reshape(3, 1))
    v = ca.DM(np.asarray(axis_v, dtype=float).reshape(3, 1))
    n = ca.DM(np.asarray(normal, dtype=float).reshape(3, 1))
    return ca.vertcat(
        ca.dot(u, delta),
        ca.dot(v, delta),
        ca.dot(n, delta),
    )


def outside_oriented_box_constraint_ca(
    pos,
    box_center,
    half_sizes,
    axis_u,
    axis_v,
    normal,
):
    """Planner-identical residual g=-max(abs(q)-h) <= 0."""
    q = oriented_coordinates_ca(
        pos=pos,
        center=box_center,
        axis_u=axis_u,
        axis_v=axis_v,
        normal=normal,
    )
    h = ca.DM(np.asarray(half_sizes, dtype=float).reshape(3, 1))
    outside_margin = ca.mmax(ca.fabs(q) - h)
    return -outside_margin


def oriented_coordinates_np(position, center, axis_u, axis_v, normal):
    position = np.asarray(position, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    delta = position - center
    return np.array([
        np.dot(np.asarray(axis_u, dtype=float), delta),
        np.dot(np.asarray(axis_v, dtype=float), delta),
        np.dot(np.asarray(normal, dtype=float), delta),
    ], dtype=float)


def outside_oriented_box_residual_np(
    position,
    box_center,
    half_sizes,
    axis_u,
    axis_v,
    normal,
):
    """Numerical version of the exact gate-bar residual used in the MPC."""
    q = oriented_coordinates_np(
        position, box_center, axis_u, axis_v, normal
    )
    h = np.asarray(half_sizes, dtype=float).reshape(3)
    return float(-np.max(np.abs(q) - h))


def closest_point_on_oriented_box_boundary(
    position,
    box_center,
    half_sizes,
    axis_u,
    axis_v,
    normal,
):
    """Closest point on an oriented gate-bar box boundary."""
    position = np.asarray(position, dtype=float).reshape(3)
    center = np.asarray(box_center, dtype=float).reshape(3)
    h = np.asarray(half_sizes, dtype=float).reshape(3)
    u = np.asarray(axis_u, dtype=float).reshape(3)
    v = np.asarray(axis_v, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)

    q = oriented_coordinates_np(position, center, u, v, n)
    q_target = np.clip(q, -h, h)

    # If already inside the bar, project to the nearest face.
    if np.all(np.abs(q) <= h):
        face_margin = h - np.abs(q)
        axis = int(np.argmin(face_margin))
        sign = 1.0 if q[axis] >= 0.0 else -1.0
        q_target[axis] = sign * h[axis]

    return center + q_target[0] * u + q_target[1] * v + q_target[2] * n


def iter_gate_bars(gate_geometry):
    """Yield (gate_idx, bar_name, center, half_sizes, u, v, n)."""
    gate_centers = np.asarray(gate_geometry["gate_centers"], dtype=float)
    gate_normals = np.asarray(gate_geometry["gate_normals"], dtype=float)
    gate_axis_u = np.asarray(gate_geometry["gate_axis_u"], dtype=float)
    gate_axis_v = np.asarray(gate_geometry["gate_axis_v"], dtype=float)
    bars = gate_geometry["gate_bars"]

    for gate_idx in range(len(gate_centers)):
        u = gate_axis_u[gate_idx]
        v = gate_axis_v[gate_idx]
        n = gate_normals[gate_idx]
        yield (
            gate_idx, "top", bars["top_centers"][gate_idx],
            bars["top_bottom_half_sizes"][gate_idx], u, v, n
        )
        yield (
            gate_idx, "bottom", bars["bottom_centers"][gate_idx],
            bars["top_bottom_half_sizes"][gate_idx], u, v, n
        )
        yield (
            gate_idx, "left", bars["left_centers"][gate_idx],
            bars["left_right_half_sizes"][gate_idx], u, v, n
        )
        yield (
            gate_idx, "right", bars["right_centers"][gate_idx],
            bars["left_right_half_sizes"][gate_idx], u, v, n
        )


def nearest_gate_obstacle(position, gate_geometry):
    """Closest point on any physical gate bar."""
    position = np.asarray(position, dtype=float).reshape(3)
    best_distance = np.inf
    best_target = None
    best_gate_idx = -1
    best_bar_name = ""

    for gate_idx, bar_name, center, half_sizes, u, v, n in iter_gate_bars(gate_geometry):
        target = closest_point_on_oriented_box_boundary(
            position=position,
            box_center=center,
            half_sizes=half_sizes,
            axis_u=u,
            axis_v=v,
            normal=n,
        )
        distance = float(np.linalg.norm(target - position))
        if distance < best_distance:
            best_distance = distance
            best_target = target
            best_gate_idx = int(gate_idx)
            best_bar_name = bar_name

    return best_target, best_gate_idx, best_bar_name, float(best_distance)


def nearest_gate_bar_direction_np(position, gate_geometry, eps=1e-12):
    """GPU-SLS-matched direction toward the nearest inflated gate-bar OBB.

    Distance is to the closest point ON/IN each inflated oriented box.  If the
    current point is already touching/inside a box, the returned direction is
    zero; the discrete collision checker handles termination separately.
    """
    position = np.asarray(position, dtype=float).reshape(3)

    best_distance = np.inf
    best_target = None
    best_gate_idx = -1
    best_bar_name = ""

    for gate_idx, bar_name, center, half_sizes, u, v, n in iter_gate_bars(gate_geometry):
        center = np.asarray(center, dtype=float).reshape(3)
        half_sizes = np.asarray(half_sizes, dtype=float).reshape(3)
        u = np.asarray(u, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)
        n = np.asarray(n, dtype=float).reshape(3)

        delta = position - center
        q = np.array([
            np.dot(delta, u),
            np.dot(delta, v),
            np.dot(delta, n),
        ], dtype=float)

        q_closest = np.clip(q, -half_sizes, half_sizes)
        closest_world = (
            center
            + q_closest[0] * u
            + q_closest[1] * v
            + q_closest[2] * n
        )

        toward = closest_world - position
        distance = float(np.linalg.norm(toward))
        if distance < best_distance:
            best_distance = distance
            best_target = closest_world.copy()
            best_gate_idx = int(gate_idx)
            best_bar_name = bar_name

    if best_target is None:
        raise ValueError("No physical gate bars are available for the adversary.")

    if best_distance <= eps:
        direction = np.zeros(3, dtype=float)
    else:
        direction = (best_target - position) / best_distance

    return direction, best_distance, best_gate_idx, best_bar_name, best_target


def adversarial_disturbance_direction(position, gate_geometry):
    """Unit-norm w toward the closest physical gate-bar obstacle."""
    target, gate_idx, bar_name, gate_distance = nearest_gate_obstacle(
        position, gate_geometry
    )
    delta = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    delta_norm = float(np.linalg.norm(delta))
    w = delta / delta_norm if delta_norm > 1e-12 else np.zeros(3, dtype=float)
    w_norm = float(np.linalg.norm(w))
    if w_norm > 1.0:
        w = w / w_norm
    return w, gate_idx, gate_distance, target, bar_name


def box_constraint_violation_np(value, lower, upper):
    """Maximum component-wise violation of lower <= value <= upper."""
    value = np.asarray(value, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.max(np.maximum(lower - value, value - upper)))


def gate_collision_violation_np(position, gate_geometry):
    """Return maximum gate-bar penetration residual and its gate/bar index."""
    max_residual = -np.inf
    max_gate_idx = -1
    max_bar_name = ""

    for gate_idx, bar_name, center, half_sizes, u, v, n in iter_gate_bars(gate_geometry):
        residual = outside_oriented_box_residual_np(
            position=position,
            box_center=center,
            half_sizes=half_sizes,
            axis_u=u,
            axis_v=v,
            normal=n,
        )
        if residual > max_residual:
            max_residual = residual
            max_gate_idx = int(gate_idx)
            max_bar_name = bar_name

    return float(max_residual), max_gate_idx, max_bar_name


def check_rollout_constraints(
    x,
    u=None,
    gate_geometry=None,
    tol=CONSTRAINT_TOL,
):
    """Check exactly the gates-only hard feasible sets used by the MPC."""
    x = np.asarray(x, dtype=float).reshape(NX)

    if CHECK_ROLLOUT_STATE_BOUNDS:
        violation = box_constraint_violation_np(x, X_MIN, X_MAX)
        if violation > tol:
            return False, "physical_state_bounds", float(violation), -1, ""

    if u is not None:
        violation = box_constraint_violation_np(u, U_MIN, U_MAX)
        if violation > tol:
            return False, "control_bounds", float(violation), -1, ""

    if gate_geometry is not None:
        violation, gate_idx, bar_name = gate_collision_violation_np(
            x[:3], gate_geometry
        )
        if violation > tol:
            return False, "gate_collision", float(violation), gate_idx, bar_name

    return True, "", 0.0, -1, ""


# ============================================================
# CasADi dynamics used INSIDE the MPC
# ============================================================

def rotation_matrix_ca(phi, theta, psi):
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cpsi, spsi = ca.cos(psi), ca.sin(psi)

    return ca.vertcat(
        ca.horzcat(
            cpsi * cth,
            cpsi * sth * sphi - spsi * cphi,
            cpsi * sth * cphi + spsi * sphi,
        ),
        ca.horzcat(
            spsi * cth,
            spsi * sth * sphi + cpsi * cphi,
            spsi * sth * cphi - cpsi * sphi,
        ),
        ca.horzcat(
            -sth,
            cth * sphi,
            cth * cphi,
        ),
    )


def euler_angle_rates_matrix_ca(phi, theta):
    theta_safe = ca.fmin(ca.fmax(theta, -PITCH_MAX), PITCH_MAX)
    sphi, cphi = ca.sin(phi), ca.cos(phi)
    tth = ca.tan(theta_safe)
    cth = ca.cos(theta_safe)

    return ca.vertcat(
        ca.horzcat(1.0, sphi * tth, cphi * tth),
        ca.horzcat(0.0, cphi, -sphi),
        ca.horzcat(0.0, sphi / cth, cphi / cth),
    )


def quadrotor_xdot_ca(x, u):
    phi, theta, psi = x[3], x[4], x[5]

    v = x[6:9]
    omega = x[9:12]

    thrust = u[0]
    tau = u[1:4]

    R = rotation_matrix_ca(phi, theta, psi)
    E = euler_angle_rates_matrix_ca(phi, theta)

    e3 = ca.DM([0.0, 0.0, 1.0])

    pos_dot = v
    v_dot = (R @ (thrust * e3)) / MASS - GRAVITY * e3
    euler_dot = E @ omega
    omega_dot = J_INV @ (
        tau - ca.cross(omega, J @ omega)
    )

    return ca.vertcat(
        pos_dot,
        euler_dot,
        v_dot,
        omega_dot,
    )


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


def disturbance_at_state(
    x,
    gate_geometry,
    disturbance_k=E_MAG_MODEL,
):
    """Evaluate the GPU-SLS-matched XY gate-directed disturbance rate.

    The closest-bar direction is computed in 3-D, then projected into XY and
    renormalized.  Therefore the applied position disturbance uses the full
    allowed XY magnitude whenever that projection is nonzero, and z is exactly
    zero.
    """
    w_xyz, gate_distance, gate_idx, bar_name, target = nearest_gate_bar_direction_np(
        position=x[:3],
        gate_geometry=gate_geometry,
    )

    k = float(disturbance_k)
    if k < 0.0:
        raise ValueError("E_MAG_MODEL / disturbance_k must be nonnegative.")

    w_xy_norm = float(np.linalg.norm(w_xyz[:2]))
    if w_xy_norm > 1e-12:
        disturbance_direction = np.array([
            w_xyz[0] / w_xy_norm,
            w_xyz[1] / w_xy_norm,
            0.0,
        ], dtype=float)
    else:
        disturbance_direction = np.zeros(3, dtype=float)

    disturbance_xyz = k * disturbance_direction
    return (
        disturbance_xyz,
        disturbance_direction,
        gate_idx,
        gate_distance,
        target,
        bar_name,
    )


def plant_step(
    x,
    u,
    dt,
    gate_geometry,
    disturbance_k=E_MAG_MODEL,
):
    """One actual closed-loop step matched to the GPU-SLS tracking plant.

    Nominal Euler rollout:
        x_next = x + dt * f(x, u)

    Adversarial XY position perturbation:
        delta_p = dt * E_MAG_MODEL * [w_x, w_y, 0]
        ||[w_x, w_y]||_2 = 1 whenever the projected direction is nonzero.

    State-dependent +vy perturbation:
        delta_vy = dt * DISTURBANCE_MAG * spatial_scale(px, py)

    FATROP does NOT model either disturbance inside its prediction problem.
    """
    x = np.asarray(x, dtype=float)
    x_next = x + dt * quadrotor_xdot_np(x, u)

    (
        disturbance_xyz,
        disturbance_direction,
        gate_idx,
        gate_distance,
        target,
        bar_name,
    ) = disturbance_at_state(
        x=x,
        gate_geometry=gate_geometry,
        disturbance_k=disturbance_k,
    )

    position_delta = dt * disturbance_xyz
    x_next[:3] += position_delta

    spatial_scale = disturbance_spatial_scale(x[0], x[1])
    vy_delta = dt * DISTURBANCE_MAG * spatial_scale
    x_next[7] += vy_delta

    return (
        x_next,
        position_delta,
        disturbance_direction,
        gate_idx,
        gate_distance,
        target,
        bar_name,
    )


# ============================================================
# MPC tracking cost
# ============================================================

def tracking_stage_cost(x, u, xref, uref):
    dpos = x[0:3] - xref[0:3]
    dang = x[3:6] - xref[3:6]
    dvel = x[6:9] - xref[6:9]
    drate = x[9:12] - xref[9:12]
    du = u - uref

    # Use 1-cos for Euler-angle tracking, matching the planner's
    # periodic angle treatment.
    angle_cost = (
        Q_ANG[0] * (1.0 - ca.cos(dang[0]))
        + Q_ANG[1] * (1.0 - ca.cos(dang[1]))
        + Q_ANG[2] * (1.0 - ca.cos(dang[2]))
    )

    return (
        Q_POS[0] * dpos[0] ** 2
        + Q_POS[1] * dpos[1] ** 2
        + Q_POS[2] * dpos[2] ** 2
        + angle_cost
        + Q_VEL[0] * dvel[0] ** 2
        + Q_VEL[1] * dvel[1] ** 2
        + Q_VEL[2] * dvel[2] ** 2
        + Q_RATE[0] * drate[0] ** 2
        + Q_RATE[1] * drate[1] ** 2
        + Q_RATE[2] * drate[2] ** 2
        + R_U[0] * du[0] ** 2
        + R_U[1] * du[1] ** 2
        + R_U[2] * du[2] ** 2
        + R_U[3] * du[3] ** 2
    )


def tracking_terminal_cost(x, xref, motion_weight=1.0):
    """
    Terminal tracking cost.

    motion_weight = 1:
        Normal full-state terminal tracking.

    motion_weight = 0:
        Track only terminal position/orientation. This is used when the fixed
        MPC horizon extends beyond the end of the offline trajectory, so the
        artificial horizon padding does not make the vehicle brake simply
        because the saved trajectory ends.
    """
    dpos = x[0:3] - xref[0:3]
    dang = x[3:6] - xref[3:6]
    dvel = x[6:9] - xref[6:9]
    drate = x[9:12] - xref[9:12]

    angle_cost = (
        Q_ANG[0] * (1.0 - ca.cos(dang[0]))
        + Q_ANG[1] * (1.0 - ca.cos(dang[1]))
        + Q_ANG[2] * (1.0 - ca.cos(dang[2]))
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

    return TERMINAL_MULTIPLIER * (
        pose_cost + motion_weight * motion_cost
    )


# ============================================================
# FATROP MPC construction
# ============================================================

def dm_bound(value, n):
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return float(arr) * ca.DM.ones(n, 1)

    return ca.DM(
        arr.reshape(n, 1)
    )


def build_fatrop_mpc(horizon):
    """Build one reusable FATROP MPC with state + control bounds only."""
    H = int(horizon)
    K = H + 1

    opti = ca.Opti()

    nx = [NX] * K
    nu = [NU] * H + [0]
    ng = [0] * K

    X = []
    U = []
    for j in range(K):
        X.append(opti.variable(NX))
        U.append(opti.variable(NU if j < H else 0))

    x_measured = opti.parameter(NX)
    X_ref = [opti.parameter(NX) for _ in range(K)]
    U_ref = [opti.parameter(NU) for _ in range(H)]
    dt_ref = [opti.parameter() for _ in range(H)]
    stage_active = [opti.parameter() for _ in range(H)]
    terminal_motion_weight = opti.parameter()

    def add_local_constraint(stage, expr, lb, ub):
        n_expr = int(expr.numel())
        opti.subject_to(
            opti.bounded(
                dm_bound(lb, n_expr),
                expr,
                dm_bound(ub, n_expr),
            )
        )
        ng[stage] += n_expr

    objective = 0

    for j in range(K):
        xj = X[j]

        # FATROP dynamics gap must be first.
        if j < H:
            uj = U[j]
            x_next = xj + dt_ref[j] * quadrotor_xdot_ca(xj, uj)
            opti.subject_to(X[j + 1] == x_next)

        if j == 0:
            opti.subject_to(xj == x_measured)
            ng[j] += NX

        if ENABLE_PHYSICAL_STATE_BOUNDS and j > 0:
            add_local_constraint(j, xj, X_MIN, X_MAX)

        if j < H:
            add_local_constraint(j, U[j], U_MIN, U_MAX)
            objective += stage_active[j] * tracking_stage_cost(
                xj, U[j], X_ref[j], U_ref[j]
            )
        else:
            objective += tracking_terminal_cost(
                xj, X_ref[j], motion_weight=terminal_motion_weight
            )

    opti.minimize(objective)

    solver_options = {
        "structure_detection": "manual",
        "nx": nx,
        "nu": nu,
        "ng": ng,
        "N": H,
        "expand": True,

        "fatrop": {
            "tol": 1e-3,
            "max_iter": 400,
            "mu_init": 1e-1,
        },
    }

    opti.solver("fatrop", solver_options)

    return {
        "opti": opti,
        "X": X,
        "U": U,
        "x_measured": x_measured,
        "X_ref": X_ref,
        "U_ref": U_ref,
        "dt_ref": dt_ref,
        "stage_active": stage_active,
        "terminal_motion_weight": terminal_motion_weight,
        "objective": objective,
        "H": H,
    }


# ============================================================
# Offline optimized trajectory loading
# ============================================================

def load_offline_plan(filename):
    filename = Path(filename).expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Offline FATROP plan not found: {filename}")

    with np.load(filename, allow_pickle=False) as data:
        required = [
            "X", "U", "phase_times", "waypoint_steps",
            "gate_centers", "gate_normals", "gate_axis_u", "gate_axis_v",
            "gate_opening_half_widths", "gate_bar_thickness",
            "gate_depth", "drone_radius",
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"{filename} is missing required key '{key}'")

        X_full = np.asarray(data["X"], dtype=float)
        U_plan = np.asarray(data["U"], dtype=float)
        phase_times = np.asarray(data["phase_times"], dtype=float).reshape(-1)
        waypoint_steps = np.asarray(data["waypoint_steps"], dtype=int).reshape(-1)

        gate_centers = np.asarray(data["gate_centers"], dtype=float)
        gate_normals = np.asarray(data["gate_normals"], dtype=float)
        gate_axis_u = np.asarray(data["gate_axis_u"], dtype=float)
        gate_axis_v = np.asarray(data["gate_axis_v"], dtype=float)
        gate_opening_half_widths = np.asarray(
            data["gate_opening_half_widths"], dtype=float
        )
        gate_bar_thickness = float(np.asarray(data["gate_bar_thickness"]))
        gate_depth = float(np.asarray(data["gate_depth"]))
        drone_radius = float(np.asarray(data["drone_radius"]))

    if X_full.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got {X_full.shape}")
    if X_full.shape[1] < NX:
        raise ValueError(f"Expected at least {NX} state columns, got {X_full.shape[1]}")

    X_plan = X_full[:, :NX]
    N = X_plan.shape[0] - 1
    if U_plan.shape != (N, NU):
        raise ValueError(f"U must have shape {(N, NU)}, got {U_plan.shape}")

    if np.any(np.diff(waypoint_steps) <= 0):
        raise ValueError("waypoint_steps/phase-transition steps must be strictly increasing")
    if waypoint_steps[-1] != N:
        raise ValueError(
            "Final phase-transition step must equal the number of control intervals."
        )

    num_gates = gate_centers.shape[0]
    for name, arr in [
        ("gate_normals", gate_normals),
        ("gate_axis_u", gate_axis_u),
        ("gate_axis_v", gate_axis_v),
    ]:
        if arr.shape != (num_gates, 3):
            raise ValueError(f"{name} must have shape {(num_gates, 3)}, got {arr.shape}")
    if gate_opening_half_widths.shape != (num_gates, 2):
        raise ValueError(
            "gate_opening_half_widths must have shape "
            f"{(num_gates, 2)}, got {gate_opening_half_widths.shape}"
        )

    segment_lengths = np.concatenate([
        waypoint_steps[:1],
        waypoint_steps[1:] - waypoint_steps[:-1],
    ]).astype(float)
    if len(phase_times) != len(segment_lengths):
        raise ValueError(
            "phase_times and phase-transition steps imply different numbers of phases."
        )

    dt_plan = np.empty(N, dtype=float)
    for k in range(N):
        phase_idx = int(np.searchsorted(waypoint_steps, k, side="right"))
        dt_plan[k] = phase_times[phase_idx] / segment_lengths[phase_idx]

    gate_bars = build_gate_bar_geometry_np(
        gate_centers=gate_centers,
        gate_normals=gate_normals,
        gate_axis_u=gate_axis_u,
        gate_axis_v=gate_axis_v,
        opening_half_widths=gate_opening_half_widths,
        bar_thickness=gate_bar_thickness,
        gate_depth=gate_depth,
        drone_radius=drone_radius,
    )

    gate_geometry = {
        "gate_centers": gate_centers,
        "gate_normals": gate_normals,
        "gate_axis_u": gate_axis_u,
        "gate_axis_v": gate_axis_v,
        "gate_opening_half_widths": gate_opening_half_widths,
        "gate_bar_thickness": gate_bar_thickness,
        "gate_depth": gate_depth,
        "drone_radius": drone_radius,
        "gate_bars": gate_bars,
    }

    return {
        "filename": filename,
        "X_plan": X_plan,
        "U_plan": U_plan,
        "phase_times": phase_times,
        "waypoint_steps": waypoint_steps,  # phase-time bookkeeping only
        "dt_plan": dt_plan,
        "gate_geometry": gate_geometry,
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

    Since dt = 0 on padded intervals,

        x[j+1] = x[j]

    and since active = 0, those fictitious stages cannot make the controller
    brake early. The FATROP problem size remains fixed.
    """
    H = int(horizon)
    N = len(U_plan)

    if N < 1:
        raise ValueError("The offline plan must contain at least one control interval.")

    # Node N has no outgoing interval.  Match GPU-SLS by rooting the MPC at the
    # last real interval if nearest-point matching reaches the terminal node
    # before the final position tolerance is satisfied.
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

    # Keep final-state padding only as a sensible warm-start/reference value.
    # Its stage cost is masked out once the real trajectory has ended.
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
            # Fictitious fixed-horizon padding:
            # - no dynamics advancement
            # - no stage tracking objective
            U_ref[j] = np.array(
                [T_HOVER, 0.0, 0.0, 0.0],
                dtype=float,
            )

            dt_ref[j] = 0.0
            stage_active[j] = 0.0

    # Once the prediction horizon reaches beyond the end of the saved path,
    # do not ask the artificial terminal stage to match final velocity/rates.
    # Final position/orientation tracking remains active.
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

def solve_mpc_step(
    mpc,
    x_current,
    X_ref_np,
    U_ref_np,
    dt_ref_np,
    stage_active_np,
    terminal_motion_weight_np,
    X_guess=None,
    U_guess=None,
):
    opti = mpc["opti"]
    H = mpc["H"]
    x_current = np.asarray(x_current, dtype=float).reshape(NX)
    if not np.all(np.isfinite(x_current)):
        raise ValueError("Measured state contains NaN or Inf.")

    # Match GPU-SLS: do not clip or constrain the measured node x0.  State-box
    # constraints apply only to predicted nodes j > 0 inside build_fatrop_mpc().
    opti.set_value(mpc["x_measured"], x_current)
    for j in range(H + 1):
        opti.set_value(mpc["X_ref"][j], X_ref_np[j])
    for j in range(H):
        opti.set_value(mpc["U_ref"][j], U_ref_np[j])
        opti.set_value(mpc["dt_ref"][j], float(dt_ref_np[j]))
        opti.set_value(mpc["stage_active"][j], float(stage_active_np[j]))
    opti.set_value(
        mpc["terminal_motion_weight"], float(terminal_motion_weight_np)
    )

    if X_guess is None:
        X_guess = X_ref_np.copy()
    if U_guess is None:
        U_guess = U_ref_np.copy()

    X_guess = np.asarray(X_guess, dtype=float).copy()
    X_guess[0] = x_current

    for j in range(H + 1):
        opti.set_initial(mpc["X"][j], X_guess[j])
    for j in range(H):
        opti.set_initial(mpc["U"][j], U_guess[j])

    tic = time.perf_counter()
    sol = opti.solve()
    solve_time = time.perf_counter() - tic

    X_sol = np.stack([
        np.asarray(sol.value(mpc["X"][j])).reshape(-1)
        for j in range(H + 1)
    ])
    U_sol = np.stack([
        np.asarray(sol.value(mpc["U"][j])).reshape(-1)
        for j in range(H)
    ])
    if not np.all(np.isfinite(X_sol)):
        raise RuntimeError("FATROP produced NaN or Inf in its predicted state trajectory.")
    if not np.all(np.isfinite(U_sol)):
        raise RuntimeError("FATROP produced NaN or Inf in its predicted control trajectory.")

    u0 = U_sol[0]
    if np.any(u0 < U_MIN - CONTROL_BOUND_TOL) or np.any(
        u0 > U_MAX + CONTROL_BOUND_TOL
    ):
        raise RuntimeError(
            "FATROP rejected an out-of-bounds first control: "
            f"u0={u0}, bounds=[{U_MIN}, {U_MAX}]."
        )

    objective = float(sol.value(mpc["objective"]))
    if not np.isfinite(objective):
        raise RuntimeError("FATROP produced a nonfinite tracking objective.")

    return X_sol, U_sol, objective, solve_time


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
    state is still tracked by the FATROP MPC cost.
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
    disturbance_k=DISTURBANCE_K,
    initial_offset=None,
    progress_search_window=DEFAULT_PROGRESS_SEARCH_WINDOW,
    goal_tol=DEFAULT_GOAL_TOL,
    max_control_steps=None,
):
    """Closed-loop tracking with no gate/obstacle constraints.

    The MPC retains the original dynamics, tracking objective, physical state
    bounds, and control bounds. Gate/obstacle collisions do not terminate the
    rollout.
    """
    X_plan = plan["X_plan"]
    U_plan = plan["U_plan"]
    dt_plan = plan["dt_plan"]
    gate_geometry = plan["gate_geometry"]
    N = plan["N"]

    if progress_search_window < 1:
        raise ValueError("progress_search_window must be >= 1")
    if max_control_steps is None:
        max_control_steps = DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER * N
    max_control_steps = int(max_control_steps)
    if max_control_steps < 1:
        raise ValueError("max_control_steps must be >= 1")

    mpc = build_fatrop_mpc(horizon=horizon)

    X_closed_loop = np.zeros((max_control_steps + 1, NX), dtype=float)
    U_closed_loop = np.zeros((max_control_steps, NU), dtype=float)
    objective_history = np.full(max_control_steps, np.nan, dtype=float)
    solve_times = np.full(max_control_steps, np.nan, dtype=float)
    fallback_used = np.zeros(max_control_steps, dtype=bool)

    # Actual discrete XYZ position perturbation applied at each plant step.
    disturbance_history = np.zeros((max_control_steps, 3), dtype=float)
    disturbance_w_history = np.zeros((max_control_steps, 3), dtype=float)
    disturbance_gate_index_history = np.full(max_control_steps, -1, dtype=int)
    disturbance_gate_distance_history = np.full(max_control_steps, np.nan, dtype=float)
    dt_applied_history = np.zeros(max_control_steps, dtype=float)

    progress_idx_history = np.zeros(max_control_steps + 1, dtype=int)
    nearest_distance_history = np.full(max_control_steps + 1, np.nan, dtype=float)

    constraint_violation_occurred = False
    constraint_violation_step = -1
    constraint_violation_type = ""
    constraint_violation_gate_index = -1
    constraint_violation_gate_bar = ""
    constraint_violation_amount = 0.0

    x_current = X_plan[0].copy()
    if initial_offset is not None:
        x_current += np.asarray(initial_offset, dtype=float)
    X_closed_loop[0] = x_current

    progress_idx, nearest_distance = find_monotonic_progress_index(
        x_current=x_current,
        X_plan=X_plan,
        previous_progress_idx=0,
        search_window=progress_search_window,
    )
    progress_idx_history[0] = progress_idx
    nearest_distance_history[0] = nearest_distance

    X_guess = None
    U_guess = None

    print("\n================ FATROP MPC TRACKING ================")
    print(f"Offline plan: {plan['filename']}")
    print(f"Offline plan intervals: {N}")
    print(f"MPC horizon: {horizon}")
    print("Progress mode: monotonic nearest point")
    print(f"Progress search window: {progress_search_window} nodes")
    print(f"Goal tolerance: {goal_tol:.3f} m")
    print(f"Max controller steps: {max_control_steps}")
    print(f"E_MAG_MODEL: {disturbance_k}")
    print(f"State-dependent vy disturbance magnitude: {DISTURBANCE_MAG}")
    print("Disturbance target: closest physical gate bar")
    print("Plant disturbance: XY-only gate-directed position + state-dependent +vy")
    print(f"MPC physical state bounds enabled: {ENABLE_PHYSICAL_STATE_BOUNDS}")
    print(f"Rollout state-bound termination enabled: {CHECK_ROLLOUT_STATE_BOUNDS}")
    print("MPC constraints: physical state box + control box only")
    print("Gate/obstacle constraints in MPC: NONE")
    print("Gate/obstacle collision termination: DISABLED")
    print("Waypoint position constraints: NONE")
    print("=====================================================")

    num_control_steps = 0
    reached_goal = False
    previous_reference_progress = progress_idx

    for control_step in range(max_control_steps):
        progress_idx, nearest_distance = find_monotonic_progress_index(
            x_current=x_current,
            X_plan=X_plan,
            previous_progress_idx=progress_idx,
            search_window=progress_search_window,
        )
        progress_advance = progress_idx - previous_reference_progress

        final_pos_error_before = float(np.linalg.norm(x_current[:3] - X_plan[-1, :3]))
        if progress_idx >= N and final_pos_error_before <= goal_tol:
            reached_goal = True
            break

        # Node N has no outgoing control interval.  Keep optimizing the last
        # real interval until the final XYZ tolerance is actually satisfied.
        mpc_progress_idx = min(progress_idx, N - 1)

        (
            X_ref, U_ref, dt_ref, stage_active, terminal_motion_weight
        ) = local_reference_window(
            X_plan=X_plan,
            U_plan=U_plan,
            dt_plan=dt_plan,
            k=mpc_progress_idx,
            horizon=horizon,
        )

        dt_apply = float(dt_plan[mpc_progress_idx])

        reset_warm_start = (
            control_step > 0
            and (
                mpc_progress_idx
                - min(previous_reference_progress, N - 1)
            ) != 1
        )
        if reset_warm_start:
            X_guess = None
            U_guess = None

        try:
            X_mpc, U_mpc, objective, solve_time = solve_mpc_step(
                mpc=mpc,
                x_current=x_current,
                X_ref_np=X_ref,
                U_ref_np=U_ref,
                dt_ref_np=dt_ref,
                stage_active_np=stage_active,
                terminal_motion_weight_np=terminal_motion_weight,
                X_guess=X_guess,
                U_guess=U_guess,
            )
            # Match GPU-SLS applied-control semantics: reject material bound
            # violations in solve_mpc_step(), then remove harmless roundoff.
            u_apply = np.clip(U_mpc[0], U_MIN, U_MAX)
            objective_history[control_step] = objective
            solve_times[control_step] = solve_time
            X_guess, U_guess = shift_warm_start(X_mpc, U_mpc)
        except RuntimeError as error:
            print(
                f"\nMPC solve failed at controller step {control_step}; "
                f"using offline feedforward control at progress node {mpc_progress_idx}."
            )
            print(error)
            u_apply = U_ref[0].copy()
            fallback_used[control_step] = True
            X_guess = None
            U_guess = None

        previous_reference_progress = progress_idx

        # Independently validate the control sent to the plant, matching the
        # GPU-SLS tolerance.  (A successful solve has already been clipped only
        # for tiny numerical roundoff at active bounds.)
        control_violation = box_constraint_violation_np(u_apply, U_MIN, U_MAX)
        if control_violation > CONTROL_BOUND_TOL:
            constraint_violation_occurred = True
            constraint_violation_step = control_step
            constraint_violation_type = "control_bounds"
            constraint_violation_amount = float(control_violation)
            print(
                f"\nCONSTRAINT VIOLATION at controller step {control_step}: "
                f"type=control_bounds, violation={control_violation:.6e}"
            )
            break

        U_closed_loop[control_step] = u_apply
        dt_applied_history[control_step] = dt_apply

        (
            x_current,
            position_delta,
            disturbance_w,
            disturbance_gate_idx,
            disturbance_gate_distance,
            disturbance_target,
            disturbance_gate_bar,
        ) = plant_step(
            x=x_current,
            u=u_apply,
            dt=dt_apply,
            gate_geometry=gate_geometry,
            disturbance_k=disturbance_k,
        )

        disturbance_history[control_step] = position_delta
        disturbance_w_history[control_step] = disturbance_w
        disturbance_gate_index_history[control_step] = disturbance_gate_idx
        disturbance_gate_distance_history[control_step] = disturbance_gate_distance

        num_control_steps = control_step + 1
        X_closed_loop[num_control_steps] = x_current

        progress_idx, next_nearest_distance = find_monotonic_progress_index(
            x_current=x_current,
            X_plan=X_plan,
            previous_progress_idx=progress_idx,
            search_window=progress_search_window,
        )
        progress_idx_history[num_control_steps] = progress_idx
        nearest_distance_history[num_control_steps] = next_nearest_distance

        reference_pos_error = float(
            np.linalg.norm(x_current[:3] - X_plan[progress_idx, :3])
        )
        final_pos_error = float(np.linalg.norm(x_current[:3] - X_plan[-1, :3]))
        solve_string = (
            f"{1000.0 * solve_times[control_step]:9.3f} ms"
            if np.isfinite(solve_times[control_step])
            else " fallback"
        )

        print(
            f"MPC {control_step:03d}: "
            f"progress={progress_idx:03d}/{N:03d}  "
            f"advance={progress_advance:+d}  "
            f"nearest_err={reference_pos_error:.4f} m  "
            f"final_err={final_pos_error:.4f} m  "
            f"solve={solve_string}  "
            f"delta_p={np.array2string(position_delta, precision=3)}  "
            f"|w|={np.linalg.norm(disturbance_w):.3f}  "
            f"dist_gate={disturbance_gate_idx + 1}:{disturbance_gate_bar}"
        )

        if progress_idx >= N and final_pos_error <= goal_tol:
            reached_goal = True
            break

    X_closed_loop = X_closed_loop[:num_control_steps + 1]
    U_closed_loop = U_closed_loop[:num_control_steps]
    objective_history = objective_history[:num_control_steps]
    solve_times = solve_times[:num_control_steps]
    fallback_used = fallback_used[:num_control_steps]
    disturbance_history = disturbance_history[:num_control_steps]
    disturbance_w_history = disturbance_w_history[:num_control_steps]
    disturbance_gate_index_history = disturbance_gate_index_history[:num_control_steps]
    disturbance_gate_distance_history = disturbance_gate_distance_history[:num_control_steps]
    dt_applied_history = dt_applied_history[:num_control_steps]
    progress_idx_history = progress_idx_history[:num_control_steps + 1]
    nearest_distance_history = nearest_distance_history[:num_control_steps + 1]

    X_progress_reference = X_plan[progress_idx_history]
    pos_errors = np.linalg.norm(
        X_closed_loop[:, :3] - X_progress_reference[:, :3], axis=1
    )
    state_errors = np.linalg.norm(X_closed_loop - X_progress_reference, axis=1)
    final_pos_error = float(np.linalg.norm(X_closed_loop[-1, :3] - X_plan[-1, :3]))
    valid_solve_times = solve_times[np.isfinite(solve_times)]

    # Rollout duration semantics:
    #
    #   If the final goal is reached:
    #       duration = elapsed physical time when the goal tolerance is met.
    #
    #   If the final goal is NOT reached:
    #       duration = elapsed physical time at the closest approach to the
    #       final goal over the entire executed closed-loop rollout.
    #
    # X_closed_loop[k] is the state after k applied control intervals, so the
    # elapsed physical time at state k is sum(dt_applied_history[:k]).
    rollout_search_start_step = ROLLOUT_DURATION_SEARCH_START_STEP
    goal_distance_history = np.linalg.norm(
        X_closed_loop[:, :3] - X_plan[-1, :3][None, :],
        axis=1,
    )

    if reached_goal:
        # run_tracking() terminates immediately after detecting goal satisfaction,
        # so the final stored state is the state at which the goal was reached.
        rollout_duration_step = num_control_steps
    else:
        # Failed rollout: report the time corresponding to the closest approach
        # to the final goal, rather than the total time until max_control_steps.
        rollout_duration_step = int(np.argmin(goal_distance_history))

    min_goal_distance = float(goal_distance_history[rollout_duration_step])
    rollout_duration = float(
        np.sum(dt_applied_history[:rollout_duration_step])
    )

    print("\n================ TRACKING RESULT =====================")
    print(f"Reached final goal: {reached_goal}")
    print(f"Controller steps executed: {num_control_steps}")
    print(f"Final progress index: {progress_idx_history[-1]}/{N}")
    print(f"Final position error [m]: {final_pos_error:.6f}")
    if reached_goal:
        print(
            f"Rollout duration [s]: {rollout_duration:.6f} "
            f"(goal reached at step {rollout_duration_step})"
        )
    else:
        print(
            f"Rollout duration [s]: {rollout_duration:.6f} "
            f"(closest approach to goal at step {rollout_duration_step})"
        )
    print(
        f"Minimum goal distance [m]: {min_goal_distance:.6f}"
    )
    print(f"Max progress-matched position error [m]: {np.max(pos_errors):.6f}")
    print(f"Mean progress-matched position error [m]: {np.mean(pos_errors):.6f}")
    if len(valid_solve_times) > 0:
        print(f"Mean FATROP MPC solve [ms]: {1000.0 * np.mean(valid_solve_times):.3f}")
        print(f"Max FATROP MPC solve [ms]:  {1000.0 * np.max(valid_solve_times):.3f}")
    print(f"Fallback controls used: {np.sum(fallback_used)}")
    print(f"Constraint violation: {constraint_violation_occurred}")
    if constraint_violation_occurred:
        print(f"Violation type: {constraint_violation_type}")
        print(f"Violation controller step: {constraint_violation_step}")
        print(f"Violation amount: {constraint_violation_amount:.6e}")
        if constraint_violation_gate_index >= 0:
            print(
                f"Collision gate/bar: {constraint_violation_gate_index + 1}/"
                f"{constraint_violation_gate_bar}"
            )
    if not reached_goal and not constraint_violation_occurred:
        print(
            "WARNING: maximum controller steps reached before satisfying "
            "the final goal tolerance."
        )
    print("======================================================")

    return {
        "X_closed_loop": X_closed_loop,
        "U_closed_loop": U_closed_loop,
        "objective_history": objective_history,
        "solve_times": solve_times,
        "fallback_used": fallback_used,
        "pos_errors": pos_errors,
        "state_errors": state_errors,
        "disturbance_history": disturbance_history,
        "disturbance_w_history": disturbance_w_history,
        "disturbance_gate_index_history": disturbance_gate_index_history,
        "disturbance_gate_distance_history": disturbance_gate_distance_history,
        "dt_applied_history": dt_applied_history,
        "progress_idx_history": progress_idx_history,
        "nearest_distance_history": nearest_distance_history,
        "X_progress_reference": X_progress_reference,
        "reached_goal": reached_goal,
        "num_control_steps": num_control_steps,
        "rollout_duration": rollout_duration,
        "rollout_duration_step": rollout_duration_step,
        "rollout_duration_search_start_step": rollout_search_start_step,
        "min_goal_distance": min_goal_distance,
        "goal_distance_history": goal_distance_history,
        "constraint_violation_occurred": constraint_violation_occurred,
        "constraint_violation_step": constraint_violation_step,
        "constraint_violation_type": constraint_violation_type,
        "constraint_violation_gate_index": constraint_violation_gate_index,
        "constraint_violation_gate_bar": constraint_violation_gate_bar,
        "constraint_violation_amount": constraint_violation_amount,
    }


# ============================================================
# Visualization
# ============================================================

def plot_tracking_top_down(
    X_plan,
    X_closed_loop,
    gate_geometry,
    disturbance_history=None,
    save_path="fatrop_mpc_tracking_topdown.png",
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    x_min, x_max = -6.0, 5.0
    y_min, y_max = -7.5, 8.0
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(
        X_plan[:, 0], X_plan[:, 1], "--", linewidth=2.0,
        label="Offline optimized trajectory", zorder=4,
    )
    ax.plot(
        X_closed_loop[:, 0], X_closed_loop[:, 1], "-o", markersize=2,
        linewidth=2.0, label="Online FATROP MPC rollout", zorder=5,
    )
    ax.scatter(
        X_closed_loop[0, 0], X_closed_loop[0, 1], marker="x", s=100,
        linewidths=3, label="Start", zorder=7,
    )

    gate_centers = np.asarray(gate_geometry["gate_centers"], dtype=float)
    gate_normals = np.asarray(gate_geometry["gate_normals"], dtype=float)
    gate_axis_u = np.asarray(gate_geometry["gate_axis_u"], dtype=float)
    openings = np.asarray(gate_geometry["gate_opening_half_widths"], dtype=float)
    bar_thickness = float(gate_geometry["gate_bar_thickness"])
    gate_depth = float(gate_geometry["gate_depth"])

    for i, center in enumerate(gate_centers):
        u_xy = gate_axis_u[i, :2]
        n_xy = gate_normals[i, :2]
        u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
        n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

        half_outer_width = float(openings[i, 0]) + bar_thickness
        half_depth = 0.5 * gate_depth
        c = center[:2]
        corners = np.array([
            c - half_outer_width * u_xy - half_depth * n_xy,
            c + half_outer_width * u_xy - half_depth * n_xy,
            c + half_outer_width * u_xy + half_depth * n_xy,
            c - half_outer_width * u_xy + half_depth * n_xy,
        ])
        ax.add_patch(Polygon(
            corners, closed=True, facecolor="none", edgecolor="black",
            linewidth=2.5, zorder=7, label="Physical gate" if i == 0 else None,
        ))
        ax.scatter(center[0], center[1], marker="x", s=45, color="black", zorder=8)
        ax.annotate(
            f"G{i + 1}", (center[0], center[1]), xytext=(5, -14),
            textcoords="offset points", fontsize=9, fontweight="bold", zorder=9,
        )

    if disturbance_history is not None:
        disturbance_history = np.asarray(disturbance_history, dtype=float)
        if disturbance_history.ndim == 2 and disturbance_history.shape[1] == 3:
            n = min(len(disturbance_history), len(X_closed_loop) - 1)
            if n > 0:
                stride = max(1, n // 20)
                idx = np.arange(0, n, stride)
                dxy = disturbance_history[idx, :2]
                norms = np.linalg.norm(dxy, axis=1, keepdims=True)
                dxy_unit = np.divide(
                    dxy, norms, out=np.zeros_like(dxy), where=norms > 1e-12
                )
                ax.quiver(
                    X_closed_loop[idx, 0], X_closed_loop[idx, 1],
                    dxy_unit[:, 0], dxy_unit[:, 1], angles="xy",
                    scale_units="xy", scale=2.0, width=0.004,
                    label="Adversarial disturbance direction", zorder=6,
                )

    final_error = np.linalg.norm(X_closed_loop[-1, :3] - X_plan[-1, :3])
    ax.set_title(
        "FATROP Receding-Horizon Tracking — No Gate Constraints\n"
        f"Final position error: {final_error:.3f} m"
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-down tracking plot to: {save_path}")


def plot_tracking_error(
    pos_errors,
    solve_times,
    save_path="fatrop_mpc_tracking_error.png",
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
        "FATROP MPC Position Tracking Error"
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
            "Track an offline FATROP minimum-time drone-racing "
            "trajectory using online receding-horizon FATROP MPC."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="multiphase_trajectory.npz",
        help=(
            "Offline FATROP trajectory NPZ "
            "(default: fatrop_replicated_gate_angle_experiment.npz)"
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
        default="fatrop_mpc_tracking.npz",
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
        disturbance_k=DISTURBANCE_K,
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
        gate_centers=plan["gate_geometry"]["gate_centers"],
        gate_normals=plan["gate_geometry"]["gate_normals"],
        gate_axis_u=plan["gate_geometry"]["gate_axis_u"],
        gate_axis_v=plan["gate_geometry"]["gate_axis_v"],
        gate_opening_half_widths=plan["gate_geometry"]["gate_opening_half_widths"],
        gate_bar_thickness=np.array(plan["gate_geometry"]["gate_bar_thickness"]),
        gate_depth=np.array(plan["gate_geometry"]["gate_depth"]),
        drone_radius=np.array(plan["gate_geometry"]["drone_radius"]),
        pos_errors=result["pos_errors"],
        state_errors=result["state_errors"],
        solve_times=result["solve_times"],
        objective_history=result["objective_history"],
        fallback_used=result["fallback_used"],
        mpc_horizon=np.array(args.horizon),
        disturbance_k=np.array(E_MAG_MODEL),
        e_mag_model=np.array(E_MAG_MODEL),
        disturbance_mag=np.array(DISTURBANCE_MAG),
        disturbance_injection=np.array("xy_position_plus_state_dependent_vy"),
        disturbance_history=result[
            "disturbance_history"
        ],
        disturbance_w_history=result[
            "disturbance_w_history"
        ],
        disturbance_gate_index_history=result[
            "disturbance_gate_index_history"
        ],
        disturbance_gate_distance_history=result[
            "disturbance_gate_distance_history"
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
        rollout_duration=np.array(
            result["rollout_duration"]
        ),
        rollout_duration_step=np.array(
            result["rollout_duration_step"]
        ),
        rollout_duration_search_start_step=np.array(
            result["rollout_duration_search_start_step"]
        ),
        min_goal_distance=np.array(
            result["min_goal_distance"]
        ),
        goal_distance_history=result[
            "goal_distance_history"
        ],
        constraint_violation_occurred=np.array(
            result["constraint_violation_occurred"]
        ),
        constraint_violation_step=np.array(
            result["constraint_violation_step"]
        ),
        constraint_violation_type=np.array(
            result["constraint_violation_type"]
        ),
        constraint_violation_gate_index=np.array(
            result["constraint_violation_gate_index"]
        ),
        constraint_violation_gate_bar=np.array(
            result["constraint_violation_gate_bar"]
        ),
        constraint_violation_amount=np.array(
            result["constraint_violation_amount"]
        ),
        enable_physical_state_bounds=np.array(
            ENABLE_PHYSICAL_STATE_BOUNDS
        ),
        check_rollout_state_bounds=np.array(
            CHECK_ROLLOUT_STATE_BOUNDS
        ),
        control_bound_tol=np.array(CONTROL_BOUND_TOL),
        pitch_max=np.array(PITCH_MAX),
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
        gate_geometry=plan["gate_geometry"],
        disturbance_history=result["disturbance_history"],
        save_path="fatrop_mpc_tracking_topdown.png",
    )

    plot_tracking_error(
        pos_errors=result["pos_errors"],
        solve_times=result["solve_times"],
        save_path="fatrop_mpc_tracking_error.png",
    )


if __name__ == "__main__":
    main()