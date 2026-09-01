from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_control_box_constraints, make_state_box_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_drone_racing import visualize_trajectory_and_gates

config.update("jax_enable_x64", False)

DTYPE = jnp.float64 if config.x64_enabled else jnp.float32
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

# -----------------------------
# 3D quadrotor parameters
# x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r, T]
# u = [T, tau_phi, tau_theta, tau_psi]
# -----------------------------
MASS = 1.0
GRAVITY = 9.81
E_MAG = 15.0
JX = 0.02
JY = 0.02
JZ = 0.04
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=DTYPE))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=DTYPE))

NUM_RANDOM = 5
NUM_ADV = 26


# -----------------------------
# Shared disturbance-field parameters
# -----------------------------
# These values are used by BOTH the optimization disturbance model and
# the top-down visualization, so changing them here changes both.
@dataclass(frozen=True)
class DisturbanceFieldParams:
    x_min: float = -2.0
    x_max: float = 1.0
    y_min: float = -8.0
    y_max: float = 0.0
    kx: float = 1.0
    ky: float = 1.0


DISTURBANCE_FIELD = DisturbanceFieldParams()


def disturbance_spatial_scale(px, py, params: DisturbanceFieldParams, xp):
    """Spatial window shared by the disturbance model and visualization."""
    x_window = (
        0.5 * (1.0 + xp.tanh(params.kx * (px - params.x_min)))
        *
        0.5 * (1.0 - xp.tanh(params.kx * (px - params.x_max)))
    )

    y_window = (
        0.5 * (1.0 + xp.tanh(params.ky * (py - params.y_min)))
        *
        0.5 * (1.0 - xp.tanh(params.ky * (py - params.y_max)))
    )

    return x_window * y_window


def rotation_matrix(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)

    # R = Rz(psi) Ry(theta) Rx(phi)
    return jnp.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                          cth * cphi],
    ], dtype=DTYPE)

def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta)
    cth = jnp.cos(theta)

    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=DTYPE)

def dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: Any,
) -> jnp.ndarray:
    """
    Multi-phase dynamics.

    State:
        x = [physical_state (12), T1, T2, ..., Tn]

    parameter should contain:
        waypoint_steps
        segment_lengths
    """

    waypoint_steps, segment_lengths = parameter

    num_phases = waypoint_steps.shape[0]

    # Determine which phase the current transition belongs to.
    #
    # t = 0, ..., N1-1       -> phase 0
    # t = N1, ..., N2-1      -> phase 1
    # t = N2, ..., N3-1      -> phase 2
    phase_idx = jnp.searchsorted(
        waypoint_steps,
        t,
        side="right",
    )

    # Safety in case dynamics is evaluated at the final state.
    phase_idx = jnp.clip(phase_idx, 0, num_phases - 1)

    # Duration corresponding to the active phase
    T_i = x[12 + phase_idx]

    # Number of discrete intervals in this phase
    N_i = segment_lengths[phase_idx]

    # Physical timestep for this phase
    dt = T_i / N_i

    return rigid_body_3d_step(x, u, dt)

def rigid_body_3d_step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:

    px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r = x[:12]

    T, tau_phi, tau_theta, tau_psi = u

    v = jnp.array([vx, vy, vz], dtype=x.dtype)
    omega = jnp.array([p, q, r], dtype=x.dtype)
    tau = jnp.array(
        [tau_phi, tau_theta, tau_psi],
        dtype=x.dtype,
    )

    R = rotation_matrix(phi, theta, psi)
    E = euler_angle_rates_matrix(phi, theta)

    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    # Translational dynamics
    pos_dot = v
    v_dot = (R @ (T * e3)) / MASS - GRAVITY * e3

    # Rotational dynamics
    euler_dot = E @ omega
    omega_dot = J_INV @ (
        tau - jnp.cross(omega, J @ omega)
    )

    x_dot = jnp.concatenate([
        pos_dot,
        euler_dot,
        v_dot,
        omega_dot,
    ])

    physical_next = x[:12] + dt * x_dot

    # All duration variables are constant states:
    # T_i,k+1 = T_i,k
    return jnp.concatenate([
        physical_next,
        x[12:],
    ])


def cost(W, reference, x, u, t):
    """
    W =
    [wpx, wpy, wpz,
     wphi, wtheta, wpsi,
     wvx, wvy, wvz,
     wp, wq, wr,
     wT, wtau_phi, wtau_theta, wtau_psi, wtime]
    """
    (
        wpx, wpy, wpz,
        wphi, wtheta, wpsi,
        wvx, wvy, wvz,
        wp, wq, wr,
        wT, wtau_phi, wtau_theta, wtau_psi
    ) = W[:16]

    wtime = W[16:]

    xref = reference[t]

    dpos = x[:3] - xref[:3]
    dang = x[3:6] - xref[3:6]
    dvel = x[6:9] - xref[6:9]
    drates = x[9:12] - xref[9:12]

    T_hover = MASS * GRAVITY
    du = jnp.array([
        u[0] - T_hover,
        u[1],
        u[2],
        u[3],
    ], dtype=x.dtype)

    angle_cost = (
        wphi * (1.0 - jnp.cos(dang[0]))
        + wtheta * (1.0 - jnp.cos(dang[1]))
        + wpsi * (1.0 - jnp.cos(dang[2]))
    )

    durations = x[12:]
    time_cost = jnp.dot(wtime, durations)

    return (
        wpx * dpos[0] ** 2
        + wpy * dpos[1] ** 2
        + wpz * dpos[2] ** 2
        + angle_cost
        + wvx * dvel[0] ** 2
        + wvy * dvel[1] ** 2
        + wvz * dvel[2] ** 2
        + wp * drates[0] ** 2
        + wq * drates[1] ** 2
        + wr * drates[2] ** 2
        + wT * du[0] ** 2
        + wtau_phi * du[1] ** 2
        + wtau_theta * du[2] ** 2
        + wtau_psi * du[3] ** 2
        + time_cost
    )

def build_piecewise_reference(
    x0: jnp.ndarray,
    waypoint_centers: jnp.ndarray,
    waypoint_steps: jnp.ndarray,
    initial_durations: jnp.ndarray,
    N: int,
) -> jnp.ndarray:
    """
    Build a piecewise straight-line reference trajectory:

        x0 -> waypoint_1 -> waypoint_2 -> ... -> waypoint_n

    Each segment ends at its corresponding waypoint_steps[i].

    State ordering:
        [px, py, pz,
         phi, theta, psi,
         vx, vy, vz,
         p, q, r,
         T1, ..., Tn]
    """

    num_phases = waypoint_centers.shape[0]
    nx = 12 + num_phases

    X_ref = jnp.zeros((N + 1, nx), dtype=DTYPE)

    # --------------------------------------------------
    # Duration states are constant reference values
    # --------------------------------------------------
    X_ref = X_ref.at[:, 12:].set(
        jnp.broadcast_to(initial_durations, (N + 1, num_phases))
    )

    # --------------------------------------------------
    # Piecewise position / velocity reference
    # --------------------------------------------------
    start_pos = x0[:3]
    start_step = 0

    for i in range(num_phases):
        end_pos = waypoint_centers[i]
        end_step = int(waypoint_steps[i])

        segment_length = end_step - start_step

        # Includes both endpoints
        alpha = jnp.linspace(
            0.0,
            1.0,
            segment_length + 1
        )

        # Straight line between current point and waypoint
        pos_segment = (
            (1.0 - alpha[:, None]) * start_pos
            + alpha[:, None] * end_pos
        )

        X_ref = X_ref.at[start_step:end_step + 1, :3].set(
            pos_segment
        )

        # Constant velocity reference for this phase
        vel_segment = (
            end_pos - start_pos
        ) / initial_durations[i]

        X_ref = X_ref.at[start_step:end_step + 1, 6:9].set(
            vel_segment
        )

        # Next phase starts here
        start_pos = end_pos
        start_step = end_step

    return X_ref

def make_waypoint_constraints(
    waypoint_steps: jnp.ndarray,
    waypoint_centers: jnp.ndarray,
    waypoint_axis_u: jnp.ndarray,
    waypoint_axis_v: jnp.ndarray,
    waypoint_normals: jnp.ndarray,
    waypoint_half_widths_local: jnp.ndarray,
):
    """
    Enforce an ORIENTED waypoint box at each waypoint step.

    The waypoint box is expressed in the corresponding gate-local frame:

        u : horizontal direction across the gate
        v : vertical direction (+z)
        n : gate traversal direction / gate normal

    For waypoint i, define

        delta = position - waypoint_center[i]

        q_u = u_i^T delta
        q_v = v_i^T delta
        q_n = n_i^T delta

    and enforce

        |q_u| <= half_width_u
        |q_v| <= half_width_v
        |q_n| <= half_width_n

    at t == waypoint_steps[i].

    waypoint_half_widths_local is ordered [u, v, n].
    """
    waypoint_steps = jnp.asarray(waypoint_steps)
    waypoint_centers = jnp.asarray(waypoint_centers)
    waypoint_axis_u = jnp.asarray(waypoint_axis_u)
    waypoint_axis_v = jnp.asarray(waypoint_axis_v)
    waypoint_normals = jnp.asarray(waypoint_normals)
    waypoint_half_widths_local = jnp.asarray(
        waypoint_half_widths_local
    )

    def constraints(x, u, t):
        pos = x[:3]

        # Shape: (NUM_PHASES, 3)
        delta = pos[None, :] - waypoint_centers

        q_u = jnp.sum(delta * waypoint_axis_u, axis=1)
        q_v = jnp.sum(delta * waypoint_axis_v, axis=1)
        q_n = jnp.sum(delta * waypoint_normals, axis=1)

        q = jnp.stack([q_u, q_v, q_n], axis=1)

        # |q| <= h  ->  q - h <= 0 and -q - h <= 0
        upper = q - waypoint_half_widths_local
        lower = -q - waypoint_half_widths_local

        waypoint_constraints = jnp.concatenate(
            [upper, lower],
            axis=1,
        )

        active = (t == waypoint_steps)

        waypoint_constraints = jnp.where(
            active[:, None],
            waypoint_constraints,
            -jnp.ones_like(waypoint_constraints),
        )

        return waypoint_constraints.reshape(-1)

    return constraints


def build_gate_axes_from_path(
    start_position: jnp.ndarray,
    gate_centers: jnp.ndarray,
):
    """
    Build vertical gate frames whose normal points along the incoming path.

    For gate i, the desired traversal direction is

        previous_position -> gate_center[i].

    The gate is kept vertical, so its normal is projected into the XY plane.
    Local gate coordinates are:

        u : horizontal direction across the gate opening
        v : world vertical (+z)
        n : desired direction through the gate
    """
    start_position = jnp.asarray(start_position)
    gate_centers = jnp.asarray(gate_centers)

    previous = jnp.concatenate([
        start_position[None, :],
        gate_centers[:-1],
    ], axis=0)

    incoming = gate_centers - previous

    # Keep gates vertical: normal lies in the XY plane.
    normals = incoming.at[:, 2].set(0.0)
    normal_norm = jnp.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / jnp.maximum(normal_norm, 1e-8)

    axis_v = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 1.0], dtype=gate_centers.dtype),
        normals.shape,
    )

    # u x v = n, equivalently u = v x n for this convention.
    axis_u = jnp.cross(axis_v, normals)
    axis_u = axis_u / jnp.maximum(
        jnp.linalg.norm(axis_u, axis=1, keepdims=True),
        1e-8,
    )

    return normals, axis_u, axis_v


def rotate_gate_orientation(
    gate_normals,
    gate_axis_u,
    gate_axis_v,
    gate_idx: int,
    angle_deg: float,
    invert: bool = False,
):
    """
    Apply an additional yaw rotation to one vertical gate.

    Parameters
    ----------
    gate_idx
        Zero-based gate index:
            Gate 1 -> 0
            Gate 2 -> 1
            Gate 3 -> 2
            Gate 4 -> 3
            Gate 5 -> 4

    angle_deg
        Rotation in degrees about world +z, applied relative to the
        automatically generated incoming-path orientation.

        Positive -> counter-clockwise in the XY plane.
        Negative -> clockwise in the XY plane.

    invert
        If True, flip the automatically generated normal first and then
        apply angle_deg. This preserves the previous Gate 3 convention.

    Returns
    -------
    gate_normals, gate_axis_u
        Updated arrays. gate_axis_v remains unchanged because all gates
        stay vertical.
    """
    n = gate_normals[gate_idx]

    if invert:
        n = -n

    angle = jnp.deg2rad(
        jnp.asarray(angle_deg, dtype=gate_normals.dtype)
    )
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    Rz = jnp.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=gate_normals.dtype)

    # Rotate the traversal normal.
    n = Rz @ n
    n = n / jnp.maximum(jnp.linalg.norm(n), 1e-8)

    # Recompute horizontal gate axis so u x v = n.
    v = gate_axis_v[gate_idx]
    u = jnp.cross(v, n)
    u = u / jnp.maximum(jnp.linalg.norm(u), 1e-8)

    gate_normals = gate_normals.at[gate_idx].set(n)
    gate_axis_u = gate_axis_u.at[gate_idx].set(u)

    return gate_normals, gate_axis_u


def build_gate_transition_waypoints(
    gate_centers: jnp.ndarray,
    gate_normals: jnp.ndarray,
    transition_offset: float,
):
    """
    Create exactly two optimization waypoints per physical gate:

        pre_i  = gate_center_i - transition_offset * gate_normal_i
        post_i = gate_center_i + transition_offset * gate_normal_i

    The waypoint order is therefore

        pre_1 -> post_1 -> pre_2 -> post_2 -> ...

    so every gate transition is explicitly represented by a point on the
    incoming side and a point on the outgoing side. The physical gate itself
    remains centered at gate_centers[i].
    """
    gate_centers = jnp.asarray(gate_centers)
    gate_normals = jnp.asarray(gate_normals)

    offset = jnp.asarray(
        transition_offset,
        dtype=gate_centers.dtype,
    )

    pre_gate = gate_centers - offset * gate_normals
    post_gate = gate_centers + offset * gate_normals

    # Interleave [pre_1, post_1, pre_2, post_2, ...].
    transition_waypoints = jnp.stack(
        [pre_gate, post_gate],
        axis=1,
    ).reshape(-1, 3)

    return transition_waypoints, pre_gate, post_gate


def make_gate_direction_constraints(
    gate_steps: jnp.ndarray,
    gate_normals: jnp.ndarray,
    min_forward_speed: float = 0.25,
):
    """
    At gate_steps[i], enforce traversal in the +normal direction:

        gate_normal[i]^T velocity >= min_forward_speed.

    Returned inequalities use the solver convention g(x,u,t) <= 0.
    """
    gate_steps = jnp.asarray(gate_steps)
    gate_normals = jnp.asarray(gate_normals)

    def constraints(x, u, t):
        vel = x[6:9]

        forward_speed = jnp.sum(
            gate_normals * vel[None, :],
            axis=1,
        )

        gate_constraint = min_forward_speed - forward_speed
        active = (t == gate_steps)

        return jnp.where(
            active,
            gate_constraint,
            -jnp.ones_like(gate_constraint),
        )

    return constraints


def make_gate_frame_constraints(
    gate_centers: jnp.ndarray,
    gate_normals: jnp.ndarray,
    gate_axis_u: jnp.ndarray,
    gate_axis_v: jnp.ndarray,
    opening_half_widths: jnp.ndarray,
    bar_thickness: float = 0.10,
    gate_depth: float = 0.10,
    drone_radius: float = 0.10,
):
    """
    Model every gate as four oriented hyperbox obstacles:

                    top
             +---------------+
             |               |
        left |    opening    | right
             |               |
             +---------------+
                   bottom

    Each bar contributes exactly one outside-hyperbox constraint.
    With q=[q_u,q_v,q_n] the point expressed in a bar's local frame and
    h its half-sizes, a point is outside the box iff

        max(|q| - h) >= 0.

    The solver uses g <= 0, therefore we return

        g = -max(|q| - h) <= 0.

    All bar half-sizes are inflated by drone_radius so x[:3] can be treated
    as the drone center. The obstacle constraints are active at every node.

    opening_half_widths has shape (num_gates, 2):
        [:, 0] = horizontal half-width along gate_axis_u
        [:, 1] = vertical half-height along gate_axis_v
    """
    gate_centers = jnp.asarray(gate_centers)
    gate_normals = jnp.asarray(gate_normals)
    gate_axis_u = jnp.asarray(gate_axis_u)
    gate_axis_v = jnp.asarray(gate_axis_v)
    opening_half_widths = jnp.asarray(opening_half_widths)

    num_gates = gate_centers.shape[0]
    hu = opening_half_widths[:, 0]
    hv = opening_half_widths[:, 1]

    t_bar = jnp.asarray(bar_thickness, dtype=gate_centers.dtype)
    half_depth = jnp.asarray(0.5 * gate_depth, dtype=gate_centers.dtype)
    radius = jnp.asarray(drone_radius, dtype=gate_centers.dtype)

    # Bar centers in local [u, v, n] coordinates.
    zeros = jnp.zeros(num_gates, dtype=gate_centers.dtype)

    top_center_local = jnp.stack([
        zeros,
        hv + 0.5 * t_bar,
        zeros,
    ], axis=1)

    bottom_center_local = jnp.stack([
        zeros,
        -(hv + 0.5 * t_bar),
        zeros,
    ], axis=1)

    left_center_local = jnp.stack([
        -(hu + 0.5 * t_bar),
        zeros,
        zeros,
    ], axis=1)

    right_center_local = jnp.stack([
        hu + 0.5 * t_bar,
        zeros,
        zeros,
    ], axis=1)

    # Top/bottom span the opening plus both side bars.
    top_bottom_half_sizes = jnp.stack([
        hu + t_bar,
        0.5 * t_bar * jnp.ones_like(hu),
        half_depth * jnp.ones_like(hu),
    ], axis=1)

    # Left/right fill the vertical sides between top and bottom.
    left_right_half_sizes = jnp.stack([
        0.5 * t_bar * jnp.ones_like(hu),
        hv,
        half_depth * jnp.ones_like(hu),
    ], axis=1)

    # Inflate every bar by the drone collision radius.
    top_bottom_half_sizes = top_bottom_half_sizes + radius
    left_right_half_sizes = left_right_half_sizes + radius

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

        # Positive outside_margin means the point lies outside in at least
        # one local coordinate. Negative means it lies inside the box.
        outside_margin = jnp.max(
            jnp.abs(q) - half_sizes,
            axis=1,
        )

        return -outside_margin

    def constraints(x, u, t):
        pos = x[:3]

        top = outside_box_constraint(
            pos, top_centers, top_bottom_half_sizes
        )
        bottom = outside_box_constraint(
            pos, bottom_centers, top_bottom_half_sizes
        )
        left = outside_box_constraint(
            pos, left_centers, left_right_half_sizes
        )
        right = outside_box_constraint(
            pos, right_centers, left_right_half_sizes
        )

        # Shape (num_gates, 4), then flattened. Exactly four constraints
        # per gate: top, bottom, left, right.
        return jnp.stack(
            [top, bottom, left, right],
            axis=1,
        ).reshape(-1)

    return constraints


def make_min_time_disturbance(
    n: int,
    E_mag,
    NUM_PHASES,
    WAYPOINT_STEPS,
    SEGMENT_LENGTHS,
    field_params: DisturbanceFieldParams = DISTURBANCE_FIELD,
):
    WAYPOINT_STEPS = jnp.asarray(WAYPOINT_STEPS)
    SEGMENT_LENGTHS = jnp.asarray(SEGMENT_LENGTHS)

    def disturbance_at_state(
        x_k: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        phase_idx = jnp.searchsorted(
            WAYPOINT_STEPS,
            t,
            side="right",
        )

        phase_idx = jnp.clip(
            phase_idx,
            0,
            NUM_PHASES - 1,
        )

        T_i = x_k[12 + phase_idx]
        N_i = SEGMENT_LENGTHS[phase_idx]
        dt = T_i / N_i

        px = x_k[0]
        py = x_k[1]

        # -------------------------------------------
        # Spatial disturbance field
        # -------------------------------------------
        # Uses the same helper/parameters as the visualization.
        spatial_scale = disturbance_spatial_scale(
            px,
            py,
            field_params,
            xp=jnp,
        )

        # Disturbance acts only on vy
        E_k = jnp.zeros((n, n), dtype=x_k.dtype)

        E_k = E_k.at[7, 7].set(
            dt * E_mag * spatial_scale
        )

        return E_k

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        ts = jnp.arange(X.shape[0])

        return jax.vmap(
            disturbance_at_state,
            in_axes=(0, 0),
        )(X, ts)

    disturbance.at_state = disturbance_at_state

    return disturbance

def build_controller(
    N,
    X_IN,
    WAYPOINT_STEPS,
    WAYPOINT_CENTERS,
    WAYPOINT_HALF_WIDTHS,
    WAYPOINT_AXIS_U,
    WAYPOINT_AXIS_V,
    WAYPOINT_NORMALS,
    SEGMENT_LENGTHS,
    GATE_CENTERS,
    GATE_NORMALS,
    GATE_AXIS_U,
    GATE_AXIS_V,
    GATE_OPENING_HALF_WIDTHS,
    GATE_BAR_THICKNESS=0.10,
    GATE_DEPTH=0.10,
    DRONE_RADIUS=0.10,
    disturbance_field: DisturbanceFieldParams = DISTURBANCE_FIELD,
):
    # -----------------------------
    # Dimensions
    # -----------------------------
    NUM_PHASES = len(WAYPOINT_STEPS)
    n = 12 + NUM_PHASES
    nu = 4
    TIME_WEIGHT = 5.0

    # -----------------------------
    # Cost weights
    # -----------------------------
    W = jnp.concatenate([jnp.array([
            0.001, 0.001, 0.001,        # position
            0.01, 0.01, 0.01,        # roll, pitch, yaw
            0.01, 0.01, 0.01,        # velocities
            0.01, 0.01, 0.01,        # body rates
            0.01, 0.01, 0.01, 0.01,  # control
        ], dtype=DTYPE),
        TIME_WEIGHT * jnp.ones(NUM_PHASES),  # time
    ])

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=DTYPE),
    )

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = MASS * GRAVITY
    T_max = 2.0 * T_hover
    tau_max = 10.0

    u_min = jnp.array([0.0, -tau_max, -tau_max, -tau_max], dtype=DTYPE)
    u_max = jnp.array([T_max, tau_max, tau_max, tau_max], dtype=DTYPE)
    constraints_u = make_control_box_constraints(u_min, u_max)

    MIN_TIME = 0.005
    MAX_TIME = 20.0

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.concatenate(
        [jnp.array([
            15.0, 15.0, 15.0,       # px, py, pz
            jnp.pi / 2.0,           # phi
            jnp.pi / 2.0,           # theta
            10.0 * jnp.pi,          # psi
            5.0, 5.0, 5.0,          # vx, vy, vz
            8.0, 8.0, 8.0,          # p, q, r
        ], dtype=DTYPE),
        MAX_TIME * jnp.ones(NUM_PHASES)
    ])
    x_min = -x_max
    x_min = x_min.at[2].set(-1.0)
    x_min = x_min.at[-NUM_PHASES:].set(MIN_TIME)

    constraints_x = make_state_box_constraints(x_min, x_max)
    waypoint_constraint = make_waypoint_constraints(
        waypoint_steps=WAYPOINT_STEPS,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_axis_u=WAYPOINT_AXIS_U,
        waypoint_axis_v=WAYPOINT_AXIS_V,
        waypoint_normals=WAYPOINT_NORMALS,
        waypoint_half_widths_local=WAYPOINT_HALF_WIDTHS,
    )

    gate_frame_constraint = make_gate_frame_constraints(
        gate_centers=GATE_CENTERS,
        gate_normals=GATE_NORMALS,
        gate_axis_u=GATE_AXIS_U,
        gate_axis_v=GATE_AXIS_V,
        opening_half_widths=GATE_OPENING_HALF_WIDTHS,
        bar_thickness=GATE_BAR_THICKNESS,
        gate_depth=GATE_DEPTH,
        drone_radius=DRONE_RADIUS,
    )

    constraints_all = combine_constraints(
        constraints_x,
        constraints_u,
        waypoint_constraint,
        gate_frame_constraint,
    )

    obstacles = jnp.zeros((0, 3))

    disturbance = make_min_time_disturbance(
        n=n,
        E_mag=E_MAG,
        NUM_PHASES=NUM_PHASES,
        WAYPOINT_STEPS=WAYPOINT_STEPS,
        SEGMENT_LENGTHS=SEGMENT_LENGTHS,
        field_params=disturbance_field,
    )

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=1e-2,
        eps_rel=1e-2,
        rho_max=1e6,
        max_iterations=3000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
        num_phases=NUM_PHASES,
    )

    # Q_bar = jnp.broadcast_to(jnp.eye(n), (N +1, n, n)).at[..., 1, 1].set(1000).at[..., 7, 7].set(0.001)
    # Q_bar = Q_bar + 1e-6 * jnp.eye(n)[None, :, :]
    # R_bar = jnp.broadcast_to(jnp.eye(nu) * 0.5, (N, nu, nu))
    Q_bar = jnp.broadcast_to(jnp.eye(n), (N +1, n, n))
    R_bar = jnp.broadcast_to(jnp.eye(nu), (N, nu, nu))

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=True,
        rti=False,
        gradient_window=10,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=10,
        line_search=True,
        lm_regularization=1e-2,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=obstacles,
        cost=cost,
        disturbance=disturbance,
        shift=1,
        X_in=X_IN,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=DTYPE).at[:, 0].set(T_hover),
        Q_bar=Q_bar,
        R_bar=R_bar,
    )
    return controller

def plot_top_down_trajectory(
    X,
    waypoint_centers,
    waypoint_half_widths=None,
    waypoint_axis_u=None,
    waypoint_normals=None,
    gate_centers=None,
    gate_normals=None,
    gate_axis_u=None,
    gate_opening_half_widths=None,
    gate_bar_thickness=0.10,
    gate_depth=0.10,
    drone_radius=0.10,
    show_gate_collision_inflation=False,
    reference=None,
    Phi_x=None,
    E_mag=0.05,
    disturbance_field: DisturbanceFieldParams = DISTURBANCE_FIELD,
    tube_stride=1,
    min_time=None,
    save_path="trajectory_topdown.png",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch, Polygon
    from matplotlib.colors import LinearSegmentedColormap

    X = np.asarray(X)
    waypoints = np.asarray(waypoint_centers)

    gates = (
        np.asarray(gate_centers, dtype=float)
        if gate_centers is not None
        else None
    )
    gate_n = (
        np.asarray(gate_normals, dtype=float)
        if gate_normals is not None
        else None
    )
    gate_u = (
        np.asarray(gate_axis_u, dtype=float)
        if gate_axis_u is not None
        else None
    )
    gate_openings = (
        np.asarray(gate_opening_half_widths, dtype=float)
        if gate_opening_half_widths is not None
        else None
    )

    if reference is not None:
        ref = np.asarray(reference)
    else:
        ref = None

    # ==================================================
    # Plot bounds
    # ==================================================
    xy_data = [X[:, :2], waypoints[:, :2]]

    if gates is not None:
        xy_data.append(gates[:, :2])

    if ref is not None:
        xy_data.append(ref[:, :2])

    all_xy = np.concatenate(xy_data, axis=0)

    margin = 0.75

    # Bounds required by trajectory / waypoints / reference
    data_x_min = np.min(all_xy[:, 0]) - margin
    data_x_max = np.max(all_xy[:, 0]) + margin
    data_y_min = np.min(all_xy[:, 1]) - margin
    data_y_max = np.max(all_xy[:, 1]) + margin

    # Bounds required to show the disturbance-field tails
    tail_width = 4.0

    field_x_min = (
        disturbance_field.x_min
        - tail_width / disturbance_field.kx
    )
    field_x_max = (
        disturbance_field.x_max
        + tail_width / disturbance_field.kx
    )

    field_y_min = (
        disturbance_field.y_min
        - tail_width / disturbance_field.ky
    )
    field_y_max = (
        disturbance_field.y_max
        + tail_width / disturbance_field.ky
    )

    # Final plot bounds include both
    # x_min = min(data_x_min, field_x_min)
    # x_max = max(data_x_max, field_x_max)

    # y_min = min(data_y_min, field_y_min)
    # y_max = max(data_y_max, field_y_max)
    x_min = -6
    x_max = 5
    y_min = -7.5
    y_max = 8

    fig, ax = plt.subplots(figsize=(9, 8))

    # ==================================================
    # Disturbance field
    #
    # Uses the exact same helper and parameters as
    # make_min_time_disturbance().
    #
    # disturbance acts only in vy
    # ==================================================
    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)

    XX, YY = np.meshgrid(xs, ys)

    spatial_scale = disturbance_spatial_scale(
        XX,
        YY,
        disturbance_field,
        xp=np,
    )

    disturbance_field_values = E_mag * spatial_scale

    # White at zero, then smoothly transition into magma as the
    # disturbance magnitude increases.  The color scale is pinned
    # to [0, E_mag], so a true zero always renders as pure white.
    magma = plt.get_cmap("magma")
    disturbance_cmap = LinearSegmentedColormap.from_list(
        "white_to_magma",
        [
            (0.00, "white"),
            (0.08, magma(0.18)),
            (0.35, magma(0.40)),
            (0.65, magma(0.68)),
            (1.00, magma(1.00)),
        ],
    )

    # Explicit levels starting at zero guarantee that the bottom of
    # the colorbar/field corresponds to white.  No contour-line
    # overlay is drawn.
    field_levels = np.linspace(0.0, float(E_mag), 100)

    field = ax.contourf(
        XX,
        YY,
        disturbance_field_values,
        levels=field_levels,
        cmap=disturbance_cmap,
        vmin=0.0,
        vmax=float(E_mag),
        alpha=0.85,
        zorder=0,
    )

    cbar = fig.colorbar(field, ax=ax, pad=0.02)
    cbar.set_label("Disturbance magnitude on $v_y$")

    # ==================================================
    # SLS tubes
    #
    # EXACT tube calculation:
    #
    # get_trajectory_tubes(Phi_x)
    # =
    # norm(Phi_x, axis=-1).sum(axis=1)
    # ==================================================
    tube_handle = None

    if Phi_x is not None:

        tube = np.asarray(
            get_trajectory_tubes(Phi_x)
        )

        # print(tube)

        print("X shape:", X.shape)
        print("Phi_x shape:", np.asarray(Phi_x).shape)
        print("tube shape:", tube.shape)

        # Usually both are N+1.
        # Handle N tube nodes just in case x0 is omitted.
        if tube.shape[0] == X.shape[0] - 1:
            tube = np.concatenate(
                [
                    np.zeros(
                        (1, tube.shape[1]),
                        dtype=tube.dtype,
                    ),
                    tube,
                ],
                axis=0,
            )

        if tube.shape[0] != X.shape[0]:
            raise ValueError(
                f"Tube/X mismatch: "
                f"tube={tube.shape}, X={X.shape}"
            )

        # ----------------------------------------------
        # Draw XY projection of the box tube
        #
        # center:
        #   (X[k,0], X[k,1])
        #
        # half widths:
        #   tube[k,0], tube[k,1]
        # ----------------------------------------------
        for k in range(
            0,
            X.shape[0],
            max(1, tube_stride),
        ):

            cx = X[k, 0]
            cy = X[k, 1]

            rx = tube[k, 0]
            ry = tube[k, 1]

            if not np.all(
                np.isfinite([cx, cy, rx, ry])
            ):
                continue

            rect = Rectangle(
                (cx - rx, cy - ry),
                2.0 * rx,
                2.0 * ry,
                facecolor="cyan",
                edgecolor="blue",
                linewidth=0.6,
                alpha=0.18,
                zorder=2,
            )

            ax.add_patch(rect)

        tube_handle = Patch(
            facecolor="cyan",
            edgecolor="blue",
            alpha=0.18,
            label="SLS tube",
        )

    # ==================================================
    # Reference trajectory
    # ==================================================
    if ref is not None:
        ax.plot(
            ref[:, 0],
            ref[:, 1],
            "--",
            linewidth=1.5,
            label="Reference",
            zorder=4,
        )

    # ==================================================
    # Optimized trajectory
    # ==================================================
    ax.plot(
        X[:, 0],
        X[:, 1],
        "-o",
        markersize=2,
        linewidth=2,
        label="Optimized trajectory",
        zorder=5,
    )

    # ==================================================
    # Start
    # ==================================================
    ax.scatter(
        X[0, 0],
        X[0, 1],
        s=100,
        marker="x",
        linewidths=3,
        label="Start",
        zorder=7,
    )

    # ==================================================
    # Physical gates -- XY/top-down projection
    #
    # A vertical square gate projects from above to an oriented rectangular
    # footprint. The physical outer width is:
    #
    #     opening_width + 2 * bar_thickness
    #
    # and its thickness along the gate normal is gate_depth.
    # ==================================================
    gate_handle = None

    if (
        gates is not None
        and gate_n is not None
        and gate_u is not None
        and gate_openings is not None
    ):
        for i, center in enumerate(gates):
            u_xy = np.asarray(gate_u[i, :2], dtype=float)
            n_xy = np.asarray(gate_n[i, :2], dtype=float)

            u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
            n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

            # Physical gate dimensions in top view.
            hu_open = float(gate_openings[i, 0])
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

            # Optional drone-radius-inflated collision footprint.
            if show_gate_collision_inflation:
                inflated_half_width = half_outer_width + float(drone_radius)
                inflated_half_depth = half_depth + float(drone_radius)

                inflated_corners = np.array([
                    c - inflated_half_width * u_xy - inflated_half_depth * n_xy,
                    c + inflated_half_width * u_xy - inflated_half_depth * n_xy,
                    c + inflated_half_width * u_xy + inflated_half_depth * n_xy,
                    c - inflated_half_width * u_xy + inflated_half_depth * n_xy,
                ])

                inflated_patch = Polygon(
                    inflated_corners,
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.55,
                    zorder=6,
                )
                ax.add_patch(inflated_patch)

            # Gate center + label.
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
    # Waypoints
    # ==================================================
    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=90,
        marker="s",
        label="Waypoints",
        zorder=7,
    )

    for i, waypoint in enumerate(waypoints):
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
    # Waypoint boxes
    #
    # If waypoint_axis_u and waypoint_normals are supplied, draw the actual
    # oriented top-down projection of each local [u, v, n] waypoint box.
    # In XY, this rectangle has:
    #   half-width along u = waypoint_half_widths[:, 0]
    #   half-width along n = waypoint_half_widths[:, 2]
    # ==================================================
    if waypoint_half_widths is not None:

        half_widths = np.asarray(
            waypoint_half_widths
        )

        if waypoint_axis_u is not None and waypoint_normals is not None:
            waypoint_u = np.asarray(waypoint_axis_u)
            waypoint_n = np.asarray(waypoint_normals)

            for center, hw, u_axis, n_axis in zip(
                waypoints,
                half_widths,
                waypoint_u,
                waypoint_n,
            ):
                c = center[:2]
                u_xy = u_axis[:2]
                n_xy = n_axis[:2]

                u_xy = u_xy / max(np.linalg.norm(u_xy), 1e-12)
                n_xy = n_xy / max(np.linalg.norm(n_xy), 1e-12)

                hu = hw[0]
                hn = hw[2]

                corners = np.array([
                    c - hu * u_xy - hn * n_xy,
                    c + hu * u_xy - hn * n_xy,
                    c + hu * u_xy + hn * n_xy,
                    c - hu * u_xy + hn * n_xy,
                ])

                patch = plt.Polygon(
                    corners,
                    closed=True,
                    fill=False,
                    linestyle="--",
                    linewidth=1.5,
                    zorder=6,
                )
                ax.add_patch(patch)

        else:
            # Backward-compatible axis-aligned rendering.
            for center, hw in zip(
                waypoints,
                half_widths,
            ):
                rect = Rectangle(
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

                ax.add_patch(rect)

    # ==================================================
    # Formatting
    # ==================================================
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    if min_time is not None:
        ax.set_title(
            f"Total Min Time: {float(min_time):.3f} s\n"
            "Top-Down Trajectory, Physical Gates, Disturbance Field, and SLS Tube"
        )
    else:
        ax.set_title(
            "Top-Down Trajectory, Physical Gates, Disturbance Field, and SLS Tube"
        )

    # ax.axis("equal")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()

    if tube_handle is not None:
        handles.append(tube_handle)
        labels.append("SLS tube")

    if gate_handle is not None:
        handles.append(gate_handle)
        labels.append("Physical gate")

    # ax.legend(handles, labels)

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved top-down trajectory to: {save_path}"
    )

def main():
    # --------------------------------------------------
    # Five physical gates.
    #
    # Each gate now contributes TWO optimization waypoints:
    #
    #       pre-gate waypoint  ->  physical gate  ->  post-gate waypoint
    #
    # Therefore 5 gates -> 10 waypoint phases.
    # --------------------------------------------------
    NUM_GATES = 5

    # Exactly 3 shooting intervals between every optimization waypoint.
    # Use a short horizon through each gate and a longer horizon between gates.
    #
    #   start -> pre_1       : 20 steps
    #   pre_i -> post_i      :  3 steps   (gate crossing)
    #   post_i -> pre_{i+1}  : 20 steps   (long travel segment)
    #
    # For 5 gates this gives N = 5 * (20 + 3) = 115 shooting intervals.
    GATE_CROSSING_STEPS = 3
    LONG_SEGMENT_STEPS = 20
    N = NUM_GATES * (LONG_SEGMENT_STEPS + GATE_CROSSING_STEPS)
    nu = 4

    # --------------------------------------------------
    # Physical gate centers
    # --------------------------------------------------
    GATE_CENTERS = np.array([
            [ 3.5,  -1.0, 1.0],
            [ 3.0,   7.0, 1.0],
            [-3.5,   5.0, 1.0],
            [-1.0,  3.0, 1.0],
            [-3.5,  -7.0, 1.0],
        ], dtype=float)

    # --------------------------------------------------
    # Physical gate geometry
    # --------------------------------------------------
    # 1.0 m x 1.0 m square opening.
    GATE_OPENING_HALF_WIDTHS = jnp.tile(
        jnp.array([[0.50, 0.50]], dtype=DTYPE),
        (NUM_GATES, 1),
    )

    GATE_BAR_THICKNESS = 0.10
    GATE_DEPTH = 0.10
    DRONE_RADIUS = 0.10

    # Distance from the gate plane to each transition waypoint.
    #
    # The optimizer will visit:
    #
    #   center - offset*n   (front / incoming side)
    #   center + offset*n   (behind / outgoing side)
    #
    # for every physical gate.
    GATE_TRANSITION_OFFSET = 0.30

    # Oriented box around each pre/post transition point.
    #
    # Local gate-frame dimensions [u, v, n]:
    #   u = 0.8 m  across the gate
    #   v = 0.3 m  vertically
    #   n = 0.2 m  along the gate traversal direction
    #
    # Therefore local half-widths [u, v, n] are:
    #   [0.40, 0.15, 0.10]
    TRANSITION_WAYPOINT_HALF_WIDTHS = jnp.array(
        [0.40, 0.15, 0.10],
        dtype=DTYPE,
    )

    # --------------------------------------------------
    # Initial physical state
    # --------------------------------------------------
    physical_x0 = jnp.array([
        3.5, -3.0, 1.0,   # px, py, pz
        0.0,  0.0, 0.0,   # phi, theta, psi
        0.0,  0.0, 0.0,   # vx, vy, vz
        0.0,  0.0, 0.0,   # p, q, r
    ], dtype=DTYPE)

    # --------------------------------------------------
    # Gate orientation
    # --------------------------------------------------
    # Gate normals point from the previous physical gate/start toward the
    # current physical gate. Gates remain vertical.
    GATE_NORMALS, GATE_AXIS_U, GATE_AXIS_V = build_gate_axes_from_path(
        start_position=physical_x0[:3],
        gate_centers=GATE_CENTERS,
    )

    # --------------------------------------------------
    # Adjustable orientations for ALL physical gates
    #
    # Each angle is an OFFSET relative to the automatically generated
    # incoming-path orientation above.
    #
    # Positive angle -> counter-clockwise in top-down XY view.
    # Negative angle -> clockwise in top-down XY view.
    #
    # Set *_INVERT=True if you want to flip a gate normal by 180 degrees
    # BEFORE applying its angle offset.
    #
    # Current defaults preserve the previous behavior:
    #   Gate 1: automatic orientation
    #   Gate 2: automatic orientation
    #   Gate 3: inverted, then rotated -50 deg
    #   Gate 4: automatic orientation
    #   Gate 5: automatic orientation
    # --------------------------------------------------
    GATE_1_ANGLE_DEG = 0.0
    GATE_2_ANGLE_DEG = 40.0
    GATE_3_ANGLE_DEG = -50.0
    GATE_4_ANGLE_DEG = -20.0
    GATE_5_ANGLE_DEG = 0.0

    GATE_1_INVERT = False
    GATE_2_INVERT = False
    GATE_3_INVERT = True
    GATE_4_INVERT = False
    GATE_5_INVERT = False

    GATE_ANGLE_OFFSETS_DEG = [
        GATE_1_ANGLE_DEG,
        GATE_2_ANGLE_DEG,
        GATE_3_ANGLE_DEG,
        GATE_4_ANGLE_DEG,
        GATE_5_ANGLE_DEG,
    ]

    GATE_INVERT_FLAGS = [
        GATE_1_INVERT,
        GATE_2_INVERT,
        GATE_3_INVERT,
        GATE_4_INVERT,
        GATE_5_INVERT,
    ]

    # Apply all custom gate orientations.
    for gate_idx, (angle_deg, invert_gate) in enumerate(
        zip(GATE_ANGLE_OFFSETS_DEG, GATE_INVERT_FLAGS)
    ):
        GATE_NORMALS, GATE_AXIS_U = rotate_gate_orientation(
            gate_normals=GATE_NORMALS,
            gate_axis_u=GATE_AXIS_U,
            gate_axis_v=GATE_AXIS_V,
            gate_idx=gate_idx,
            angle_deg=angle_deg,
            invert=invert_gate,
        )

    print("Gate angle offsets [deg]:")
    for gate_idx, (angle_deg, invert_gate) in enumerate(
        zip(GATE_ANGLE_OFFSETS_DEG, GATE_INVERT_FLAGS)
    ):
        print(
            f"  Gate {gate_idx + 1}: "
            f"angle_offset={angle_deg:.1f} deg, "
            f"invert={invert_gate}"
        )

    print("Gate normals:")
    print(np.asarray(GATE_NORMALS))

    # --------------------------------------------------
    # Two transition waypoints per gate
    # --------------------------------------------------
    (
        WAYPOINT_CENTERS,
        PRE_GATE_WAYPOINTS,
        POST_GATE_WAYPOINTS,
    ) = build_gate_transition_waypoints(
        gate_centers=GATE_CENTERS,
        gate_normals=GATE_NORMALS,
        transition_offset=GATE_TRANSITION_OFFSET,
    )

    NUM_PHASES = WAYPOINT_CENTERS.shape[0]
    assert NUM_PHASES == 2 * NUM_GATES

    # Each gate contributes two waypoints (pre and post), and both use the
    # same local gate frame. The local waypoint coordinate order is [u, v, n].
    WAYPOINT_AXIS_U = jnp.repeat(
        GATE_AXIS_U,
        2,
        axis=0,
    )
    WAYPOINT_AXIS_V = jnp.repeat(
        GATE_AXIS_V,
        2,
        axis=0,
    )
    WAYPOINT_NORMALS = jnp.repeat(
        GATE_NORMALS,
        2,
        axis=0,
    )

    # Alternate long travel segments and short gate-crossing segments:
    #
    #   start -> pre_1       : LONG_SEGMENT_STEPS = 20
    #   pre_1 -> post_1      : GATE_CROSSING_STEPS = 3
    #   post_1 -> pre_2      : LONG_SEGMENT_STEPS = 20
    #   pre_2 -> post_2      : GATE_CROSSING_STEPS = 3
    #   ...
    #
    # For five gates:
    #   segment lengths = [20, 3, 20, 3, 20, 3, 20, 3, 20, 3]
    #   waypoint steps  = [20, 23, 43, 46, 66, 69, 89, 92, 112, 115]
    SEGMENT_LENGTHS_INT = jnp.tile(
        jnp.array(
            [LONG_SEGMENT_STEPS, GATE_CROSSING_STEPS],
            dtype=jnp.int32,
        ),
        NUM_GATES,
    )

    WAYPOINT_STEPS = jnp.cumsum(SEGMENT_LENGTHS_INT)

    if int(WAYPOINT_STEPS[-1]) != N:
        raise ValueError(
            f"N={N}, but segment layout requires "
            f"N={int(WAYPOINT_STEPS[-1])}."
        )

    WAYPOINT_HALF_WIDTHS = jnp.tile(
        TRANSITION_WAYPOINT_HALF_WIDTHS[None, :],
        (NUM_PHASES, 1),
    )

    SEGMENT_LENGTHS = SEGMENT_LENGTHS_INT.astype(DTYPE)

    # Preserve approximately the same total initial duration as the old
    # five-phase initialization: 10 x 0.5 s = 5 s.
    INITIAL_DURATIONS = 0.5 * jnp.ones(
        NUM_PHASES,
        dtype=DTYPE,
    )

    x0 = jnp.concatenate([
        physical_x0,
        INITIAL_DURATIONS,
    ])

    print("Transition waypoints:")
    for gate_idx in range(NUM_GATES):
        pre = np.asarray(PRE_GATE_WAYPOINTS[gate_idx])
        post = np.asarray(POST_GATE_WAYPOINTS[gate_idx])
        print(
            f"  Gate {gate_idx + 1}: "
            f"pre={pre}, post={post}"
        )

    print("Waypoint steps:", np.asarray(WAYPOINT_STEPS))
    print(
        "Waypoint local full size [u, v, n]:",
        2.0 * np.asarray(TRANSITION_WAYPOINT_HALF_WIDTHS),
    )
    print(
        "Pre/post center separation:",
        2.0 * GATE_TRANSITION_OFFSET,
        "m",
    )

    # --------------------------------------------------
    # Piecewise reference:
    #
    # x0 -> pre1 -> post1 -> pre2 -> post2 -> ...
    # --------------------------------------------------
    reference = build_piecewise_reference(
        x0=x0,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_steps=WAYPOINT_STEPS,
        initial_durations=INITIAL_DURATIONS,
        N=N,
    )

    controller = build_controller(
        N,
        reference,
        WAYPOINT_STEPS,
        WAYPOINT_CENTERS,
        WAYPOINT_HALF_WIDTHS,
        WAYPOINT_AXIS_U,
        WAYPOINT_AXIS_V,
        WAYPOINT_NORMALS,
        SEGMENT_LENGTHS,
        GATE_CENTERS,
        GATE_NORMALS,
        GATE_AXIS_U,
        GATE_AXIS_V,
        GATE_OPENING_HALF_WIDTHS,
        GATE_BAR_THICKNESS=GATE_BAR_THICKNESS,
        GATE_DEPTH=GATE_DEPTH,
        DRONE_RADIUS=DRONE_RADIUS,
        disturbance_field=DISTURBANCE_FIELD,
    )

    parameter = (WAYPOINT_STEPS, SEGMENT_LENGTHS)

    # -----------------------------
    # Robust plan
    # -----------------------------
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0,
        reference=reference,
        parameter=parameter,
    )

    # -----------------------------
    # Reset / timing extraction
    # -----------------------------
    T_hover = MASS * GRAVITY
    controller.reset(
        reference,
        jnp.zeros((N, nu), dtype=DTYPE).at[:, 0].set(T_hover),
    )

    phase_times = X_pred[0, 12:]
    min_time = jnp.sum(phase_times)

    print("Phase Times:", phase_times)
    print("Computed Total Min Time:", min_time)

    # For visualization, the post-gate node is the second waypoint of each
    # [pre, post] pair.
    POST_GATE_STEPS = WAYPOINT_STEPS[1::2]

    # -----------------------------
    # Save optimized trajectory
    # -----------------------------
    trajectory_path = "multiphase_trajectory.npz"

    np.savez(
        trajectory_path,
        X=np.asarray(X_pred),
        U=np.asarray(U_pred),
        phase_times=np.asarray(phase_times),
        min_time=np.asarray(min_time),

        # Optimization transition waypoints
        waypoint_steps=np.asarray(WAYPOINT_STEPS),
        waypoint_centers=np.asarray(WAYPOINT_CENTERS),
        waypoint_half_widths=np.asarray(WAYPOINT_HALF_WIDTHS),
        waypoint_axis_u=np.asarray(WAYPOINT_AXIS_U),
        waypoint_axis_v=np.asarray(WAYPOINT_AXIS_V),
        waypoint_normals=np.asarray(WAYPOINT_NORMALS),
        segment_lengths=np.asarray(SEGMENT_LENGTHS),

        # Explicit pre/post pair data
        pre_gate_waypoints=np.asarray(PRE_GATE_WAYPOINTS),
        post_gate_waypoints=np.asarray(POST_GATE_WAYPOINTS),
        post_gate_steps=np.asarray(POST_GATE_STEPS),
        gate_transition_offset=np.asarray(GATE_TRANSITION_OFFSET),

        # Physical gate geometry
        gate_centers=np.asarray(GATE_CENTERS),
        gate_normals=np.asarray(GATE_NORMALS),
        gate_axis_u=np.asarray(GATE_AXIS_U),
        gate_axis_v=np.asarray(GATE_AXIS_V),
        gate_opening_half_widths=np.asarray(GATE_OPENING_HALF_WIDTHS),
        gate_bar_thickness=np.asarray(GATE_BAR_THICKNESS),
        gate_depth=np.asarray(GATE_DEPTH),
        drone_radius=np.asarray(DRONE_RADIUS),

        reference=np.asarray(reference),
    )

    print(f"Saved optimized trajectory to: {trajectory_path}")

    # --------------------------------------------------
    # 3D visualization
    #
    # Draw the five PHYSICAL gates, not the ten transition waypoints.
    # POST_GATE_STEPS is only used to mark a representative trajectory node
    # after each crossing.
    # --------------------------------------------------
    GATE_DISPLAY_HALF_WIDTHS = 0.20 * jnp.ones(
        (NUM_GATES, 3),
        dtype=DTYPE,
    )

    visualize_trajectory_and_gates(
        X=X_pred,
        waypoint_centers=GATE_CENTERS,
        waypoint_half_widths=GATE_DISPLAY_HALF_WIDTHS,
        waypoint_steps=POST_GATE_STEPS,
        reference=reference,
        gate_normals=GATE_NORMALS,
        gate_axis_u=GATE_AXIS_U,
        gate_axis_v=GATE_AXIS_V,
        gate_opening_half_widths=GATE_OPENING_HALF_WIDTHS,
        gate_bar_thickness=GATE_BAR_THICKNESS,
        gate_depth=GATE_DEPTH,
        drone_radius=DRONE_RADIUS,
        show_collision_boxes=True,
        show_waypoint_boxes=False,
        save_path="multiphase_trajectory.png",
    )

    # --------------------------------------------------
    # Top-down plot
    #
    # Here show all ten optimization transition waypoints.
    # --------------------------------------------------
    plot_top_down_trajectory(
        X=X_pred,
        waypoint_centers=WAYPOINT_CENTERS,
        waypoint_half_widths=WAYPOINT_HALF_WIDTHS,
        waypoint_axis_u=WAYPOINT_AXIS_U,
        waypoint_normals=WAYPOINT_NORMALS,

        # Physical gate geometry
        gate_centers=GATE_CENTERS,
        gate_normals=GATE_NORMALS,
        gate_axis_u=GATE_AXIS_U,
        gate_opening_half_widths=GATE_OPENING_HALF_WIDTHS,
        gate_bar_thickness=GATE_BAR_THICKNESS,
        gate_depth=GATE_DEPTH,
        drone_radius=DRONE_RADIUS,
        show_gate_collision_inflation=False,

        reference=reference,
        Phi_x=Phi_x,
        E_mag=E_MAG,
        disturbance_field=DISTURBANCE_FIELD,
        tube_stride=2,
        min_time=min_time,
        save_path="trajectory_topdown.png",
    )

    # -----------------------------
    # Rollout simulations
    # -----------------------------


if __name__ == "__main__":
    main()