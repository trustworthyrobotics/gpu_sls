from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
# os.environ["JAX_PLATFORMS"] = "cpu"
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from jax import config

import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import combine_constraints, make_control_box_constraints, make_state_box_constraints
from gpu_sls.utils.sls_visual import get_trajectory_tubes
from visualize_experiment import (
    plot_rocket_controls,
    plot_rocket_3d,
    plot_tube_graph_rocket,
    plot_rocket_top_down_tubes_goal_obstacles,
    plot_rocket_side_view_tubes_goal_obstacles,
)

# -----------------------------
# Goal stopping config
# -----------------------------
GOAL_TOL = 0.25  # meters in xyz


def reached_goal_xyz(x: jnp.ndarray, x_goal: jnp.ndarray, tol: float = GOAL_TOL) -> jnp.bool_:
    dpos = x[:3] - x_goal[:3]
    return (dpos @ dpos) <= (tol * tol)

# -----------------------------
# 6-DOF rocket parameters
# x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r, mass, T_final]
# u = [thrust, gimbal_x, gimbal_y, roll_torque]
#
# Benchmark geometry: the main engine points primarily along +body-x so the
# rocket naturally makes forward progress while remaining at low altitude.
# A configurable baseline gravity-support term is included specifically to make
# the "low cruise -> late pull-up" behavior easy to expose in this experiment.
# Set CRUISE_GRAVITY_SUPPORT = 0.0 for a purely ballistic rocket.
# -----------------------------
GRAVITY = 9.81
G0 = 9.80665
WET_MASS = 5.0
DRY_MASS = 3.5
ISP = 220.0
ENGINE_LEVER_ARM = 0.80

# Experiment-shaping parameter. 1.0 exactly cancels gravity in the translational
# dynamics when there is no vertical thrust component, so a level vehicle can
# cruise at constant altitude. This makes the state-dependent disturbance, not
# gravity, the dominant reason to choose when to climb.
CRUISE_GRAVITY_SUPPORT = 1.0

# Nominal forward-thrust level used only as a mild regularization center / warm
# start. With the gravity-support benchmark dynamics, ~0.5 mg moves about 10 m
# in roughly 2 s from rest.
CRUISE_THRUST_FRACTION = 0.50

# Approximate rigid-body inertia of a small slender rocket.
JX = 0.35
JY = 0.35
JZ = 0.08
J = jnp.diag(jnp.array([JX, JY, JZ], dtype=jnp.float64))
J_INV = jnp.diag(jnp.array([1.0 / JX, 1.0 / JY, 1.0 / JZ], dtype=jnp.float64))

NUM_RANDOM = 5
NUM_ADV = 26

# Unscaled disturbance magnitude used by both the solver and comparison plot:
#   E_mag(z) = scale * (tanh(slope * z + bias) + 1).
# The actual discrete-time disturbance matrix additionally includes dt.
DISTURBANCE_SCALE = 1.0
DISTURBANCE_SLOPE = 6.0
DISTURBANCE_BIAS = -2.0


def wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap an angle in radians to [-pi, pi)."""
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=angle.dtype)
    pi = jnp.asarray(jnp.pi, dtype=angle.dtype)
    return jnp.mod(angle + pi, two_pi) - pi


def wrap_euler(euler: jnp.ndarray) -> jnp.ndarray:
    """Wrap [roll, pitch, yaw] elementwise to [-pi, pi)."""
    return wrap_angle(euler)


def rotation_matrix(phi: jnp.ndarray, theta: jnp.ndarray, psi: jnp.ndarray) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)

    # R = Rz(psi) Ry(theta) Rx(phi)
    return jnp.array([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                          cth * cphi],
    ], dtype=jnp.float64)

def euler_angle_rates_matrix(phi: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    sphi, cphi = jnp.sin(phi), jnp.cos(phi)
    tth = jnp.tan(theta)
    cth = jnp.cos(theta)

    return jnp.array([
        [1.0, sphi * tth, cphi * tth],
        [0.0, cphi,       -sphi],
        [0.0, sphi / cth, cphi / cth],
    ], dtype=jnp.float64)

def rocket_6dof_step(x: jnp.ndarray, u: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Forward-Euler 6-DOF benchmark rocket dynamics.

    State:
        x = [p_I(3), euler(3), v_I(3), omega_B(3), mass, T_final]

    Control:
        u = [thrust, gimbal_x, gimbal_y, roll_torque]

    Benchmark design:
      * zero gimbal points thrust along +body-x (forward), not +body-z;
      * gimbal_y adds an upward body-z component;
      * gimbal_x adds a lateral body-y component;
      * a configurable support acceleration can cancel gravity during level
        cruise, allowing the optimizer to remain below z=0 without spending
        control authority just to avoid falling;
      * the pitch-torque sign is chosen so positive gimbal_y also commands a
        visible nose-up / pull-up response under this Euler-angle convention.

    This is intentionally shaped as a benchmark for demonstrating the benefit
    of state-dependent robust tightening rather than as a high-fidelity launch
    vehicle model.
    """
    _, _, _, phi, theta, psi, vx, vy, vz, p, q, r, mass = x[:13]
    thrust, gimbal_x, gimbal_y, roll_torque = u

    v = jnp.array([vx, vy, vz], dtype=x.dtype)
    omega = jnp.array([p, q, r], dtype=x.dtype)

    R = rotation_matrix(phi, theta, psi)
    E = euler_angle_rates_matrix(phi, theta)
    e3 = jnp.array([0.0, 0.0, 1.0], dtype=x.dtype)

    # Main-engine thrust direction in body coordinates.
    # Zero gimbal -> +body-x, which produces forward progress while level.
    thrust_dir_B = jnp.array([
        jnp.cos(gimbal_x) * jnp.cos(gimbal_y),
        jnp.sin(gimbal_x),
        jnp.sin(gimbal_y) * jnp.cos(gimbal_x),
    ], dtype=x.dtype)
    force_B = thrust * thrust_dir_B

    # Experiment-oriented attitude coupling. Under this Euler convention,
    # negative pitch theta points +body-x upward. Therefore positive gimbal_y
    # is mapped to a negative body-y torque so an upward thrust-vector command
    # also produces the desired visible nose-up pull-up behavior.
    engine_torque_B = jnp.array([
        0.0,
        -ENGINE_LEVER_ARM * force_B[2],
        ENGINE_LEVER_ARM * force_B[1],
    ], dtype=x.dtype)
    control_torque_B = engine_torque_B + jnp.array(
        [roll_torque, 0.0, 0.0], dtype=x.dtype
    )

    # Translational dynamics in the inertial frame.
    # CRUISE_GRAVITY_SUPPORT=1 cancels gravity for this benchmark so zero
    # vertical thrust approximately preserves altitude. Set it to 0 for the
    # ordinary ballistic term -g e3.
    pos_dot = v
    support_accel = CRUISE_GRAVITY_SUPPORT * GRAVITY * e3
    v_dot = (R @ force_B) / mass - GRAVITY * e3 + support_accel

    # Match the CasADi benchmark exactly: natural early-climb acceleration.
    # This provides strong upward acceleration at low forward speed and fades
    # as vx increases, with linear damping in vertical velocity.
    CLIMB_ACCEL = 6.0
    CLIMB_SPEED_DECAY = 0.12
    VERTICAL_DAMPING = 1.0

    climb_accel = CLIMB_ACCEL * jnp.exp(-CLIMB_SPEED_DECAY * vx**2)
    v_dot = v_dot + jnp.array([
        0.0,
        0.0,
        climb_accel - VERTICAL_DAMPING * vz,
    ], dtype=x.dtype)

    # Rotational rigid-body dynamics in the body frame.
    euler_dot = E @ omega
    omega_dot = J_INV @ (
        control_torque_B - jnp.cross(omega, J @ omega)
    )

    # Ideal rocket propellant flow from Isp.
    mass_dot = -thrust / (ISP * G0)

    physical_dot = jnp.concatenate(
        [pos_dot, euler_dot, v_dot, omega_dot, jnp.reshape(mass_dot, (1,))],
        axis=0,
    )
    physical_next = x[:13] + dt * physical_dot

    # Keep the Euler-angle states in a canonical interval.
    # State layout: [px, py, pz, phi, theta, psi, ...]
    # physical_next = physical_next.at[3:6].set(
    #     physical_next[3:6]
    # )
    physical_next = physical_next.at[3:6].set(
        wrap_euler(physical_next[3:6])
    )

    # T_final is an optimization state and remains constant along the horizon.
    if x.shape[0] == 14:
        return jnp.concatenate([physical_next, x[13:14]])
    return physical_next

def rocket_step_with_disturbance(
    key: jax.Array,
    x: jnp.ndarray,      # (13,)
    u: jnp.ndarray,      # (4,)
    E: jnp.ndarray,      # (13,13)
    dt: float,
    i: int
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """
    x_{k+1} = f(x_k, u_k) + E w, with ||w||_2 <= 1
    """
    x_nom = rocket_6dof_step(x, u, dt)

    # Random disturbance in unit ball
    key, key_dir, key_rad = jax.random.split(key, 3)

    z = jax.random.normal(key_dir, (12,), dtype=x.dtype)
    z = z / (jnp.linalg.norm(z) + jnp.asarray(1e-12, dtype=x.dtype))

    n = jnp.asarray(12, dtype=x.dtype)
    a = jnp.asarray(0.0, dtype=x.dtype)
    b = jnp.asarray(1.0, dtype=x.dtype)

    uu = jax.random.uniform(key_rad, (), dtype=x.dtype)
    r = (a**n + (b**n - a**n) * uu) ** (1.0 / n)
    w = jnp.concatenate([r * z, jnp.zeros((2,), dtype=x.dtype)])

    # Deterministic adversarial-ish directions
    start = i - NUM_RANDOM + 5
    if 5 <= start <= 16:
        idx = start - 5
        w = jnp.zeros((14,), dtype=x.dtype).at[idx].set(1.0)
    if 17 <= start <= 28:
        idx = start - 17
        w = jnp.zeros((14,), dtype=x.dtype).at[idx].set(-1.0)
    if start == 29:
        w = jnp.concatenate([jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype)), jnp.zeros((2,), dtype=x.dtype)])
    if start == 30:
        w = jnp.concatenate([-jnp.ones((12,), dtype=x.dtype) / jnp.sqrt(jnp.asarray(12.0, dtype=x.dtype)), jnp.zeros((2,), dtype=x.dtype)])

    # Account for linearization error
    w = w.at[-1].set(0.0)
    x_next = x_nom + E @ w * dt
    return key, x_next, w

def dynamics(x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray, *, parameter: Any) -> jnp.ndarray:
    dtau = parameter
    return rocket_6dof_step(x, u, dtau * x[-1])

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
        wthrust, wgimbal_x, wgimbal_y, wroll_torque, wtime
    ) = W

    xref = reference[t]

    dpos = x[:3] - xref[:3]
    # dang = x[3:6] - xref[3:6]
    dang = wrap_euler(x[3:6] - xref[3:6])
    dvel = x[6:9] - xref[6:9]
    drates = x[9:12] - xref[9:12]

    # Mild thrust regularization around a forward-cruise value. The control
    # weights are intentionally small so the minimum-time objective dominates.
    T_cruise = CRUISE_THRUST_FRACTION * x[12] * GRAVITY
    du = jnp.array([
        u[0] - T_cruise,
        u[1],
        u[2],
        u[3],
    ], dtype=x.dtype)

    angle_cost = (
        wphi * (1.0 - jnp.cos(dang[0]))
        + wtheta * (1.0 - jnp.cos(dang[1]))
        + wpsi * (1.0 - jnp.cos(dang[2]))
    )

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
        + wthrust * du[0] ** 2
        + wgimbal_x * du[1] ** 2
        + wgimbal_y * du[2] ** 2
        + wroll_torque * du[3] ** 2
        + wtime * x[-1]
    )

def build_piecewise_reference(x0: jnp.ndarray, x_goal: jnp.ndarray, N: int, duration: float) -> jnp.ndarray:
    """
    Build a straight-line reference trajectory from x0 to x_goal.

    Only position (px,py,pz) and yaw are interpolated.
    All other states are set to zero reference.
    """

    t = jnp.linspace(0.0, 1.0, N + 1)

    # Linear interpolation for position
    pos = (1.0 - t[:, None]) * x0[:3] + t[:, None] * x_goal[:3]

    # Shortest-path yaw interpolation on the circle.
    dpsi =x_goal[5] - x0[5]
    psi = x0[5] + t * dpsi

    X_ref = jnp.zeros((N + 1, 14), dtype=jnp.float64)

    X_ref = X_ref.at[:, :3].set(pos)
    X_ref = X_ref.at[:, 5].set(psi)

    # Optionally compute velocity reference from the line
    vel = (x_goal[:3] - x0[:3]) / duration
    X_ref = X_ref.at[:, 6:9].set(vel)

    # Mass is not directly tracked by the running cost, but keeping a reasonable
    # reference makes the nominal trajectory internally consistent.
    X_ref = X_ref.at[:, 12].set(
        jnp.linspace(x0[12], x_goal[12], N + 1)
    )
    X_ref = X_ref.at[:, -1].set(duration)

    return X_ref

def make_terminal_set_constraint(center: jnp.ndarray, half_width: jnp.ndarray, N: int):
    """Enforce |x[:3] - center| <= half_width at the terminal step."""
    center = jnp.asarray(center)
    half_width = jnp.asarray(half_width)

    def constraints(x, u, t):
        terminal = jnp.concatenate([
            x[:3] - center - half_width,
            center - x[:3] - half_width,
        ])
        return jnp.where(t == N, terminal, -jnp.ones_like(terminal))

    return constraints

def make_min_time_disturbance(
    n: int,
    N: int,
    disturbance_index: int = 6,
):
    """Altitude-dependent forward-velocity disturbance.

    The disturbance is nearly zero at low altitude and approaches
    2 * DISTURBANCE_SCALE at high altitude. Its smooth, steep transition gives
    the robust optimizer a strong state gradient near
    z = -DISTURBANCE_BIAS / DISTURBANCE_SLOPE:

        z << transition  -> E_mag ~= 0.0
        z  = transition  -> E_mag  = DISTURBANCE_SCALE
        z >> transition  -> E_mag ~= 2 * DISTURBANCE_SCALE

    With disturbance_index=6 this uncertainty acts on v_x, i.e. uncertainty in
    forward progress. The robust solution therefore benefits from postponing
    entry into the high-disturbance region until the terminal altitude forces
    it to climb.
    """

    def disturbance_at_state(x_k: jnp.ndarray) -> jnp.ndarray:
        z = x_k[2]
        final_time = x_k[-1]
        dt = final_time / N

        E_mag = DISTURBANCE_SCALE * (
            jnp.tanh(DISTURBANCE_SLOPE * z + DISTURBANCE_BIAS) + 1.0
        )

        E_k = jnp.zeros((n, n), dtype=x_k.dtype)
        E_k = E_k.at[
            disturbance_index,
            disturbance_index,
        ].set(dt * E_mag)
        E_k = E_k.at[
            disturbance_index + 2,
            disturbance_index + 2,
        ].set(dt * E_mag / 4)

        return E_k

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(disturbance_at_state)(X)

    disturbance.at_state = disturbance_at_state
    return disturbance

def make_sphere_obstacle_constraint(
    center: jnp.ndarray,
    radius: float,
    clearance: float = 0.0,
):
    """
    Create a smooth 3D spherical obstacle-avoidance constraint.

    The returned constraint follows the convention:
        g(x, u, t) <= 0

    and enforces:
        ||x[:3] - center||_2 >= radius + clearance

    Args:
        center:
            Sphere center [cx, cy, cz], shape (3,).
        radius:
            Physical obstacle radius.
        clearance:
            Additional safety distance around the obstacle.

    Returns:
        constraints(x, u, t), returning shape (1,).
    """
    center = jnp.asarray(center)
    safe_radius = jnp.asarray(radius + clearance)

    def constraints(
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        del u, t

        displacement = x[:3] - center

        # <= 0 outside/on the sphere; > 0 inside the sphere.
        sphere_constraint = (
            safe_radius**2
            - jnp.dot(displacement, displacement)
        )

        return jnp.reshape(sphere_constraint, (1,))

    return constraints


def main(*, gradient_window: int = 0, output_dir: Path | str = Path(".")):
    if gradient_window < 0:
        raise ValueError("gradient_window must be nonnegative")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Dimensions
    # -----------------------------
    n = 14
    nu = 4

    # -----------------------------
    # Horizon and dt
    # -----------------------------
    N = 50
    parameter = 1.0 / N
    initial_duration = 2.0

    # -----------------------------
    # Cost weights
    # -----------------------------
    # W = jnp.array([
    #     0.0, 0.0, 0.0,     # position
    #     0.1, 0.1, 0.1,        # roll, pitch, yaw
    #     0.5, 0.5, 0.5,        # velocities
    #     0.05, 0.05, 0.05,     # body rates
    #     0.01, 0.01, 0.01, 0.01,  # control
    #     5.0                         # total time
    # ], dtype=jnp.float64)
    W = jnp.array([
        1e-4, 1e-4, 1e-4,          # position: do not track the straight-line path
        1e-3, 1e-3, 1e-3,       # roll, pitch, yaw
        1e-3, 1e-3, 1e-3,       # velocities
        1e-3, 1e-3, 1e-3,       # body rates
        1e-4, 1e-4, 1e-4, 1e-4, # control regularization
        1.0                       # total time
    ], dtype=jnp.float64)
    

    cfg = MPCConfig(
        n=n,
        nu=nu,
        N=N,
        W=W,
        u_ref=jnp.array([CRUISE_THRUST_FRACTION * WET_MASS * GRAVITY, 0.0, 0.0, 0.0], dtype=jnp.float64),
    )

    # -----------------------------
    # Control limits
    # -----------------------------
    T_hover = WET_MASS * GRAVITY
    T_cruise = CRUISE_THRUST_FRACTION * WET_MASS * GRAVITY
    T_max = 4.0 * WET_MASS * GRAVITY
    gimbal_max = jnp.deg2rad(60.0)
    roll_torque_max = 2.0

    u_min = jnp.array(
        [0.0, -gimbal_max, -gimbal_max, -roll_torque_max],
        dtype=jnp.float64,
    )
    u_max = jnp.array(
        [T_max, gimbal_max, gimbal_max, roll_torque_max],
        dtype=jnp.float64,
    )
    constraints_u = make_control_box_constraints(u_min, u_max)

    # -----------------------------
    # State limits
    # -----------------------------
    x_max = jnp.array([
        30.0, 30.0, 30.0,       # px, py, pz
        jnp.pi / 2.0,            # phi
        jnp.pi / 2.0,            # theta
        10.0 * jnp.pi,           # psi
        8.0, 8.0, 8.0,           # vx, vy, vz
        8.0, 8.0, 8.0,           # p, q, r
        WET_MASS,                 # mass
        20.0,                     # total time
    ], dtype=jnp.float64)
    x_min = -x_max
    # Give the optimizer room to remain in the low-disturbance region below z=0.
    x_min = x_min.at[2].set(-1.1)
    x_min = x_min.at[12].set(DRY_MASS)
    # A minimum-time problem must not admit zero or negative integration steps.
    x_min = x_min.at[-1].set(0.05)
    # x_min = x_min.at[3].set(-jnp.inf)
    # x_max = x_max.at[3].set(jnp.inf)
    # x_min = x_min.at[4].set(-jnp.inf)
    # x_max = x_max.at[4].set(jnp.inf)
    # x_min = x_min.at[5].set(-jnp.inf)
    # x_max = x_max.at[5].set(jnp.inf)

    constraints_x = make_state_box_constraints(x_min, x_max)
    terminal_center = jnp.array([20.0, 0.0, 3.0], dtype=jnp.float64)
    # Tight x/z terminal box makes the late pull-up visually unambiguous.
    terminal_half_width = jnp.array([8.0, 8.0, 1.5], dtype=jnp.float64)
    terminal_constraint = make_terminal_set_constraint(
        terminal_center, terminal_half_width, N
    )
    constraints_all = combine_constraints(
            constraints_x, constraints_u, terminal_constraint
        )

    # obstacles = jnp.array([
    #     [0.0, 0.0, 0.35],
    # ], dtype=jnp.float64)

    obstacles = jnp.zeros((0, 3))

    disturbance = make_min_time_disturbance(n=n, N=N, disturbance_index=6)

    # -----------------------------
    # Initial / goal
    # -----------------------------
    x0 = jnp.array([
        0.0, 0.0, -0.75,              # px, py, pz
        0.0, -jnp.pi / 4.0, 0.0,     # phi, theta, psi
        0.0, 0.0, 0.0,               # vx, vy, vz
        0.0, 0.0, 0.0,               # p, q, r
        WET_MASS,
        initial_duration,
    ], dtype=jnp.float64)

    x_goal = jnp.array([
        15.0, 0.0, 3.0,           # px, py, pz
        0.0, 0.0, 0.0,           # phi, theta, psi
        0.0, 0.0, 0.0,           # vx, vy, vz
        0.0, 0.0, 0.0,           # p, q, r
        4.85,                     # nominal terminal mass reference
        initial_duration,         # total time
    ], dtype=jnp.float64)

    X_ref = build_piecewise_reference(x0, x_goal, N, initial_duration)
    reference = X_ref
    T_steps = N

    # -----------------------------
    # Solver configs
    # -----------------------------
    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-3,
        eps_abs_grad=1e-2,
        eps_rel_grad=1e-3,
        rho_max=1e3,
        max_iterations=400,
        rho_update_frequency=25,
        initial_rho=1e-2,
        regularized_rho_update=False,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=True,
        rti=False,
        gradient_window=gradient_window,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=1e-10,
        step_tol=1e-10,
        line_search=True,
        # lm_regularization=1.0e-2
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
        X_in=X_ref,
        U_in=jnp.zeros((cfg.N, cfg.nu), dtype=jnp.float64).at[:, 0].set(T_cruise),
    )

    # -----------------------------
    # Robust plan
    # -----------------------------
    N_ROLLOUTS = NUM_RANDOM + NUM_ADV
    u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
        x0=x0, reference=reference, parameter=parameter
    )
    import time
    # start = time.perf_counter()
    # u0, X_pred, U_pred, V_pred, backoffs, Phi_x, Phi_u = controller.run(
    #         x0=x0, reference=reference, parameter=parameter
    #     )
    # end = time.perf_counter()
    # print(end - start)
    min_time = X_pred[0, -1]
    dt = min_time / N
    min_time_s = float(np.asarray(min_time))
    print(f"Gradient window: {gradient_window}")
    print(f"Computed minimum time: {min_time_s:.9g} s")

    # -----------------------------
    # Rollout simulations
    # -----------------------------
    xs = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float32)
    disturbed = np.full((N_ROLLOUTS, T_steps, n), np.nan, dtype=np.float32)
    stop_steps = np.full((N_ROLLOUTS,), T_steps, dtype=np.int32)

    # for i in range(N_ROLLOUTS):
    #     disturbance_history = [jnp.zeros((n,), dtype=jnp.float64)]
    #     x = x0.at[-1].set(min_time)
    #     jax.debug.print("Rolling out iteration {}", i)

    #     for k in range(T_steps):
    #         disturbance_feedback = jnp.zeros((nu,), dtype=jnp.float64)
    #         for j in range(k + 1):
    #             disturbance_feedback = disturbance_feedback + Phi_u[k, j] @ disturbance_history[j]

    #         u = U_pred[k] + disturbance_feedback
    #         u = jnp.clip(u, u_min, u_max)

    #         key, x, w = rocket_step_with_disturbance(key, x, u, E_sim, dt, i)

    #         err = np.abs(np.asarray(X_pred[k + 1] - x))

    #         disturbed[i, k, :] = err
    #         disturbance_history.append(w)
    #         xs[i, k] = np.asarray(x)

    # -----------------------------
    # 3D rocket tube visualization
    # -----------------------------
    tube = get_trajectory_tubes(Phi_x)
    lower = X_pred[:, :3] - tube[:, :3]
    upper = X_pred[:, :3] + tube[:, :3]

    obstacle_centers = jnp.array([
    ], dtype=jnp.float64)

    obstacle_radii = jnp.array([
    ], dtype=jnp.float64)
    # plot_rocket_3d(
    #     xs=xs,
    #     plan=np.asarray(X_pred),
    #     lower=np.asarray(lower),
    #     upper=np.asarray(upper),
    #     centers=np.asarray(obstacle_centers),
    #     radii=np.asarray(obstacle_radii),
    #     goal_center=np.asarray(terminal_center),
    #     goal_half_width=np.asarray(terminal_half_width),
    #     tube_stride=1,
    #     filename="rocket_3d_rollouts_tube.png",
    #     tube_alpha=0.08,
    #     margin=0.2,
    #     rollout_alpha=0.5,
    #     title="Minimum-Time Rocket: 3D Spherical Obstacles",
    # )

    # plot_tube_graph_rocket(
    #     disturbed=disturbed[:, :, :6],   # position + Euler angles only
    #     tube=tube[:, :6],
    #     dt=dt,
    #     filename="rocket_3d_disturbance_vs_tube_size_pose.png",
    # )
    # plot_rocket_controls(
    #     controls=np.asarray(U_pred),
    #     dt=dt,
    #     u_min=np.asarray(u_min),
    #     u_max=np.asarray(u_max),
    #     filename="rocket_controls.png",
    # )
    # plot_rocket_top_down_tubes_goal_obstacles(
    #     plan=X_pred,
    #     lower=lower,
    #     upper=upper,
    #     centers=obstacle_centers,
    #     radii=obstacle_radii,
    #     goal_center=np.asarray(terminal_center),
    #     goal_half_width=np.asarray(terminal_half_width),
    #     tube_stride=1,
    #     filename="rocket_top_down_tubes.png",
    # )

    plot_rocket_side_view_tubes_goal_obstacles(
        plan=X_pred,
        lower=lower,
        upper=upper,
        centers=obstacle_centers,
        radii=obstacle_radii,
        goal_center=np.asarray(terminal_center),
        goal_half_width=np.asarray(terminal_half_width),
        tube_stride=2,
        ground_height=0.0,
        filename=str(output_dir / "rocket_side_view_tubes.png"),
        title=f"FastSLS rocket (gradient window = {gradient_window})",
    )

    plan_np = np.asarray(X_pred)
    disturbance_magnitude = DISTURBANCE_SCALE * (
        np.tanh(DISTURBANCE_SLOPE * plan_np[:, 2] + DISTURBANCE_BIAS) + 1.0
    )
    dynamics_prediction = jax.vmap(
        lambda state, control: rocket_6dof_step(
            state, control, state[-1] / N
        )
    )(X_pred[:-1], U_pred)
    max_dynamics_defect = float(
        np.asarray(jnp.max(jnp.abs(X_pred[1:] - dynamics_prediction)))
    )
    padded_controls = jnp.pad(U_pred, ((0, 1), (0, 0)))
    constraint_values = jax.vmap(constraints_all)(
        X_pred, padded_controls, jnp.arange(N + 1)
    )
    max_constraint_violation = float(
        np.asarray(jnp.max(jnp.maximum(constraint_values, 0.0)))
    )
    max_tightened_constraint_violation = float(
        np.asarray(jnp.max(jnp.maximum(constraint_values + backoffs, 0.0)))
    )
    print(f"Maximum dynamics defect: {max_dynamics_defect:.9g}")
    print(f"Maximum constraint violation: {max_constraint_violation:.9g}")
    print(
        "Maximum tightened-constraint violation: "
        f"{max_tightened_constraint_violation:.9g}"
    )
    np.savez(
        output_dir / "rocket_result.npz",
        plan=plan_np,
        controls=np.asarray(U_pred),
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        tube=np.asarray(tube),
        backoffs=np.asarray(backoffs),
        centers=np.asarray(obstacle_centers),
        radii=np.asarray(obstacle_radii),
        goal_center=np.asarray(terminal_center),
        goal_half_width=np.asarray(terminal_half_width),
        ground_height=np.asarray(0.0, dtype=np.float32),
        tube_stride=np.asarray(2, dtype=np.int32),
        enable_fastsls=np.asarray(True),
        gradient_window=np.asarray(gradient_window, dtype=np.int32),
        min_time=np.asarray(min_time_s),
        dt=np.asarray(float(np.asarray(dt))),
        disturbance_magnitude=disturbance_magnitude,
        disturbance_scale=np.asarray(DISTURBANCE_SCALE),
        disturbance_slope=np.asarray(DISTURBANCE_SLOPE),
        disturbance_bias=np.asarray(DISTURBANCE_BIAS),
        max_dynamics_defect=np.asarray(max_dynamics_defect),
        max_constraint_violation=np.asarray(max_constraint_violation),
        max_tightened_constraint_violation=np.asarray(
            max_tightened_constraint_violation
        ),
    )
    result_path = output_dir / "rocket_result.npz"
    print(f"Saved result: {result_path.resolve()}")

    return {
        "gradient_window": gradient_window,
        "min_time": min_time_s,
        "result_path": result_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the FastSLS rocket minimum-time experiment."
    )
    parser.add_argument(
        "--gradient-window",
        type=int,
        default=0,
        help="Number of stages used for disturbance-gradient reasoning (0 disables it).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for the NPZ result and plots.",
    )
    args = parser.parse_args()
    main(gradient_window=args.gradient_window, output_dir=args.output_dir)