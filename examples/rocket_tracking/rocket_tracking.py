from __future__ import annotations

import os
import time
from typing import Any

import jax
import jax.numpy as jnp
from jax import config

import numpy as np


# =====================================================================
# JAX CONFIG
# =====================================================================

config.update("jax_enable_x64", False)

config.update(
    "jax_compilation_cache_dir",
    "/tmp/jax_cache",
)

config.update(
    "jax_persistent_cache_min_compile_time_secs",
    0,
)

config.update(
    "jax_persistent_cache_min_entry_size_bytes",
    -1,
)

config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

DTYPE = (
    jnp.float64
    if jax.config.x64_enabled
    else jnp.float32
)


# =====================================================================
# GPU-SLS
# =====================================================================

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig

from gpu_sls.generic_mpc import (
    GenericMPC,
    MPCConfig,
)

from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
    make_state_box_constraints,
)


# =====================================================================
# DIMENSIONS
# =====================================================================

# Rocket physical state:
#
# 0  px
# 1  py
# 2  pz
#
# 3  phi
# 4  theta
# 5  psi
#
# 6  vx
# 7  vy
# 8  vz
#
# 9  p
# 10 q
# 11 r
#
# 12 mass
#
# Additional min-time states:
#
# 13 absolute time
# 14 remaining/interception time T
#
#
# x =
# [
#   px, py, pz,
#   phi, theta, psi,
#   vx, vy, vz,
#   p, q, r,
#   mass,
#   absolute_time,
#   T
# ]
#
#
# u =
# [
#   thrust,
#   tau_phi,
#   tau_theta,
#   tau_psi
# ]

NX = 15
NU = 4

IDX_MASS = 12
IDX_CLOCK = 13
IDX_TIME = 14


# =====================================================================
# MPC / MIN-TIME CONFIG
# =====================================================================

N = 60

# normalized discretization
DELTA_TAU = 1.0 / N

MIN_TIME = 0.20
MAX_TIME = 12.0

INITIAL_DURATION = 4.0

# Maximum number of receding-horizon RTI updates
MAX_RTI_STEPS = 300

# Interception criterion used for closed-loop simulation
CAPTURE_RADIUS = 0.40

# Optional velocity condition
USE_CAPTURE_VELOCITY = False
CAPTURE_REL_VEL = 2.0


# =====================================================================
# ROCKET PARAMETERS
# =====================================================================

GRAVITY = 9.81
G0 = 9.80665

INITIAL_MASS = 6.0
DRY_MASS = 4.0

# Rocket specific impulse
ISP = 220.0

# Inertia
JX = 0.12
JY = 0.12
JZ = 0.035

J = jnp.diag(
    jnp.array(
        [
            JX,
            JY,
            JZ,
        ],
        dtype=DTYPE,
    )
)

J_INV = jnp.diag(
    jnp.array(
        [
            1.0 / JX,
            1.0 / JY,
            1.0 / JZ,
        ],
        dtype=DTYPE,
    )
)


# =====================================================================
# AERODYNAMICS
# =====================================================================

AIR_DENSITY = 1.225
CD = 0.30
REFERENCE_AREA = 0.015


# =====================================================================
# ACTUATOR LIMITS
# =====================================================================

MAX_THRUST = (
    4.0
    * INITIAL_MASS
    * GRAVITY
)

MAX_TORQUE = 2.0


# =====================================================================
# OUTPUT
# =====================================================================

OUTPUT_DIR = "rocket_min_time_tracking"

PLOT_PREDICTION_EVERY = 5

SHOW_PLOTS = False


# =====================================================================
# ROTATION MATRICES
# =====================================================================

def rotation_matrix(
    phi: jnp.ndarray,
    theta: jnp.ndarray,
    psi: jnp.ndarray,
) -> jnp.ndarray:

    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)

    cth = jnp.cos(theta)
    sth = jnp.sin(theta)

    cpsi = jnp.cos(psi)
    spsi = jnp.sin(psi)

    return jnp.array(
        [
            [
                cpsi * cth,
                cpsi * sth * sphi
                - spsi * cphi,
                cpsi * sth * cphi
                + spsi * sphi,
            ],
            [
                spsi * cth,
                spsi * sth * sphi
                + cpsi * cphi,
                spsi * sth * cphi
                - cpsi * sphi,
            ],
            [
                -sth,
                cth * sphi,
                cth * cphi,
            ],
        ],
        dtype=phi.dtype,
    )


def euler_angle_rates_matrix(
    phi: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:

    sphi = jnp.sin(phi)
    cphi = jnp.cos(phi)

    ttheta = jnp.tan(theta)
    ctheta = jnp.cos(theta)

    return jnp.array(
        [
            [
                1.0,
                sphi * ttheta,
                cphi * ttheta,
            ],
            [
                0.0,
                cphi,
                -sphi,
            ],
            [
                0.0,
                sphi / ctheta,
                cphi / ctheta,
            ],
        ],
        dtype=phi.dtype,
    )


# =====================================================================
# MOVING TARGET
# =====================================================================

def target_state(
    absolute_time: jnp.ndarray,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Moving target model.

    Returns
    -------
    position : (3,)
    velocity : (3,)

    Target travels forward while weaving in y and z.
    """

    t = absolute_time

    # -------------------------------------------------------------
    # x direction
    # -------------------------------------------------------------

    x0 = 7.0
    vx = 2.0

    px = (
        x0
        + vx * t
    )

    # -------------------------------------------------------------
    # y weaving
    # -------------------------------------------------------------

    Ay = 3.0
    wy = 0.55

    py = (
        Ay
        * jnp.sin(
            wy * t
        )
    )

    vy = (
        Ay
        * wy
        * jnp.cos(
            wy * t
        )
    )

    # -------------------------------------------------------------
    # altitude variation
    # -------------------------------------------------------------

    z0 = 7.0
    Az = 1.5
    wz = 0.35

    pz = (
        z0
        + Az
        * jnp.sin(
            wz * t
        )
    )

    vz = (
        Az
        * wz
        * jnp.cos(
            wz * t
        )
    )

    position = jnp.array(
        [
            px,
            py,
            pz,
        ],
        dtype=t.dtype,
    )

    velocity = jnp.array(
        [
            vx,
            vy,
            vz,
        ],
        dtype=t.dtype,
    )

    return (
        position,
        velocity,
    )


# =====================================================================
# ROCKET CONTINUOUS DYNAMICS
# =====================================================================

def rocket_continuous_dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:
    """
    Physical rocket dynamics only.

    Input x contains the complete augmented state, but this function
    produces derivatives for the first 13 physical states:

        position
        attitude
        velocity
        body rates
        mass
    """

    (
        px,
        py,
        pz,
        phi,
        theta,
        psi,
        vx,
        vy,
        vz,
        p,
        q,
        r,
        mass,
    ) = x[:13]

    (
        thrust,
        tau_phi,
        tau_theta,
        tau_psi,
    ) = u

    # -------------------------------------------------------------
    # Vectors
    # -------------------------------------------------------------

    velocity = jnp.array(
        [
            vx,
            vy,
            vz,
        ],
        dtype=x.dtype,
    )

    omega = jnp.array(
        [
            p,
            q,
            r,
        ],
        dtype=x.dtype,
    )

    torque = jnp.array(
        [
            tau_phi,
            tau_theta,
            tau_psi,
        ],
        dtype=x.dtype,
    )

    e3 = jnp.array(
        [
            0.0,
            0.0,
            1.0,
        ],
        dtype=x.dtype,
    )

    # -------------------------------------------------------------
    # Rotation
    # -------------------------------------------------------------

    R = rotation_matrix(
        phi,
        theta,
        psi,
    )

    E = euler_angle_rates_matrix(
        phi,
        theta,
    )

    # -------------------------------------------------------------
    # Position
    # -------------------------------------------------------------

    position_dot = velocity

    # -------------------------------------------------------------
    # Drag
    # -------------------------------------------------------------

    speed = jnp.sqrt(
        jnp.dot(
            velocity,
            velocity,
        )
        + jnp.asarray(
            1e-8,
            dtype=x.dtype,
        )
    )

    drag_force = (
        0.5
        * AIR_DENSITY
        * CD
        * REFERENCE_AREA
        * speed
        * velocity
    )

    # -------------------------------------------------------------
    # Translational dynamics
    # -------------------------------------------------------------

    thrust_force = (
        thrust
        * (R @ e3)
    )

    velocity_dot = (
        thrust_force / mass
        - GRAVITY * e3
        - drag_force / mass
    )

    # -------------------------------------------------------------
    # Rotational dynamics
    # -------------------------------------------------------------

    euler_dot = (
        E @ omega
    )

    omega_dot = (
        J_INV
        @ (
            torque
            - jnp.cross(
                omega,
                J @ omega,
            )
        )
    )

    # -------------------------------------------------------------
    # Propellant depletion
    # -------------------------------------------------------------

    mass_dot = (
        -thrust
        / (ISP * G0)
    )

    return jnp.concatenate(
        [
            position_dot,
            euler_dot,
            velocity_dot,
            omega_dot,
            jnp.array(
                [
                    mass_dot,
                ],
                dtype=x.dtype,
            ),
        ]
    )


# =====================================================================
# MIN-TIME AUGMENTED DYNAMICS
# =====================================================================

def dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: Any,
) -> jnp.ndarray:
    """
    Min-time dynamics.

    parameter = DELTA_TAU = 1 / N

    physical timestep:

        dt = DELTA_TAU * T

    Absolute clock evolves as

        clock_{k+1}
            = clock_k + DELTA_TAU * T

    T itself is constant over the horizon.

    Therefore:

        clock_N = clock_0 + T

    and the terminal target is automatically evaluated at the
    optimized interception time.
    """

    del t

    delta_tau = parameter

    final_time = x[IDX_TIME]

    dt = (
        delta_tau
        * final_time
    )

    physical_dot = (
        rocket_continuous_dynamics(
            x,
            u,
        )
    )

    physical_next = (
        x[:13]
        + dt * physical_dot
    )

    absolute_time_next = (
        x[IDX_CLOCK]
        + dt
    )

    # Final-time decision variable is constant through horizon.
    final_time_next = (
        x[IDX_TIME]
    )

    return jnp.concatenate(
        [
            physical_next,
            jnp.array(
                [
                    absolute_time_next,
                    final_time_next,
                ],
                dtype=x.dtype,
            ),
        ]
    )


# =====================================================================
# MOVING TERMINAL SET
# =====================================================================

def make_moving_target_terminal_constraint(
    N: int,
    position_half_width: jnp.ndarray,
    velocity_half_width: jnp.ndarray | None = None,
):
    """
    Moving terminal set

        |p_N - p_target(t_N)| <= eps_p

    optionally

        |v_N - v_target(t_N)| <= eps_v

    Since t_N = t_now + T, the target position itself depends on
    the optimized final time.
    """

    position_half_width = jnp.asarray(
        position_half_width,
        dtype=DTYPE,
    )

    if velocity_half_width is not None:
        velocity_half_width = jnp.asarray(
            velocity_half_width,
            dtype=DTYPE,
        )

    def constraints(
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        del u

        target_position, target_velocity = (
            target_state(
                x[IDX_CLOCK]
            )
        )

        dpos = (
            x[:3]
            - target_position
        )

        position_constraints = (
            jnp.concatenate(
                [
                    dpos
                    - position_half_width,

                    -dpos
                    - position_half_width,
                ]
            )
        )

        if velocity_half_width is not None:

            dvel = (
                x[6:9]
                - target_velocity
            )

            velocity_constraints = (
                jnp.concatenate(
                    [
                        dvel
                        - velocity_half_width,

                        -dvel
                        - velocity_half_width,
                    ]
                )
            )

            terminal = (
                jnp.concatenate(
                    [
                        position_constraints,
                        velocity_constraints,
                    ]
                )
            )

        else:

            terminal = (
                position_constraints
            )

        return jnp.where(
            t == N,
            terminal,
            -jnp.ones_like(
                terminal
            ),
        )

    return constraints


# =====================================================================
# MIN-TIME COST
# =====================================================================

def make_min_time_cost(
    N: int,
):
    """
    W:

    W[0] position shaping
    W[1] velocity shaping
    W[2] attitude regularization
    W[3] body rate regularization
    W[4] thrust regularization
    W[5] torque regularization
    W[6] total-time weight

    The true objective is T.

    The other terms are deliberately small and only help shape the
    SQP trajectory.
    """

    def cost(
        W: jnp.ndarray,
        reference: jnp.ndarray,
        x: jnp.ndarray,
        u: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        del reference

        (
            w_pos,
            w_vel,
            w_attitude,
            w_rates,
            w_thrust,
            w_torque,
            w_time,
        ) = W

        target_position, target_velocity = (
            target_state(
                x[IDX_CLOCK]
            )
        )

        dpos = (
            x[:3]
            - target_position
        )

        dvel = (
            x[6:9]
            - target_velocity
        )

        attitude = (
            x[3:6]
        )

        body_rates = (
            x[9:12]
        )

        # ---------------------------------------------------------
        # Small trajectory-shaping cost
        # ---------------------------------------------------------

        tracking_cost = (
            w_pos
            * jnp.dot(
                dpos,
                dpos,
            )
            +
            w_vel
            * jnp.dot(
                dvel,
                dvel,
            )
        )

        state_regularization = (
            w_attitude
            * jnp.dot(
                attitude,
                attitude,
            )
            +
            w_rates
            * jnp.dot(
                body_rates,
                body_rates,
            )
        )

        # ---------------------------------------------------------
        # Control regularization
        # ---------------------------------------------------------

        normalized_thrust = (
            u[0]
            / MAX_THRUST
        )

        normalized_torque = (
            u[1:4]
            / MAX_TORQUE
        )

        control_regularization = (
            w_thrust
            * normalized_thrust**2
            +
            w_torque
            * jnp.dot(
                normalized_torque,
                normalized_torque,
            )
        )

        # Avoid counting padded terminal control.
        stage_cost = jnp.where(
            t < N,
            tracking_cost
            + state_regularization
            + control_regularization,
            0.0,
        )

        # ---------------------------------------------------------
        # TRUE MIN-TIME OBJECTIVE
        #
        # model_evaluator sums cost over all t, so add T only once.
        # ---------------------------------------------------------

        time_cost = jnp.where(
            t == N,
            w_time
            * x[IDX_TIME],
            0.0,
        )

        return (
            stage_cost
            + time_cost
        )

    return cost


# =====================================================================
# INITIAL NOMINAL TRAJECTORY
# =====================================================================

def build_initial_reference(
    x0: jnp.ndarray,
    N: int,
    duration: float,
) -> jnp.ndarray:
    """
    Build an initial guess from the rocket's current position toward
    the target's predicted position at

        current_time + duration.

    The target prediction is only used for initialization.

    During optimization the target is evaluated from x[IDX_CLOCK],
    so its terminal position changes correctly with T.
    """

    duration = jnp.asarray(
        duration,
        dtype=x0.dtype,
    )

    current_time = (
        x0[IDX_CLOCK]
    )

    terminal_time = (
        current_time
        + duration
    )

    target_terminal_position, target_terminal_velocity = (
        target_state(
            terminal_time
        )
    )

    alpha = jnp.linspace(
        0.0,
        1.0,
        N + 1,
        dtype=x0.dtype,
    )

    # -------------------------------------------------------------
    # Position guess
    # -------------------------------------------------------------

    positions = (
        (1.0 - alpha[:, None])
        * x0[:3]
        +
        alpha[:, None]
        * target_terminal_position
    )

    # -------------------------------------------------------------
    # Velocity guess
    # -------------------------------------------------------------

    average_velocity = (
        target_terminal_position
        - x0[:3]
    ) / duration

    velocities = (
        jnp.broadcast_to(
            average_velocity,
            (N + 1, 3),
        )
    )

    # Smooth velocity toward target velocity.
    velocities = (
        (1.0 - alpha[:, None])
        * velocities
        +
        alpha[:, None]
        * target_terminal_velocity
    )

    # -------------------------------------------------------------
    # Clock trajectory
    # -------------------------------------------------------------

    clock = (
        current_time
        + alpha * duration
    )

    # -------------------------------------------------------------
    # Approximate fuel burn
    # -------------------------------------------------------------

    nominal_thrust = (
        INITIAL_MASS
        * GRAVITY
    )

    nominal_mass_dot = (
        -nominal_thrust
        / (ISP * G0)
    )

    mass = (
        x0[IDX_MASS]
        + nominal_mass_dot
        * alpha
        * duration
    )

    mass = jnp.maximum(
        mass,
        DRY_MASS,
    )

    # -------------------------------------------------------------
    # Construct trajectory
    # -------------------------------------------------------------

    X = jnp.zeros(
        (
            N + 1,
            NX,
        ),
        dtype=x0.dtype,
    )

    X = X.at[
        :,
        :3,
    ].set(
        positions
    )

    # attitudes zero initially
    X = X.at[
        :,
        3:6,
    ].set(
        0.0
    )

    X = X.at[
        :,
        6:9,
    ].set(
        velocities
    )

    X = X.at[
        :,
        9:12,
    ].set(
        0.0
    )

    X = X.at[
        :,
        IDX_MASS,
    ].set(
        mass
    )

    X = X.at[
        :,
        IDX_CLOCK,
    ].set(
        clock
    )

    X = X.at[
        :,
        IDX_TIME,
    ].set(
        duration
    )

    return X


# =====================================================================
# DISTURBANCE
# =====================================================================

def make_min_time_disturbance(
    n: int,
    N: int,
    magnitude: float,
):
    """
    Very small disturbance for the initial non-robust experiment.

    No uncertainty is applied to

        mass
        absolute time
        final-time decision T
    """

    def disturbance_at_state(
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        final_time = (
            x[IDX_TIME]
        )

        dt = (
            final_time
            / N
        )

        E = (
            dt
            * magnitude
            * jnp.eye(
                n,
                dtype=x.dtype,
            )
        )

        # No disturbance on mass
        E = E.at[
            IDX_MASS,
            :,
        ].set(
            0.0
        )

        E = E.at[
            :,
            IDX_MASS,
        ].set(
            0.0
        )

        # No disturbance on clock
        E = E.at[
            IDX_CLOCK,
            :,
        ].set(
            0.0
        )

        E = E.at[
            :,
            IDX_CLOCK,
        ].set(
            0.0
        )

        # No disturbance on T
        E = E.at[
            IDX_TIME,
            :,
        ].set(
            0.0
        )

        E = E.at[
            :,
            IDX_TIME,
        ].set(
            0.0
        )

        return E

    def disturbance(
        X: jnp.ndarray,
    ) -> jnp.ndarray:

        return jax.vmap(
            disturbance_at_state
        )(X)

    disturbance.at_state = (
        disturbance_at_state
    )

    return disturbance


# =====================================================================
# CLOSED-LOOP PLANT
# =====================================================================

def simulate_rocket_step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Simulate one physical control interval.

    The T state is not physical. We simply retain its latest optimized
    value as the next SQP warm-start guess.
    """

    physical_dot = (
        rocket_continuous_dynamics(
            x,
            u,
        )
    )

    physical_next = (
        x[:13]
        + dt * physical_dot
    )

    # Prevent tiny numerical mass undershoot in plant simulation.
    physical_next = (
        physical_next.at[
            IDX_MASS
        ].set(
            jnp.maximum(
                physical_next[IDX_MASS],
                DRY_MASS,
            )
        )
    )

    clock_next = (
        x[IDX_CLOCK]
        + dt
    )

    return jnp.concatenate(
        [
            physical_next,
            jnp.array(
                [
                    clock_next,
                    x[IDX_TIME],
                ],
                dtype=x.dtype,
            ),
        ]
    )


# =====================================================================
# PLOTS
# =====================================================================

def plot_results(
    rocket_history: np.ndarray,
    target_history: np.ndarray,
    controls: np.ndarray,
    optimized_times: np.ndarray,
    solve_times: np.ndarray,
    predicted_rocket_plans: list[np.ndarray],
    predicted_target_plans: list[np.ndarray],
):
    import matplotlib.pyplot as plt

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    clock = (
        rocket_history[
            :,
            IDX_CLOCK,
        ]
    )

    # =================================================================
    # 3-D trajectories
    # =================================================================

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot(
        rocket_history[:, 0],
        rocket_history[:, 1],
        rocket_history[:, 2],
        linewidth=2.5,
        label="Rocket",
    )

    ax.plot(
        target_history[:, 0],
        target_history[:, 1],
        target_history[:, 2],
        linestyle="--",
        linewidth=2.5,
        label="Target",
    )

    for plan in predicted_rocket_plans:

        ax.plot(
            plan[:, 0],
            plan[:, 1],
            plan[:, 2],
            alpha=0.18,
            linewidth=1.0,
        )

    for plan in predicted_target_plans:

        ax.plot(
            plan[:, 0],
            plan[:, 1],
            plan[:, 2],
            linestyle=":",
            alpha=0.18,
            linewidth=1.0,
        )

    ax.scatter(
        rocket_history[0, 0],
        rocket_history[0, 1],
        rocket_history[0, 2],
        s=60,
        label="Rocket start",
    )

    ax.scatter(
        target_history[0, 0],
        target_history[0, 1],
        target_history[0, 2],
        s=60,
        label="Target start",
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    ax.set_zlabel(
        "z [m]"
    )

    ax.set_title(
        "Minimum-Time Rocket Interception"
    )

    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "rocket_min_time_3d.png",
        ),
        dpi=200,
    )

    # =================================================================
    # Distance
    # =================================================================

    distance = np.linalg.norm(
        rocket_history[:, :3]
        - target_history[:, :3],
        axis=1,
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.plot(
        clock,
        distance,
        linewidth=2,
    )

    ax2.axhline(
        CAPTURE_RADIUS,
        linestyle="--",
        label="Capture radius",
    )

    ax2.set_xlabel(
        "Absolute time [s]"
    )

    ax2.set_ylabel(
        "Distance [m]"
    )

    ax2.set_title(
        "Rocket-to-Target Distance"
    )

    ax2.grid(True)
    ax2.legend()

    fig2.tight_layout()

    fig2.savefig(
        os.path.join(
            OUTPUT_DIR,
            "target_distance.png",
        ),
        dpi=200,
    )

    # =================================================================
    # Optimized remaining time
    # =================================================================

    if len(optimized_times) > 0:

        fig3, ax3 = plt.subplots(
            figsize=(10, 5)
        )

        ax3.plot(
            clock[:-1],
            optimized_times,
            linewidth=2,
        )

        ax3.set_xlabel(
            "Absolute time [s]"
        )

        ax3.set_ylabel(
            "Optimal remaining time T [s]"
        )

        ax3.set_title(
            "RTI Minimum-Time Prediction"
        )

        ax3.grid(True)

        fig3.tight_layout()

        fig3.savefig(
            os.path.join(
                OUTPUT_DIR,
                "optimized_remaining_time.png",
            ),
            dpi=200,
        )

    # =================================================================
    # Solve times
    # =================================================================

    if len(solve_times) > 0:

        fig4, ax4 = plt.subplots(
            figsize=(10, 5)
        )

        ax4.plot(
            1000.0
            * solve_times
        )

        ax4.set_xlabel(
            "RTI iteration"
        )

        ax4.set_ylabel(
            "Solve time [ms]"
        )

        ax4.set_title(
            "SQP-RTI Runtime"
        )

        ax4.grid(True)

        fig4.tight_layout()

        fig4.savefig(
            os.path.join(
                OUTPUT_DIR,
                "solve_time.png",
            ),
            dpi=200,
        )

    # =================================================================
    # Thrust
    # =================================================================

    if len(controls) > 0:

        fig5, ax5 = plt.subplots(
            figsize=(10, 5)
        )

        ax5.plot(
            clock[:-1],
            controls[:, 0],
        )

        ax5.axhline(
            MAX_THRUST,
            linestyle="--",
        )

        ax5.set_xlabel(
            "Absolute time [s]"
        )

        ax5.set_ylabel(
            "Thrust [N]"
        )

        ax5.set_title(
            "Rocket Thrust"
        )

        ax5.grid(True)

        fig5.tight_layout()

        fig5.savefig(
            os.path.join(
                OUTPUT_DIR,
                "thrust.png",
            ),
            dpi=200,
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            "all"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # =================================================================
    # Initial state
    # =================================================================

    x0 = jnp.array(
        [
            # position
            0.0,
            0.0,
            0.5,

            # attitude
            0.0,
            0.0,
            0.0,

            # velocity
            0.0,
            0.0,
            0.0,

            # body rates
            0.0,
            0.0,
            0.0,

            # mass
            INITIAL_MASS,

            # absolute time
            0.0,

            # initial guess for interception time
            INITIAL_DURATION,
        ],
        dtype=DTYPE,
    )

    # =================================================================
    # Cost
    # =================================================================

    cost = make_min_time_cost(
        N=N,
    )

    W = jnp.array(
        [
            1e-3,      # position shaping
            1e-4,      # velocity shaping
            1e-5,      # attitude
            1e-5,      # body rates
            1e-6,      # thrust
            1e-6,      # torque

            5.0,       # MINIMUM TIME
        ],
        dtype=DTYPE,
    )

    # =================================================================
    # Control limits
    # =================================================================

    u_min = jnp.array(
        [
            0.0,

            -MAX_TORQUE,
            -MAX_TORQUE,
            -MAX_TORQUE,
        ],
        dtype=DTYPE,
    )

    u_max = jnp.array(
        [
            MAX_THRUST,

            MAX_TORQUE,
            MAX_TORQUE,
            MAX_TORQUE,
        ],
        dtype=DTYPE,
    )

    constraints_u = (
        make_control_box_constraints(
            u_min,
            u_max,
        )
    )

    # =================================================================
    # State limits
    # =================================================================

    max_tilt = jnp.deg2rad(
        jnp.asarray(
            80.0,
            dtype=DTYPE,
        )
    )

    x_min = jnp.array(
        [
            # position
            -100.0,
            -100.0,
            0.0,

            # attitude
            -max_tilt,
            -max_tilt,
            -20.0 * jnp.pi,

            # velocity
            -50.0,
            -50.0,
            -50.0,

            # rates
            -15.0,
            -15.0,
            -15.0,

            # mass
            DRY_MASS,

            # absolute clock
            0.0,

            # T
            MIN_TIME,
        ],
        dtype=DTYPE,
    )

    x_max = jnp.array(
        [
            # position
            100.0,
            100.0,
            100.0,

            # attitude
            max_tilt,
            max_tilt,
            20.0 * jnp.pi,

            # velocity
            50.0,
            50.0,
            50.0,

            # rates
            15.0,
            15.0,
            15.0,

            # mass
            INITIAL_MASS,

            # absolute clock
            100.0,

            # T
            MAX_TIME,
        ],
        dtype=DTYPE,
    )

    constraints_x = (
        make_state_box_constraints(
            x_min,
            x_max,
        )
    )

    # =================================================================
    # Moving-target terminal set
    # =================================================================

    terminal_constraint = (
        make_moving_target_terminal_constraint(
            N=N,

            position_half_width=jnp.array(
                [
                    0.25,
                    0.25,
                    0.25,
                ],
                dtype=DTYPE,
            ),

            # Start with position-only interception.
            velocity_half_width=None,
        )
    )

    constraints_all = combine_constraints(
        constraints_x,
        constraints_u,
        terminal_constraint,
    )

    # =================================================================
    # No explicit obstacle geometry
    # =================================================================

    obstacles = jnp.zeros(
        (
            0,
            3,
        ),
        dtype=DTYPE,
    )

    # =================================================================
    # Small SLS disturbance model
    # =================================================================

    disturbance = (
        make_min_time_disturbance(
            n=NX,
            N=N,
            magnitude=1e-6,
        )
    )

    # =================================================================
    # Initial SQP nominal trajectory
    # =================================================================

    X_initial = (
        build_initial_reference(
            x0=x0,
            N=N,
            duration=INITIAL_DURATION,
        )
    )

    # Initial hover-ish thrust
    initial_thrust = (
        INITIAL_MASS
        * GRAVITY
    )

    U_initial = jnp.zeros(
        (
            N,
            NU,
        ),
        dtype=DTYPE,
    )

    U_initial = (
        U_initial.at[
            :,
            0,
        ].set(
            initial_thrust
        )
    )

    # =================================================================
    # MPC config
    # =================================================================

    mpc_cfg = MPCConfig(
        n=NX,
        nu=NU,
        N=N,
        W=W,

        u_ref=jnp.array(
            [
                initial_thrust,
                0.0,
                0.0,
                0.0,
            ],
            dtype=DTYPE,
        ),
    )

    # =================================================================
    # ADMM
    # =================================================================

    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-3,

        rho_max=1e6,

        max_iterations=1000,

        rho_update_frequency=25,

        initial_rho=1.0,

        regularized_rho_update=False,
    )

    # =================================================================
    # SLS
    # =================================================================

    sls_cfg = SLSConfig(
        max_sls_iterations=1,

        sls_primal_tol=1e-2,

        enable_fastsls=True,

        # Full initial solve
        initialize_nominal=True,

        max_initial_sqp_iterations=100,

        # Then warm-start RTI
        warm_start=True,

        rti=True,

        gradient_window=0,
    )

    # =================================================================
    # SQP
    # =================================================================

    sqp_cfg = SQPConfig(
        # One SQP iteration for each RTI update
        max_sqp_iterations=1,

        warm_start=True,

        feas_tol=1e-8,
        step_tol=1e-8,

        # RTI typically takes full step.
        line_search=False,
    )

    # =================================================================
    # Controller
    # =================================================================

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,

        config=mpc_cfg,

        dynamics=dynamics,

        constraints=constraints_all,

        obstacles=obstacles,

        cost=cost,

        disturbance=disturbance,

        shift=1,

        X_in=X_initial,

        U_in=U_initial,
    )

    # =================================================================
    # Reference argument
    #
    # Cost does not actually use reference because target position
    # depends directly on the absolute-time state.
    #
    # GenericMPC still expects one.
    # =================================================================

    reference = jnp.zeros(
        (
            N + 1,
            NX,
        ),
        dtype=DTYPE,
    )

    # =================================================================
    # Closed-loop minimum-time RTI
    # =================================================================

    x = x0

    rocket_history = [
        np.asarray(
            x
        )
    ]

    target_position, target_velocity = (
        target_state(
            x[IDX_CLOCK]
        )
    )

    target_history = [
        np.asarray(
            target_position
        )
    ]

    control_history = []

    optimized_times = []

    solve_times = []

    predicted_rocket_plans = []

    predicted_target_plans = []

    intercepted = False

    print()
    print("=" * 72)
    print("MINIMUM-TIME ROCKET INTERCEPTION USING SQP-RTI")
    print("=" * 72)

    print(
        f"Horizon: {N}"
    )

    print(
        f"Initial T guess: "
        f"{INITIAL_DURATION:.3f} s"
    )

    print(
        f"Target initial position: "
        f"{np.asarray(target_position)}"
    )

    print()

    for iteration in range(
        MAX_RTI_STEPS
    ):

        # -------------------------------------------------------------
        # SQP RTI solve
        # -------------------------------------------------------------

        solve_start = (
            time.perf_counter()
        )

        (
            u0,
            X_pred,
            U_pred,
            V_pred,
            backoffs,
            Phi_x,
            Phi_u,
        ) = controller.run(
            x0=x,
            reference=reference,
            parameter=DELTA_TAU,
        )

        jax.block_until_ready(
            u0
        )

        solve_time = (
            time.perf_counter()
            - solve_start
        )

        solve_times.append(
            solve_time
        )

        # -------------------------------------------------------------
        # Optimized interception time
        # -------------------------------------------------------------

        optimal_time = float(
            X_pred[
                0,
                IDX_TIME,
            ]
        )

        optimal_time = float(
            np.clip(
                optimal_time,
                MIN_TIME,
                MAX_TIME,
            )
        )

        optimized_times.append(
            optimal_time
        )

        # Store optimized T as warm-start value in measured state.
        x = x.at[
            IDX_TIME
        ].set(
            optimal_time
        )

        # -------------------------------------------------------------
        # The first normalized interval has physical duration:
        #
        #       dt = T / N
        #
        # exactly like your original free-final-time formulation.
        # -------------------------------------------------------------

        dt_apply = (
            optimal_time
            / N
        )

        # -------------------------------------------------------------
        # First optimal control
        # -------------------------------------------------------------

        u = jnp.clip(
            u0,
            u_min,
            u_max,
        )

        control_history.append(
            np.asarray(
                u
            )
        )

        # -------------------------------------------------------------
        # Store predicted trajectories
        # -------------------------------------------------------------

        if (
            iteration
            % PLOT_PREDICTION_EVERY
            == 0
        ):

            predicted_rocket_plans.append(
                np.asarray(
                    X_pred
                )
            )

            target_prediction = []

            for k in range(
                N + 1
            ):

                p_target, _ = (
                    target_state(
                        X_pred[
                            k,
                            IDX_CLOCK,
                        ]
                    )
                )

                target_prediction.append(
                    np.asarray(
                        p_target
                    )
                )

            predicted_target_plans.append(
                np.asarray(
                    target_prediction
                )
            )

        # -------------------------------------------------------------
        # Apply first control to nonlinear plant
        # -------------------------------------------------------------

        x = simulate_rocket_step(
            x,
            u,
            dt_apply,
        )

        # -------------------------------------------------------------
        # Actual moving target
        # -------------------------------------------------------------

        (
            target_position,
            target_velocity,
        ) = target_state(
            x[IDX_CLOCK]
        )

        # -------------------------------------------------------------
        # Record
        # -------------------------------------------------------------

        rocket_history.append(
            np.asarray(
                x
            )
        )

        target_history.append(
            np.asarray(
                target_position
            )
        )

        # -------------------------------------------------------------
        # Interception metric
        # -------------------------------------------------------------

        position_error = (
            x[:3]
            - target_position
        )

        distance = float(
            jnp.linalg.norm(
                position_error
            )
        )

        relative_velocity = (
            x[6:9]
            - target_velocity
        )

        relative_speed = float(
            jnp.linalg.norm(
                relative_velocity
            )
        )

        # -------------------------------------------------------------
        # Print RTI status
        # -------------------------------------------------------------

        if (
            iteration % 5 == 0
            or distance
            <= CAPTURE_RADIUS
        ):

            print(
                f"RTI {iteration:4d} | "
                f"clock={float(x[IDX_CLOCK]):7.3f}s | "
                f"T*={optimal_time:7.4f}s | "
                f"dt={dt_apply:7.5f}s | "
                f"dist={distance:7.3f}m | "
                f"rel_v={relative_speed:7.3f}m/s | "
                f"thrust={float(u[0]):7.2f}N | "
                f"solve={1000.0 * solve_time:8.3f}ms"
            )

        # -------------------------------------------------------------
        # Capture condition
        # -------------------------------------------------------------

        captured_position = (
            distance
            <= CAPTURE_RADIUS
        )

        if USE_CAPTURE_VELOCITY:

            captured_velocity = (
                relative_speed
                <= CAPTURE_REL_VEL
            )

        else:

            captured_velocity = True

        if (
            captured_position
            and captured_velocity
        ):

            intercepted = True

            print()
            print("=" * 72)
            print("TARGET INTERCEPTED")
            print("=" * 72)

            print(
                f"Absolute interception time: "
                f"{float(x[IDX_CLOCK]):.6f} s"
            )

            print(
                f"Final distance: "
                f"{distance:.6f} m"
            )

            print(
                f"Relative speed: "
                f"{relative_speed:.6f} m/s"
            )

            print(
                f"Final optimized remaining T: "
                f"{optimal_time:.6f} s"
            )

            print(
                f"Remaining mass: "
                f"{float(x[IDX_MASS]):.6f} kg"
            )

            break

        # -------------------------------------------------------------
        # Fuel failure
        # -------------------------------------------------------------

        if (
            float(
                x[IDX_MASS]
            )
            <= DRY_MASS + 1e-4
        ):

            print()
            print(
                "Rocket reached dry mass."
            )

            break

    # =================================================================
    # Convert logs
    # =================================================================

    rocket_history = np.asarray(
        rocket_history
    )

    target_history = np.asarray(
        target_history
    )

    control_history = np.asarray(
        control_history
    )

    optimized_times = np.asarray(
        optimized_times
    )

    solve_times = np.asarray(
        solve_times
    )

    # =================================================================
    # Summary
    # =================================================================

    distance_history = (
        np.linalg.norm(
            rocket_history[:, :3]
            - target_history[:, :3],
            axis=1,
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Intercepted: {intercepted}"
    )

    print(
        f"Elapsed physical time: "
        f"{rocket_history[-1, IDX_CLOCK]:.6f} s"
    )

    print(
        f"Minimum distance: "
        f"{np.min(distance_history):.6f} m"
    )

    print(
        f"Final distance: "
        f"{distance_history[-1]:.6f} m"
    )

    print(
        f"Fuel used: "
        f"{INITIAL_MASS - rocket_history[-1, IDX_MASS]:.6f} kg"
    )

    if len(
        optimized_times
    ) > 0:

        print(
            f"Initial optimized T: "
            f"{optimized_times[0]:.6f} s"
        )

        print(
            f"Final optimized T: "
            f"{optimized_times[-1]:.6f} s"
        )

    # First solve includes initialization/JIT.
    if len(
        solve_times
    ) > 1:

        steady = (
            solve_times[1:]
        )

        print()
        print(
            "Steady-state SQP-RTI timing:"
        )

        print(
            f"  mean   = "
            f"{1000.0 * np.mean(steady):.3f} ms"
        )

        print(
            f"  median = "
            f"{1000.0 * np.median(steady):.3f} ms"
        )

        print(
            f"  p95    = "
            f"{1000.0 * np.percentile(steady, 95):.3f} ms"
        )

        print(
            f"  max    = "
            f"{1000.0 * np.max(steady):.3f} ms"
        )

    # =================================================================
    # Save experiment
    # =================================================================

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "rocket_min_time_tracking.npz",
        ),

        rocket_history=rocket_history,

        target_history=target_history,

        control_history=control_history,

        optimized_times=optimized_times,

        solve_times=solve_times,

        intercepted=intercepted,

        N=N,

        delta_tau=DELTA_TAU,
    )

    # =================================================================
    # Plot
    # =================================================================

    plot_results(
        rocket_history=rocket_history,

        target_history=target_history,

        controls=control_history,

        optimized_times=optimized_times,

        solve_times=solve_times,

        predicted_rocket_plans=(
            predicted_rocket_plans
        ),

        predicted_target_plans=(
            predicted_target_plans
        ),
    )

    print()
    print(
        f"Results saved in {OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()