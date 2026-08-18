from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import config
import matplotlib.pyplot as plt
import numpy as np

from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
    make_state_box_constraints,
)

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

config.update("jax_enable_x64", False)
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_compile_time_secs", 0)
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)

DTYPE = jnp.float32

# -----------------------------------------------------------------------------
# Deterministic minimum-time interception model
# -----------------------------------------------------------------------------
# Relative state:
#   x = [rx, ry, rz, vrx, vry, vrz, Tf]
# where
#   r  = p_target - p_interceptor
#   vr = v_target - v_interceptor
#   Tf = optimized final time (constant along the prediction horizon)
#
# Control:
#   u = [aIx, aIy, aIz]
# is the interceptor acceleration command.
#
# With a known deterministic target acceleration aT(t),
#   r_dot  = vr
#   vr_dot = aT(t) - u
# Therefore x_{k+1} = F(x_k, u_k, k) is deterministic and Markovian.
# -----------------------------------------------------------------------------

N_REL = 6
N_STATE = 7
N_CONTROL = 3
CAPTURE_RADIUS = 0.20


def target_acceleration(t_normalized: jnp.ndarray) -> jnp.ndarray:
    """Known deterministic target maneuver.

    Set this to zero for a constant-velocity target. The example below uses a
    small smooth lateral maneuver, known to the optimizer.
    """
    return jnp.array(
        [
            0.0,
            0.20 * jnp.sin(2.0 * jnp.pi * t_normalized),
            0.08 * jnp.cos(2.0 * jnp.pi * t_normalized),
        ],
        dtype=DTYPE,
    )


def relative_intercept_step(
    x: jnp.ndarray,
    u: jnp.ndarray,
    dt: jnp.ndarray,
    t_normalized: jnp.ndarray,
) -> jnp.ndarray:
    """One deterministic discrete-time Markov transition.

    Constant acceleration is assumed over the interval. This gives a more
    accurate transition than forward Euler for the double-integrator model.
    """
    r = x[:3]
    v_rel = x[3:6]
    final_time = x[6]

    a_rel = target_acceleration(t_normalized) - u

    r_next = r + dt * v_rel + 0.5 * dt**2 * a_rel
    v_rel_next = v_rel + dt * a_rel

    return jnp.concatenate(
        [r_next, v_rel_next, jnp.reshape(final_time, (1,))], axis=0
    )


def dynamics(
    x: jnp.ndarray,
    u: jnp.ndarray,
    t: jnp.ndarray,
    *,
    parameter: Any,
) -> jnp.ndarray:
    """GenericMPC dynamics callback.

    parameter = 1 / N, so dt = Tf / N. The normalized stage time t/N is used
    only by the known target maneuver.
    """
    inv_horizon = jnp.asarray(parameter, dtype=x.dtype)
    dt = inv_horizon * x[-1]
    t_normalized = jnp.asarray(t, dtype=x.dtype) * inv_horizon
    return relative_intercept_step(x, u, dt, t_normalized)


def cost(W, reference, x, u, t):
    """Minimum-time objective with mild regularization.

    W = [wr, wv, wu, wtime]

    The time term is divided by N through reference.shape[0] - 1, so summing
    the stage cost approximately contributes wtime * Tf rather than N*wtime*Tf.
    """
    wr, wv, wu, wtime = W
    x_ref = reference[t]

    r_err = x[:3] - x_ref[:3]
    v_err = x[3:6] - x_ref[3:6]
    horizon = jnp.asarray(reference.shape[0] - 1, dtype=x.dtype)

    return (
        wr * jnp.dot(r_err, r_err)
        + wv * jnp.dot(v_err, v_err)
        + wu * jnp.dot(u, u)
        + wtime * x[-1] / horizon
    )


def build_relative_reference(x0: jnp.ndarray, N: int) -> jnp.ndarray:
    """Straight-line warm-start reference from initial separation to capture."""
    alpha = jnp.linspace(0.0, 1.0, N + 1, dtype=DTYPE)
    reference = jnp.zeros((N + 1, N_STATE), dtype=DTYPE)
    reference = reference.at[:, :3].set((1.0 - alpha[:, None]) * x0[:3])
    reference = reference.at[:, 3:6].set((1.0 - alpha[:, None]) * x0[3:6])
    reference = reference.at[:, -1].set(x0[-1])
    return reference


def make_terminal_capture_constraint(capture_radius: float, N: int):
    """At the final stage enforce ||r_N||_2 <= capture_radius."""
    radius_sq = jnp.asarray(capture_radius**2, dtype=DTYPE)

    def constraints(x, u, t):
        terminal_value = jnp.dot(x[:3], x[:3]) - radius_sq
        inactive_value = jnp.asarray(-1.0, dtype=x.dtype)
        return jnp.reshape(jnp.where(t == N, terminal_value, inactive_value), (1,))

    return constraints


def make_zero_disturbance(n: int):
    """Zero uncertainty map: deterministic dynamics only."""

    def at_state(x_k: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros((n, n), dtype=x_k.dtype)

    def disturbance(X: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(at_state)(X)

    disturbance.at_state = at_state
    return disturbance


def rollout_relative_trajectory(
    x0: jnp.ndarray,
    U: jnp.ndarray,
    final_time: float,
    N: int,
) -> np.ndarray:
    """Deterministically roll out the optimized open-loop control sequence."""
    dt = jnp.asarray(final_time / N, dtype=DTYPE)
    x = x0.at[-1].set(final_time)
    states = [np.asarray(x)]

    for k in range(N):
        x = relative_intercept_step(
            x,
            U[k],
            dt,
            jnp.asarray(k / N, dtype=DTYPE),
        )
        states.append(np.asarray(x))

    return np.stack(states)


def reconstruct_absolute_trajectories(
    relative_states: np.ndarray,
    target_p0: np.ndarray,
    target_v0: np.ndarray,
    final_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct target and interceptor positions for visualization."""
    N = relative_states.shape[0] - 1
    dt = final_time / N

    target_positions = np.zeros((N + 1, 3), dtype=np.float64)
    target_vel = np.asarray(target_v0, dtype=np.float64).copy()
    target_positions[0] = np.asarray(target_p0, dtype=np.float64)

    for k in range(N):
        a_t = np.asarray(target_acceleration(jnp.asarray(k / N, dtype=DTYPE)))
        target_positions[k + 1] = (
            target_positions[k] + dt * target_vel + 0.5 * dt**2 * a_t
        )
        target_vel = target_vel + dt * a_t

    interceptor_positions = target_positions - relative_states[:, :3]
    return interceptor_positions, target_positions


def plot_intercept(
    interceptor_positions: np.ndarray,
    target_positions: np.ndarray,
    relative_states: np.ndarray,
    final_time: float,
    filename: str,
) -> None:
    """Save a simple 3-D trajectory and range history visualization."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        interceptor_positions[:, 0],
        interceptor_positions[:, 1],
        interceptor_positions[:, 2],
        label="Interceptor",
    )
    ax.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        linestyle="--",
        label="Target",
    )
    ax.scatter(*interceptor_positions[0], marker="o", label="Interceptor start")
    ax.scatter(*target_positions[0], marker="^", label="Target start")
    ax.scatter(*target_positions[-1], marker="x", s=80, label="Intercept")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Deterministic minimum-time intercept: Tf={final_time:.3f} s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)

    time = np.linspace(0.0, final_time, relative_states.shape[0])
    range_history = np.linalg.norm(relative_states[:, :3], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, range_history)
    ax.axhline(CAPTURE_RADIUS, linestyle="--", label="Capture radius")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Relative range")
    ax.set_title("Interceptor-target separation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("intercept_range_history.png", dpi=180)
    plt.close(fig)


def main() -> None:
    # ------------------------------------------------------------------
    # Horizon and free final time
    # ------------------------------------------------------------------
    N = 1000
    parameter = 1.0 / N
    initial_duration = 8.0

    # Mostly minimum-time, with small state/control regularization for SQP.
    W = jnp.array([1e-2, 1e-2, 1e-2, 1.0], dtype=DTYPE)

    cfg = MPCConfig(
        n=N_STATE,
        nu=N_CONTROL,
        N=N,
        W=W,
        u_ref=jnp.zeros((N_CONTROL,), dtype=DTYPE),
    )

    # ------------------------------------------------------------------
    # Interceptor acceleration bounds
    # ------------------------------------------------------------------
    a_max = 3.0
    u_min = -a_max * jnp.ones((N_CONTROL,), dtype=DTYPE)
    u_max = a_max * jnp.ones((N_CONTROL,), dtype=DTYPE)
    constraints_u = make_control_box_constraints(u_min, u_max)

    # ------------------------------------------------------------------
    # Relative-state and final-time bounds
    # ------------------------------------------------------------------
    x_min = jnp.array(
        [-100.0, -100.0, -100.0, -30.0, -30.0, -30.0, 0.25],
        dtype=DTYPE,
    )
    x_max = jnp.array(
        [100.0, 100.0, 100.0, 30.0, 30.0, 30.0, 30.0],
        dtype=DTYPE,
    )
    constraints_x = make_state_box_constraints(x_min, x_max)
    terminal_constraint = make_terminal_capture_constraint(CAPTURE_RADIUS, N)
    constraints_all = combine_constraints(
        constraints_x,
        constraints_u,
        terminal_constraint,
    )

    # ------------------------------------------------------------------
    # Initial absolute states, converted to a relative Markov state
    # ------------------------------------------------------------------
    interceptor_p0 = jnp.array([0.0, 0.0, 0.0], dtype=DTYPE)
    interceptor_v0 = jnp.array([0.0, 0.0, 0.0], dtype=DTYPE)
    target_p0 = jnp.array([8.0, 3.0, 1.5], dtype=DTYPE)
    target_v0 = jnp.array([0.7, -0.15, 0.0], dtype=DTYPE)

    relative_p0 = target_p0 - interceptor_p0
    relative_v0 = target_v0 - interceptor_v0
    x0 = jnp.concatenate(
        [
            relative_p0,
            relative_v0,
            jnp.array([initial_duration], dtype=DTYPE),
        ]
    )

    X_ref = build_relative_reference(x0, N)
    U_ref = jnp.zeros((N, N_CONTROL), dtype=DTYPE)

    # ------------------------------------------------------------------
    # Solver configuration
    # ------------------------------------------------------------------
    admm_cfg = ADMMConfig(
        eps_abs=5e-2,
        eps_rel=1e-3,
        rho_max=1e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=1.0,
        regularized_rho_update=False,
    )

    sls_cfg = SLSConfig(
        max_sls_iterations=1,
        sls_primal_tol=1e-2,
        enable_fastsls=False,
        initialize_nominal=True,
        max_initial_sqp_iterations=0,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )

    sqp_cfg = SQPConfig(
        max_sqp_iterations=100,
        warm_start=True,
        feas_tol=1e-8,
        step_tol=1e-8,
        line_search=True,
    )

    controller = GenericMPC(
        sls_cfg,
        sqp_cfg,
        admm_cfg,
        config=cfg,
        dynamics=dynamics,
        constraints=constraints_all,
        obstacles=jnp.zeros((0, 3), dtype=DTYPE),
        cost=cost,
        disturbance=make_zero_disturbance(N_STATE),
        shift=1,
        X_in=X_ref,
        U_in=U_ref,
    )

    # ------------------------------------------------------------------
    # Deterministic minimum-time plan
    # ------------------------------------------------------------------

    _, X_pred, U_pred, _, _, _, _ = controller.run(
        x0=x0,
        reference=X_ref,
        parameter=parameter,
    )

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=True):
    #     _, X_pred, U_pred, _, _, _, _ = controller.run(
    #         x0=x0,
    #         reference=X_ref,
    #         parameter=parameter,
    #     )

    # Tf is a state copied through every transition, so any row is equivalent.
    min_time = float(np.asarray(X_pred[0, -1]))
    print(f"Computed minimum intercept time: {min_time:.6f} s")

    relative_states = rollout_relative_trajectory(x0, U_pred, min_time, N)
    terminal_range = float(np.linalg.norm(relative_states[-1, :3]))
    print(f"Terminal relative range: {terminal_range:.6f}")
    print(f"Capture radius: {CAPTURE_RADIUS:.6f}")

    interceptor_positions, target_positions = reconstruct_absolute_trajectories(
        relative_states,
        np.asarray(target_p0),
        np.asarray(target_v0),
        min_time,
    )
    plot_intercept(
        interceptor_positions,
        target_positions,
        relative_states,
        min_time,
        filename="deterministic_min_time_intercept.png",
    )


if __name__ == "__main__":
    main()
