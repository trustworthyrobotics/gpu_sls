#!/usr/bin/env python3
"""
Parallel Monte Carlo closed-loop rollouts of the saved Go2 barrel-roll plan.

All Monte Carlo rollouts are advanced in parallel with JAX:

    outer parallel axis:   rollout index r = 0,...,R-1
    sequential axis:       time k = 0,...,N-1

At each step:

    dx_j^r = x_j^r - X[j]

    du_k^r = sum_{j=0}^k K_x[k,j] dx_j^r
    u_k^r  = U[k] + du_k^r

    x_next_nom^r = f(x_k^r, u_k^r, k)
    x_{k+1}^r    = x_next_nom^r + E(x_k^r,k) w_k^r

The time recursion must remain sequential because x_{k+1} depends on x_k,
but every rollout at a given time step is evaluated together on the GPU.

The optimizer is NOT rerun.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(
    "/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/"
    "gpu_sls/examples/quadruped_barrel_roll_obstacle_robust_new"
)

INPUT_NPZ = EXPERIMENT_DIR / "quadruped_barrel_roll_obstacle_min_time.npz"

OUTPUT_NPZ = (
    EXPERIMENT_DIR
    / "quadruped_barrel_roll_random_disturbance_feedback_rollouts.npz"
)

sys.path.insert(0, str(EXPERIMENT_DIR))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np

import quadruped_barrel_roll as barrel


# -----------------------------------------------------------------------------
# Monte Carlo settings
# -----------------------------------------------------------------------------

NUM_ROLLOUTS = 256
RNG_SEED = 0

DISTURBANCE_RADIUS = 1.0

# True:
#     ||w_k||_2 = DISTURBANCE_RADIUS
#
# False:
#     uniform-by-volume sample inside the l2 ball.
SAMPLE_ON_BOUNDARY = False

# Do not spend disturbance norm on zero E columns, especially duration states.
SAMPLE_ONLY_ACTIVE_CHANNELS = True

# Leave False to test the actual robust feedback law without hiding saturation.
CLIP_CONTROLS = False

# False exactly matches the additive Euclidean disturbance model used by SLS.
RENORMALIZE_QUATERNION = False

TUBE_EPS = 1.0e-10


# -----------------------------------------------------------------------------
# Saved solution
# -----------------------------------------------------------------------------

def load_solution():
    if not INPUT_NPZ.is_file():
        raise FileNotFoundError(INPUT_NPZ)

    with np.load(INPUT_NPZ, allow_pickle=False) as f:
        required = [
            "X",
            "U",
            "K_x_history",
            "trajectory_tubes",
        ]

        missing = [
            key for key in required
            if key not in f.files
        ]

        if missing:
            raise KeyError(
                f"Missing keys {missing}; available={f.files}"
            )

        saved = {
            key: np.asarray(f[key])
            for key in f.files
        }

    X = saved["X"]
    U = saved["U"]
    K = saved["K_x_history"]
    tubes = saved["trajectory_tubes"]

    N = U.shape[0]

    if X.shape[0] != N + 1:
        raise ValueError(
            f"Expected N+1 states: X={X.shape}, U={U.shape}"
        )

    if K.shape[:2] != (N, N):
        raise ValueError(
            f"Expected K horizon {(N, N)}, got {K.shape}"
        )

    if K.shape[2] != U.shape[1]:
        raise ValueError(
            f"K control dim {K.shape[2]} != U dim {U.shape[1]}"
        )

    if K.shape[3] != barrel.PHYSICAL_N:
        raise ValueError(
            f"K physical state dim {K.shape[3]} != "
            f"PHYSICAL_N={barrel.PHYSICAL_N}"
        )

    # Align saved tube rows to x_0,...,x_N.
    if tubes.shape[0] == N:
        tubes = np.concatenate(
            [
                np.zeros(
                    (1, tubes.shape[1]),
                    dtype=tubes.dtype,
                ),
                tubes,
            ],
            axis=0,
        )

    elif tubes.shape[0] != N + 1:
        raise ValueError(
            f"Cannot align tubes={tubes.shape} with X={X.shape}"
        )

    return saved, X, U, K, tubes


# -----------------------------------------------------------------------------
# Rebuild nonlinear plant and disturbance model
# -----------------------------------------------------------------------------

def build_model():
    """Rebuild the exact nonlinear dynamics and E callback used by the planner."""

    barrel.configure_problem()

    reference, parameter = barrel.build_barrel_roll_reference()

    obstacle_constraints = barrel.make_obstacle_clearance_constraints()

    constraints = barrel.make_barrel_roll_constraints(
        reference,
        obstacle_constraints,
    )

    admm_config = barrel.ADMMConfig(
        eps_abs=6.0e-3,
        eps_rel=1.0e-3,
        rho_max=1.0e6,
        max_iterations=1000,
        rho_update_frequency=25,
        initial_rho=25.0,
        regularized_rho_update=False,
        num_phases=barrel.NUM_PHASES,
    )

    sls_config = barrel.SLSConfig(
        max_sls_iterations=2,
        sls_primal_tol=1.0e-2,
        enable_fastsls=True,
        initialize_nominal=True,
        max_initial_sqp_iterations=50,
        warm_start=True,
        rti=False,
        gradient_window=0,
    )

    sqp_config = barrel.SQPConfig(
        max_sqp_iterations=50,
        warm_start=True,
        feas_tol=1.0e-5,
        step_tol=1.0e-5,
        line_search=True,
        lm_regularization=5.0e-2,
    )

    disturbance_model = barrel.make_phase_scaled_disturbance(
        barrel.config.n
    )

    mpc = barrel.mpc_wrapper.MPCWrapper(
        barrel.config,
        sls_config=sls_config,
        sqp_config=sqp_config,
        admm_config=admm_config,
        constraints=constraints,
        obstacles=jnp.zeros(
            (0, 3),
            dtype=barrel.config.initial_state.dtype,
        ),
        disturbance=disturbance_model,
    )

    return mpc, parameter, disturbance_model


# -----------------------------------------------------------------------------
# Parallel disturbance generation
# -----------------------------------------------------------------------------

def generate_random_sources(
    key: jax.Array,
    num_rollouts: int,
    horizon: int,
    disturbance_dim: int,
):
    """
    Generate the random numbers needed by every rollout and every time step.

    We generate Gaussian directions and scalar radial samples in one batch.
    The E-dependent active-channel mask is applied later inside the scan.
    """

    key_direction, key_radius = jax.random.split(key)

    directions = jax.random.normal(
        key_direction,
        shape=(
            num_rollouts,
            horizon,
            disturbance_dim,
        ),
    )

    radial_uniform = jax.random.uniform(
        key_radius,
        shape=(
            num_rollouts,
            horizon,
        ),
        minval=0.0,
        maxval=1.0,
    )

    return directions, radial_uniform


def sample_w_parallel(
    E_batch: jnp.ndarray,
    raw_direction: jnp.ndarray,
    radial_uniform: jnp.ndarray,
):
    """
    Convert random sources into one l2-ball disturbance per rollout.

    Inputs:
        E_batch          (R, nx, nw)
        raw_direction    (R, nw)
        radial_uniform   (R,)

    Output:
        w                (R, nw)
    """

    if SAMPLE_ONLY_ACTIVE_CHANNELS:
        active = (
            jnp.linalg.norm(
                E_batch,
                axis=1,
            )
            > 0.0
        )
    else:
        active = jnp.ones(
            raw_direction.shape,
            dtype=bool,
        )

    z = jnp.where(
        active,
        raw_direction,
        0.0,
    )

    z_norm = jnp.linalg.norm(
        z,
        axis=1,
        keepdims=True,
    )

    # Extremely unlikely for Gaussian draws, but keep normalization safe.
    z = z / jnp.maximum(
        z_norm,
        1.0e-12,
    )

    if SAMPLE_ON_BOUNDARY:
        radius = jnp.full(
            (E_batch.shape[0],),
            DISTURBANCE_RADIUS,
            dtype=E_batch.dtype,
        )

    else:
        # Uniform volume sample in the active d-dimensional Euclidean ball:
        #
        #     r = R * U^(1/d)
        #
        active_dim = jnp.sum(
            active,
            axis=1,
        )

        active_dim = jnp.maximum(
            active_dim,
            1,
        )

        radius = (
            DISTURBANCE_RADIUS
            * radial_uniform
            ** (
                1.0
                / active_dim.astype(E_batch.dtype)
            )
        )

    return z * radius[:, None]


def normalize_quaternion_parallel(
    X_batch: jnp.ndarray,
) -> jnp.ndarray:
    if not RENORMALIZE_QUATERNION:
        return X_batch

    q = X_batch[:, 3:7]

    q_norm = jnp.linalg.norm(
        q,
        axis=1,
        keepdims=True,
    )

    q_normalized = q / jnp.maximum(
        q_norm,
        1.0e-12,
    )

    return X_batch.at[:, 3:7].set(
        q_normalized
    )


# -----------------------------------------------------------------------------
# Batched closed-loop rollout
# -----------------------------------------------------------------------------

def make_parallel_rollout(
    dynamics,
    disturbance_model,
    parameter,
):
    """
    Build one jitted Monte Carlo rollout.

    Time is handled by lax.scan.
    Rollout dimension is handled by vmap.
    """

    # Vectorize the nonlinear dynamics over rollout index.
    batched_dynamics = jax.vmap(
        dynamics,
        in_axes=(0, 0, None, None),
        out_axes=0,
    )

    # Vectorize E(x,k) over rollout index.
    batched_disturbance = jax.vmap(
        disturbance_model.at_state,
        in_axes=(0, None),
        out_axes=0,
    )

    @jax.jit
    def rollout_batch(
        X_nominal,
        U_nominal,
        K,
        random_directions,
        random_radii,
    ):
        """
        Shapes:
            X_nominal          (N+1, nx)
            U_nominal          (N, nu)
            K                  (N, N, nu, physical_n)
            random_directions  (R, N, nx)
            random_radii       (R, N)
        """

        R = random_directions.shape[0]
        N = U_nominal.shape[0]
        nx = X_nominal.shape[1]

        x0_batch = jnp.broadcast_to(
            X_nominal[0],
            (R, nx),
        )

        # Full state-error history for each rollout.
        error_history0 = jnp.zeros(
            (
                R,
                N + 1,
                nx,
            ),
            dtype=X_nominal.dtype,
        )

        # dx_0 = 0 because every rollout starts exactly at nominal x0.
        error_history0 = error_history0.at[:, 0, :].set(
            x0_batch - X_nominal[0]
        )

        def scan_step(carry, k):
            x_batch, error_history = carry

            # ---------------------------------------------------------
            # State-history feedback for all rollouts simultaneously.
            #
            # K[k,j] maps dx_j -> du_k.
            # Mask future history j > k.
            # ---------------------------------------------------------

            K_k = K[k]  # (N, nu, physical_n)

            history_mask = (
                jnp.arange(N) <= k
            ).astype(X_nominal.dtype)

            K_k_causal = (
                K_k
                * history_mask[:, None, None]
            )

            # (R,N,p) x (N,m,p) -> (R,m)
            du_batch = jnp.einsum(
                "rjp,jmp->rm",
                error_history[
                    :, :N, :barrel.PHYSICAL_N
                ],
                K_k_causal,
            )

            u_batch = (
                U_nominal[k][None, :]
                + du_batch
            )

            if CLIP_CONTROLS:
                u_batch = jnp.clip(
                    u_batch,
                    jnp.asarray(
                        barrel.config.min_torque
                    ),
                    jnp.asarray(
                        barrel.config.max_torque
                    ),
                )

            # ---------------------------------------------------------
            # Exact nonlinear plant transition, vectorized over R.
            # ---------------------------------------------------------

            x_det_next = batched_dynamics(
                x_batch,
                u_batch,
                k,
                parameter,
            )

            # ---------------------------------------------------------
            # Exact E callback used by SLS, also vectorized over R.
            # ---------------------------------------------------------

            E_batch = batched_disturbance(
                x_batch,
                k,
            )

            w_batch = sample_w_parallel(
                E_batch,
                random_directions[:, k, :],
                random_radii[:, k],
            )

            disturbance_batch = jnp.einsum(
                "rij,rj->ri",
                E_batch,
                w_batch,
            )

            x_next = (
                x_det_next
                + disturbance_batch
            )

            x_next = normalize_quaternion_parallel(
                x_next
            )

            error_next = (
                x_next
                - X_nominal[k + 1][None, :]
            )

            error_history = error_history.at[
                :, k + 1, :
            ].set(error_next)

            outputs = (
                x_next,
                u_batch,
                error_next,
                du_batch,
                w_batch,
                disturbance_batch,
            )

            return (
                x_next,
                error_history,
            ), outputs

        (
            _,
            error_history_final,
        ), outputs = jax.lax.scan(
            scan_step,
            (
                x0_batch,
                error_history0,
            ),
            jnp.arange(
                N,
                dtype=jnp.int32,
            ),
        )

        (
            X_next_time_major,
            U_time_major,
            error_next_time_major,
            dU_time_major,
            W_time_major,
            D_time_major,
        ) = outputs

        # lax.scan returns time-major arrays:
        #     (N,R,...)
        # Convert to rollout-major:
        #     (R,N,...)
        X_successors = jnp.swapaxes(
            X_next_time_major,
            0,
            1,
        )

        U_all = jnp.swapaxes(
            U_time_major,
            0,
            1,
        )

        dU_all = jnp.swapaxes(
            dU_time_major,
            0,
            1,
        )

        W_all = jnp.swapaxes(
            W_time_major,
            0,
            1,
        )

        D_all = jnp.swapaxes(
            D_time_major,
            0,
            1,
        )

        X_all = jnp.concatenate(
            [
                x0_batch[:, None, :],
                X_successors,
            ],
            axis=1,
        )

        return (
            X_all,
            U_all,
            error_history_final,
            dU_all,
            W_all,
            D_all,
        )

    return rollout_batch


# -----------------------------------------------------------------------------
# Batched diagnostics
# -----------------------------------------------------------------------------

def compute_diagnostics(
    X_all,
    U_all,
    error_history,
    tubes,
):
    """
    All diagnostics are vectorized over rollouts.
    """

    physical_error = jnp.abs(
        error_history[
            :, :, :barrel.PHYSICAL_N
        ]
    )

    tube = jnp.asarray(
        tubes[
            :, :barrel.PHYSICAL_N
        ]
    )

    valid = tube > TUBE_EPS

    ratio = jnp.where(
        valid[None, :, :],
        physical_error
        / jnp.maximum(
            tube[None, :, :],
            TUBE_EPS,
        ),
        jnp.nan,
    )

    state_err_max = jnp.max(
        physical_error,
        axis=(1, 2),
    )

    tube_ratio_max = jnp.nanmax(
        ratio,
        axis=(1, 2),
    )

    # Flatten node/state for each rollout.
    safe_ratio = jnp.where(
        jnp.isnan(ratio),
        -jnp.inf,
        ratio,
    )

    flat_idx = jnp.argmax(
        safe_ratio.reshape(
            safe_ratio.shape[0],
            -1,
        ),
        axis=1,
    )

    num_physical_states = barrel.PHYSICAL_N

    worst_nodes = (
        flat_idx
        // num_physical_states
    )

    worst_states = (
        flat_idx
        % num_physical_states
    )

    max_torque = jnp.asarray(
        barrel.config.max_torque
    )

    min_torque = jnp.asarray(
        barrel.config.min_torque
    )

    upper_violation = U_all - max_torque
    lower_violation = min_torque - U_all

    torque_violation = jnp.maximum(
        0.0,
        jnp.maximum(
            jnp.max(
                upper_violation,
                axis=(1, 2),
            ),
            jnp.max(
                lower_violation,
                axis=(1, 2),
            ),
        ),
    )

    return (
        state_err_max,
        tube_ratio_max,
        torque_violation,
        worst_nodes,
        worst_states,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    saved, X_np, U_np, K_np, tubes = load_solution()

    mpc, parameter, disturbance_model = build_model()

    X = jnp.asarray(X_np)
    U = jnp.asarray(U_np)
    K = jnp.asarray(K_np)

    N = U.shape[0]
    nx = X.shape[1]

    print(f"Input:             {INPUT_NPZ}")
    print(f"X/U/K:             {X.shape} / {U.shape} / {K.shape}")
    print(f"Parallel rollouts: {NUM_ROLLOUTS}")
    print(f"RNG seed:          {RNG_SEED}")
    print(f"||w_k|| radius:    {DISTURBANCE_RADIUS}")
    print(f"JAX devices:       {jax.devices()}")

    key = jax.random.PRNGKey(
        RNG_SEED
    )

    random_directions, random_radii = (
        generate_random_sources(
            key,
            NUM_ROLLOUTS,
            N,
            nx,
        )
    )

    rollout_batch = make_parallel_rollout(
        mpc.dynamics,
        disturbance_model,
        parameter,
    )

    print("Compiling/executing parallel rollout batch...")

    (
        X_all,
        U_all,
        Ehist_all,
        dU_all,
        W_all,
        D_all,
    ) = rollout_batch(
        X,
        U,
        K,
        random_directions,
        random_radii,
    )

    # Synchronize once for the complete Monte Carlo batch.
    X_all.block_until_ready()

    print("Parallel rollout batch complete.")

    (
        state_err_max,
        tube_ratio_max,
        torque_viol,
        worst_nodes,
        worst_states,
    ) = compute_diagnostics(
        X_all,
        U_all,
        Ehist_all,
        tubes,
    )

    # One device -> host transfer after all rollouts are complete.
    X_all = np.asarray(X_all)
    U_all = np.asarray(U_all)
    Ehist_all = np.asarray(Ehist_all)
    dU_all = np.asarray(dU_all)
    W_all = np.asarray(W_all)
    D_all = np.asarray(D_all)

    state_err_max = np.asarray(
        state_err_max
    )
    tube_ratio_max = np.asarray(
        tube_ratio_max
    )
    torque_viol = np.asarray(
        torque_viol
    )
    worst_nodes = np.asarray(
        worst_nodes
    )
    worst_states = np.asarray(
        worst_states
    )

    if np.any(
        np.isfinite(tube_ratio_max)
    ):
        worst_idx = int(
            np.nanargmax(
                tube_ratio_max
            )
        )
    else:
        worst_idx = int(
            np.argmax(
                state_err_max
            )
        )

    print()
    print("Summary")
    print("-------")
    print(
        f"Worst rollout:            {worst_idx}"
    )
    print(
        "Worst max |dx|/tube:      "
        f"{tube_ratio_max[worst_idx]:.6f}"
    )
    print(
        "Worst node/state:         "
        f"{worst_nodes[worst_idx]} / "
        f"{worst_states[worst_idx]}"
    )
    print(
        "Largest max |dx|:         "
        f"{np.max(state_err_max):.6e}"
    )
    print(
        "Largest torque violation: "
        f"{np.max(torque_viol):.6e}"
    )

    # Optional concise rollout table.
    print()
    print("Per-rollout diagnostics")
    print("-----------------------")

    for r in range(NUM_ROLLOUTS):
        print(
            f"[{r:03d}] "
            f"max|dx|={state_err_max[r]:.4e}  "
            f"max(|dx|/tube)={tube_ratio_max[r]:.4f}  "
            f"torque_violation={torque_viol[r]:.4e}"
        )

    metadata = {}

    for key_name in [
        "phase_names",
        "phase_end_steps",
        "phase_times",
        "node_times",
        "obstacle_center",
        "obstacle_size",
        "lateral_displacement",
    ]:
        if key_name in saved:
            metadata[key_name] = saved[
                key_name
            ]

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_NPZ,

        X=X_np,
        U=U_np,
        K_x_history=K_np,
        trajectory_tubes=tubes,

        X_random_feedback_rollouts=X_all,
        U_random_feedback_rollouts=U_all,
        x_perturbation_histories=Ehist_all,
        feedback_corrections_random=dU_all,
        W_random=W_all,
        disturbance_injections=D_all,

        max_state_errors=state_err_max,
        max_tube_ratios=tube_ratio_max,
        max_torque_violations=torque_viol,
        worst_tube_nodes=worst_nodes,
        worst_tube_states=worst_states,
        worst_rollout_index=np.asarray(
            worst_idx
        ),

        # Keep compatibility with the single-rollout plotter/renderer.
        X_state_feedback_rollout=X_all[
            worst_idx
        ],
        U_state_feedback_rollout=U_all[
            worst_idx
        ],
        x_perturbation_history=Ehist_all[
            worst_idx
        ],
        feedback_corrections=dU_all[
            worst_idx
        ],
        W_state_feedback_rollout=W_all[
            worst_idx
        ],
        disturbance_injection_state_feedback_rollout=D_all[
            worst_idx
        ],

        num_rollouts=np.asarray(
            NUM_ROLLOUTS
        ),
        rng_seed=np.asarray(
            RNG_SEED
        ),
        disturbance_radius=np.asarray(
            DISTURBANCE_RADIUS
        ),
        sample_on_boundary=np.asarray(
            SAMPLE_ON_BOUNDARY
        ),
        sample_only_active_channels=np.asarray(
            SAMPLE_ONLY_ACTIVE_CHANNELS
        ),
        clip_controls=np.asarray(
            CLIP_CONTROLS
        ),
        renormalize_quaternion=np.asarray(
            RENORMALIZE_QUATERNION
        ),

        **metadata,
    )

    print()
    print(f"Saved:\n  {OUTPUT_NPZ}")
    print(
        "X_state_feedback_rollout is the worst sampled rollout "
        "by max |dx|/tube."
    )


if __name__ == "__main__":
    main()
