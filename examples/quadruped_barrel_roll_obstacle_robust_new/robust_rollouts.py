"""
Parallel random + adversarial + coordinate-basis closed-loop rollouts of the
saved Go2 barrel-roll plan.

This script evaluates the saved robust trajectory and state-history feedback law
without rerunning the optimizer.

Three test groups are generated:

    1) RANDOM:       256 rollouts
       - each rollout is assigned one fixed disturbance family for the entire
         horizon (vx only, vy only, vz only, wx only, ...)
       - w_k is sampled from the allowed L2 ball inside that family's active
         disturbance subspace

    2) ADVERSARIAL:  256 rollouts
       - the same disturbance-family assignment is used
       - projected-gradient ascent optimizes the entire disturbance sequence
         w_0,...,w_{N-1} to increase a smooth approximation of

             max_{k,i} |x_k[i] - X_k[i]| / tube_k[i]

       - after every PGD step, each w_k is projected back into

             ||w_k||_2 <= DISTURBANCE_RADIUS

         and onto the rollout's fixed channel support

    3) BASIS +/-1: 2*nx deterministic rollouts
       - for each disturbance coordinate i, one rollout uses

             w_k = +e_i    for every k

         and another uses

             w_k = -e_i    for every k

       - examples:

             [1, 0, 0, ...] for the full rollout
             [-1, 0, 0, ...] for the full rollout
             [0, 1, 0, ...] for the full rollout
             [0,-1, 0, ...] for the full rollout
             ...

       - every basis test satisfies ||w_k||_2 = 1 exactly.

The exact disturbance matrix E(x,k) is rebuilt from
quadruped_barrel_roll.make_phase_scaled_disturbance(), i.e. the same callback
used by the robust planner. The actual injected state perturbation is

    d_k = E(x_k,k) @ w_k

and the closed-loop controller is

    dx_j = x_j - X[j]
    du_k = sum_{j=0}^k K_x[k,j] dx_j
    u_k  = U[k] + du_k.

The optimized phase-duration disturbance columns of E are zero, so basis
rollouts along those coordinates have no physical injection; they are retained
so the basis sweep spans the complete saved disturbance vector.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Hard-coded experiment paths
# -----------------------------------------------------------------------------

EXPERIMENT_DIR = Path(
    "/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/"
    "gpu_sls/examples/quadruped_barrel_roll_obstacle_robust_new"
)

INPUT_NPZ = EXPERIMENT_DIR / "quadruped_barrel_roll_obstacle_min_time.npz"

OUTPUT_NPZ = (
    EXPERIMENT_DIR
    / "disturbance_rollouts.npz"
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
# Experiment settings
# -----------------------------------------------------------------------------

NUM_RANDOM_ROLLOUTS = 256
NUM_ADVERSARIAL_ROLLOUTS = 256
# BASIS rollouts are determined after loading nx: 2 * nx (+e_i and -e_i).
RNG_SEED = 0

# Robust SLS disturbance set:
#
#     ||w_k||_2 <= DISTURBANCE_RADIUS
#
DISTURBANCE_RADIUS = 1.0

# Random rollouts:
# False -> uniform-by-volume in the allowed L2 ball.
# True  -> every random disturbance lies on the L2 boundary.
RANDOM_SAMPLE_ON_BOUNDARY = True

# Adversarial sequence optimization.
ADVERSARIAL_PGD_STEPS = 40
ADVERSARIAL_STEP_SIZE = 0.25

# The adversarial objective uses a smooth max over tube-normalized errors.
# Lower values approximate max() more tightly but can make gradients sharper.
ADVERSARIAL_SOFTMAX_TEMPERATURE = 0.05

# Ignore nominally zero tube entries in the adversarial objective/diagnostics.
TUBE_EPS = 1.0e-10
ADVERSARIAL_TUBE_FLOOR = 1.0e-8
ADVERSARIAL_ABS_EPS = 1.0e-12

# Leave False to expose feedback actions that exceed torque limits rather than
# silently saturating them.
CLIP_CONTROLS = False

# False exactly preserves the additive Euclidean disturbance model used by SLS.
RENORMALIZE_QUATERNION = False

# Printing all random/adversarial rollouts is noisy. Set True if you want every row.
PRINT_PER_ROLLOUT = False


# -----------------------------------------------------------------------------
# Fixed disturbance families
# -----------------------------------------------------------------------------

# State layout from quadruped_barrel_roll.py:
#
#   [0:3]                       base position
#   [3:7]                       quaternion
#   [7:7+nj]                    joint positions
#   [7+nj:10+nj]                base linear velocity vx,vy,vz
#   [10+nj:13+nj]               base angular velocity wx,wy,wz
#   [13+nj:13+2*nj]             joint velocities
#   [13+2*nj:13+2*nj+3*nc]      foot positions
#   [GRF_START:GRF_STOP]         GRFs
#   [DURATION_SLICE]             optimized phase durations
#
# The planner disturbance callback is diagonal, so selecting one w coordinate
# selects the corresponding physical disturbance channel.

NJ = barrel.config.n_joints

LINEAR_VEL_START = 7 + NJ
ANGULAR_VEL_START = 10 + NJ
JOINT_VEL_START = 13 + NJ

VX = LINEAR_VEL_START + 0
VY = LINEAR_VEL_START + 1
VZ = LINEAR_VEL_START + 2
WX = ANGULAR_VEL_START + 0
WY = ANGULAR_VEL_START + 1
WZ = ANGULAR_VEL_START + 2

DISTURBANCE_MODES = (
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "horizontal_velocity",
    "angular_velocity",
    "joint_velocity",
    "grf",
    "all_active",
)


def make_disturbance_mode_masks(
    num_rollouts: int,
    disturbance_dim: int,
):
    """
    Assign one fixed disturbance support to each rollout for the full horizon.

    The assignment cycles through DISTURBANCE_MODES, so the random and
    adversarial batches use the same mode pattern.

    Returns
    -------
    masks : (R, nw)
        0/1 mask in disturbance coordinates.
    mode_ids : (R,)
        Integer index into DISTURBANCE_MODES.
    """

    masks = np.zeros(
        (num_rollouts, disturbance_dim),
        dtype=np.float32,
    )

    mode_ids = (
        np.arange(num_rollouts, dtype=np.int32)
        % len(DISTURBANCE_MODES)
    )

    for r in range(num_rollouts):
        mode = DISTURBANCE_MODES[int(mode_ids[r])]

        if mode == "vx":
            masks[r, VX] = 1.0

        elif mode == "vy":
            masks[r, VY] = 1.0

        elif mode == "vz":
            masks[r, VZ] = 1.0

        elif mode == "wx":
            masks[r, WX] = 1.0

        elif mode == "wy":
            masks[r, WY] = 1.0

        elif mode == "wz":
            masks[r, WZ] = 1.0

        elif mode == "horizontal_velocity":
            masks[r, VX] = 1.0
            masks[r, VY] = 1.0

        elif mode == "angular_velocity":
            masks[r, WX:WZ + 1] = 1.0

        elif mode == "joint_velocity":
            masks[
                r,
                JOINT_VEL_START:JOINT_VEL_START + NJ,
            ] = 1.0

        elif mode == "grf":
            masks[
                r,
                barrel.GRF_START:barrel.GRF_STOP,
            ] = 1.0

        elif mode == "all_active":
            # Only the physical state subspace. Duration disturbance channels
            # are zero by construction in the planner.
            masks[r, :barrel.PHYSICAL_N] = 1.0

        else:
            raise ValueError(f"Unknown disturbance mode: {mode}")

    return jnp.asarray(masks), mode_ids


# -----------------------------------------------------------------------------
# Load saved robust solution
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

        missing = [key for key in required if key not in f.files]

        if missing:
            raise KeyError(
                f"Missing keys {missing}; available={f.files}"
            )

        saved = {key: np.asarray(f[key]) for key in f.files}

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

    # Align tube rows with x_0,...,x_N.
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
# Rebuild exact nonlinear plant and exact planner disturbance E(x,k)
# -----------------------------------------------------------------------------


def build_model():
    """Rebuild the same nonlinear dynamics and disturbance callback as planner."""

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

    # This is the exact function from the planner file. With the current
    # quadruped_barrel_roll.py defaults it uses:
    #   position             0.02 * dt
    #   quaternion           0.01 * dt
    #   joint position       0.02 * dt
    #   linear velocity      0.05 * dt
    #   angular velocity     0.05 * dt
    #   joint velocity       0.05 * dt
    #   foot position        0.02 * dt
    #   GRF                  5.00 * dt
    #   duration             0
    disturbance_model = barrel.make_phase_scaled_disturbance(
        barrel.config.n,
        position_magnitude=0.05,
        quaternion_magnitude=0.01,
        joint_position_magnitude=0.02,
        linear_velocity_magnitude=0.1,
        angular_velocity_magnitude=0.1,
        joint_velocity_magnitude=0.05,
        foot_position_magnitude=0.02,
        grf_magnitude=5.0,
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
# Disturbance-sequence generation / projection
# -----------------------------------------------------------------------------


def generate_gaussian_sources(
    key: jax.Array,
    num_rollouts: int,
    horizon: int,
    disturbance_dim: int,
):
    return jax.random.normal(
        key,
        shape=(num_rollouts, horizon, disturbance_dim),
    )


def generate_uniform_radii(
    key: jax.Array,
    num_rollouts: int,
    horizon: int,
):
    return jax.random.uniform(
        key,
        shape=(num_rollouts, horizon),
        minval=0.0,
        maxval=1.0,
    )


def masked_unit_directions(
    raw: jnp.ndarray,
    rollout_masks: jnp.ndarray,
):
    """Normalize raw directions inside each rollout's fixed mode subspace."""

    masked = raw * rollout_masks[:, None, :]

    norm = jnp.linalg.norm(
        masked,
        axis=-1,
        keepdims=True,
    )

    return masked / jnp.maximum(norm, 1.0e-12)


def random_w_sequences(
    raw_directions: jnp.ndarray,
    radial_uniform: jnp.ndarray,
    rollout_masks: jnp.ndarray,
    *,
    on_boundary: bool,
):
    """Generate random w sequences inside each rollout's allowed L2 subspace."""

    directions = masked_unit_directions(
        raw_directions,
        rollout_masks,
    )

    active_dim = jnp.sum(
        rollout_masks > 0.0,
        axis=1,
    )
    active_dim = jnp.maximum(active_dim, 1)

    if on_boundary:
        radius = jnp.full(
            radial_uniform.shape,
            DISTURBANCE_RADIUS,
            dtype=raw_directions.dtype,
        )
    else:
        # Uniform-by-volume in a d-dimensional L2 ball:
        #
        #     r = R * U^(1/d)
        radius = (
            DISTURBANCE_RADIUS
            * radial_uniform
            ** (
                1.0
                / active_dim[:, None].astype(raw_directions.dtype)
            )
        )

    return directions * radius[..., None]


def project_w_to_ball(
    W: jnp.ndarray,
    rollout_masks: jnp.ndarray,
):
    """
    Project every w_{r,k} onto the rollout's fixed support and L2 ball.

        ||w_{r,k}||_2 <= DISTURBANCE_RADIUS
    """

    W = W * rollout_masks[:, None, :]

    norm = jnp.linalg.norm(
        W,
        axis=-1,
        keepdims=True,
    )

    scale = jnp.minimum(
        1.0,
        DISTURBANCE_RADIUS / jnp.maximum(norm, 1.0e-12),
    )

    return W * scale


def constant_basis_w_sequences(
    horizon: int,
    disturbance_dim: int,
    dtype,
):
    """
    Build deterministic coordinate-basis stress tests.

    For each disturbance coordinate i, create two complete-horizon rollouts:

        +e_i:  w_k[i] = +1 for every k; all other coordinates are zero.
        -e_i:  w_k[i] = -1 for every k; all other coordinates are zero.

    Ordering is

        +e_0, -e_0, +e_1, -e_1, ..., +e_{d-1}, -e_{d-1}.

    Every w_k has L2 norm exactly one, so these tests are inside the original
    ||w_k||_2 <= 1 disturbance set (for DISTURBANCE_RADIUS == 1).
    """

    eye = jnp.eye(disturbance_dim, dtype=dtype)

    # (d, 2, d): coordinate i contains +e_i then -e_i.
    signed_basis = jnp.stack([eye, -eye], axis=1)

    # (2*d, d), ordered +e_0,-e_0,+e_1,-e_1,...
    signed_basis = signed_basis.reshape(2 * disturbance_dim, disturbance_dim)

    # Hold the same basis vector for the entire horizon.
    W_basis = jnp.broadcast_to(
        signed_basis[:, None, :],
        (2 * disturbance_dim, horizon, disturbance_dim),
    )

    coordinate_indices = np.repeat(
        np.arange(disturbance_dim, dtype=np.int32),
        2,
    )
    signs = np.tile(
        np.asarray([1.0, -1.0], dtype=np.float32),
        disturbance_dim,
    )
    labels = np.asarray([
        f"w_{i:02d}_{'plus' if sign > 0 else 'minus'}"
        for i, sign in zip(coordinate_indices, signs)
    ])

    return W_basis, coordinate_indices, signs, labels


# -----------------------------------------------------------------------------
# Closed-loop rollout for an explicitly supplied disturbance sequence W
# -----------------------------------------------------------------------------


def make_rollout_from_w(
    dynamics,
    disturbance_model,
    parameter,
):
    """Return a pure JAX rollout function suitable for jit and differentiation."""

    batched_dynamics = jax.vmap(
        dynamics,
        in_axes=(0, 0, None, None),
        out_axes=0,
    )

    batched_disturbance = jax.vmap(
        disturbance_model.at_state,
        in_axes=(0, None),
        out_axes=0,
    )

    def rollout_from_w(
        X_nominal,
        U_nominal,
        K,
        W_sequences,
    ):
        """
        Shapes
        ------
        X_nominal   : (N+1, nx)
        U_nominal   : (N, nu)
        K           : (N, N, nu, physical_n)
        W_sequences : (R, N, nx)
        """

        R = W_sequences.shape[0]
        N = U_nominal.shape[0]
        nx = X_nominal.shape[1]

        x0_batch = jnp.broadcast_to(
            X_nominal[0],
            (R, nx),
        )

        error_history0 = jnp.zeros(
            (R, N + 1, nx),
            dtype=X_nominal.dtype,
        )

        error_history0 = error_history0.at[:, 0, :].set(
            x0_batch - X_nominal[0]
        )

        def scan_step(carry, k):
            x_batch, error_history = carry

            # Causal state-history feedback:
            #
            #   du_k = sum_{j=0}^k K[k,j] dx_j
            K_k = K[k]  # (N, nu, physical_n)

            history_mask = (
                jnp.arange(N) <= k
            ).astype(X_nominal.dtype)

            K_k_causal = (
                K_k
                * history_mask[:, None, None]
            )

            du_batch = jnp.einsum(
                "rjp,jmp->rm",
                error_history[:, :N, :barrel.PHYSICAL_N],
                K_k_causal,
            )

            u_batch = U_nominal[k][None, :] + du_batch

            if CLIP_CONTROLS:
                u_batch = jnp.clip(
                    u_batch,
                    jnp.asarray(barrel.config.min_torque),
                    jnp.asarray(barrel.config.max_torque),
                )

            # Exact nonlinear plant without external disturbance.
            x_det_next = batched_dynamics(
                x_batch,
                u_batch,
                k,
                parameter,
            )

            # Exact E(x,k) used by the planner.
            E_batch = batched_disturbance(
                x_batch,
                k,
            )

            w_batch = W_sequences[:, k, :]

            disturbance_batch = jnp.einsum(
                "rij,rj->ri",
                E_batch,
                w_batch,
            )

            x_next = x_det_next + disturbance_batch

            if RENORMALIZE_QUATERNION:
                q = x_next[:, 3:7]
                q_norm = jnp.linalg.norm(
                    q,
                    axis=1,
                    keepdims=True,
                )
                x_next = x_next.at[:, 3:7].set(
                    q / jnp.maximum(q_norm, 1.0e-12)
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
                du_batch,
                w_batch,
                disturbance_batch,
            )

            return (
                x_next,
                error_history,
            ), outputs

        # Rematerialize scan-step intermediates during reverse-mode PGD.
        # This is substantially more memory-friendly than retaining every MJX
        # intermediate for all 256 adversarial rollouts across the horizon.
        scan_step_for_grad = jax.checkpoint(scan_step)

        (_, error_history_final), outputs = jax.lax.scan(
            scan_step_for_grad,
            (x0_batch, error_history0),
            jnp.arange(N, dtype=jnp.int32),
        )

        (
            X_next_time_major,
            U_time_major,
            dU_time_major,
            W_time_major,
            D_time_major,
        ) = outputs

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

    return rollout_from_w


# -----------------------------------------------------------------------------
# Adversarial objective + projected-gradient ascent
# -----------------------------------------------------------------------------


def adversarial_scores_from_error(
    error_history: jnp.ndarray,
    tubes: jnp.ndarray,
):
    """
    Smooth approximation of max |dx|/tube for each rollout separately.

    The score is only used to optimize W. Final reporting uses the exact max.
    """

    err = error_history[:, :, :barrel.PHYSICAL_N]
    tube = tubes[:, :barrel.PHYSICAL_N]

    valid = tube > TUBE_EPS

    smooth_abs = jnp.sqrt(
        err * err + ADVERSARIAL_ABS_EPS
    )

    ratio = smooth_abs / jnp.maximum(
        tube[None, :, :],
        ADVERSARIAL_TUBE_FLOOR,
    )

    # Invalid zero-tube entries should never participate in the max.
    masked_ratio = jnp.where(
        valid[None, :, :],
        ratio,
        -1.0e6,
    )

    flat = masked_ratio.reshape(
        masked_ratio.shape[0],
        -1,
    )

    tau = jnp.asarray(
        ADVERSARIAL_SOFTMAX_TEMPERATURE,
        dtype=error_history.dtype,
    )

    return tau * jax.scipy.special.logsumexp(
        flat / tau,
        axis=1,
    )


def make_adversarial_optimizer(
    rollout_from_w,
    X_nominal,
    U_nominal,
    K,
    tubes,
    rollout_masks,
):
    """
    Build one jitted PGD optimizer over all adversarial disturbance sequences.

    Each rollout is independent. Summing the per-rollout objective therefore
    gives exactly the collection of per-rollout gradients while allowing one
    batched reverse-mode pass.
    """

    def objective_sum(W_sequences):
        outputs = rollout_from_w(
            X_nominal,
            U_nominal,
            K,
            W_sequences,
        )

        error_history = outputs[2]

        scores = adversarial_scores_from_error(
            error_history,
            tubes,
        )

        return jnp.sum(scores)

    objective_grad = jax.grad(objective_sum)

    @jax.jit
    def optimize(W_initial):
        def body(_, W_sequences):
            grad = objective_grad(W_sequences)

            # Never allow gradients to create components outside the assigned
            # disturbance family.
            grad = grad * rollout_masks[:, None, :]

            # Normalize each (rollout,time) gradient before stepping. This
            # keeps the PGD step size interpretable across very different
            # physical state scalings (e.g. velocity versus GRF channels).
            grad_norm = jnp.linalg.norm(
                grad,
                axis=-1,
                keepdims=True,
            )

            unit_grad = grad / jnp.maximum(
                grad_norm,
                1.0e-12,
            )

            W_next = (
                W_sequences
                + ADVERSARIAL_STEP_SIZE * unit_grad
            )

            return project_w_to_ball(
                W_next,
                rollout_masks,
            )

        W_final = jax.lax.fori_loop(
            0,
            ADVERSARIAL_PGD_STEPS,
            body,
            W_initial,
        )

        return W_final

    return optimize


# -----------------------------------------------------------------------------
# Exact diagnostics
# -----------------------------------------------------------------------------


def compute_diagnostics(
    U_all,
    error_history,
    tubes,
):
    """Vectorized exact diagnostics for a rollout batch."""

    physical_error = jnp.abs(
        error_history[:, :, :barrel.PHYSICAL_N]
    )

    tube = jnp.asarray(
        tubes[:, :barrel.PHYSICAL_N]
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

    safe_ratio = jnp.where(
        jnp.isnan(ratio),
        -jnp.inf,
        ratio,
    )

    tube_ratio_max = jnp.max(
        safe_ratio,
        axis=(1, 2),
    )

    flat_idx = jnp.argmax(
        safe_ratio.reshape(
            safe_ratio.shape[0],
            -1,
        ),
        axis=1,
    )

    num_physical_states = barrel.PHYSICAL_N

    worst_nodes = flat_idx // num_physical_states
    worst_states = flat_idx % num_physical_states

    max_torque = jnp.asarray(barrel.config.max_torque)
    min_torque = jnp.asarray(barrel.config.min_torque)

    upper_violation = U_all - max_torque
    lower_violation = min_torque - U_all

    torque_violation = jnp.maximum(
        0.0,
        jnp.maximum(
            jnp.max(upper_violation, axis=(1, 2)),
            jnp.max(lower_violation, axis=(1, 2)),
        ),
    )

    return (
        state_err_max,
        tube_ratio_max,
        torque_violation,
        worst_nodes,
        worst_states,
    )


def to_numpy_outputs(outputs):
    return tuple(np.asarray(x) for x in outputs)


def to_numpy_diagnostics(diag):
    return tuple(np.asarray(x) for x in diag)


def choose_worst_index(
    tube_ratio_max: np.ndarray,
    state_err_max: np.ndarray,
):
    if np.any(np.isfinite(tube_ratio_max)):
        return int(np.nanargmax(tube_ratio_max))
    return int(np.argmax(state_err_max))


def print_batch_summary(
    name: str,
    diag,
):
    (
        state_err_max,
        tube_ratio_max,
        torque_viol,
        worst_nodes,
        worst_states,
    ) = diag

    worst_idx = choose_worst_index(
        tube_ratio_max,
        state_err_max,
    )

    print()
    print(name)
    print("-" * len(name))
    print(f"Worst rollout:            {worst_idx}")
    print(
        "Worst max |dx|/tube:      "
        f"{tube_ratio_max[worst_idx]:.6f}"
    )
    print(
        "Worst node/state:         "
        f"{worst_nodes[worst_idx]} / {worst_states[worst_idx]}"
    )
    print(
        "Largest max |dx|:         "
        f"{np.max(state_err_max):.6e}"
    )
    print(
        "Largest torque violation: "
        f"{np.max(torque_viol):.6e}"
    )

    return worst_idx


def print_mode_summary(
    label: str,
    mode_ids: np.ndarray,
    diag,
):
    state_err_max, tube_ratio_max, torque_viol, _, _ = diag

    print()
    print(f"{label} by disturbance mode")
    print("-" * (len(label) + 20))

    for mode_id, mode_name in enumerate(DISTURBANCE_MODES):
        idx = np.flatnonzero(mode_ids == mode_id)
        if idx.size == 0:
            continue

        ratios = tube_ratio_max[idx]
        errors = state_err_max[idx]
        torques = torque_viol[idx]

        print(
            f"{mode_name:<20s} "
            f"n={idx.size:3d}  "
            f"mean_ratio={np.nanmean(ratios):9.4f}  "
            f"max_ratio={np.nanmax(ratios):9.4f}  "
            f"max|dx|={np.max(errors):.4e}  "
            f"max_tau_viol={np.max(torques):.4e}"
        )


def print_per_rollout(
    label: str,
    mode_ids: np.ndarray,
    diag,
):
    if not PRINT_PER_ROLLOUT:
        return

    state_err_max, tube_ratio_max, torque_viol, _, _ = diag

    print()
    print(f"{label} per-rollout diagnostics")
    print("-" * (len(label) + 24))

    for r in range(len(mode_ids)):
        mode_name = DISTURBANCE_MODES[int(mode_ids[r])]
        print(
            f"[{r:03d}] "
            f"mode={mode_name:<20s}  "
            f"max|dx|={state_err_max[r]:.4e}  "
            f"max(|dx|/tube)={tube_ratio_max[r]:.4f}  "
            f"torque_violation={torque_viol[r]:.4e}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    saved, X_np, U_np, K_np, tubes_np = load_solution()

    mpc, parameter, disturbance_model = build_model()

    X = jnp.asarray(X_np)
    U = jnp.asarray(U_np)
    K = jnp.asarray(K_np)
    tubes = jnp.asarray(tubes_np)

    N = U.shape[0]
    nx = X.shape[1]

    if NUM_RANDOM_ROLLOUTS != NUM_ADVERSARIAL_ROLLOUTS:
        print(
            "Warning: random/adversarial batch sizes differ, so their mode "
            "assignments will not be one-to-one."
        )

    NUM_BASIS_ROLLOUTS = 2 * nx
    total_rollouts = (
        NUM_RANDOM_ROLLOUTS
        + NUM_ADVERSARIAL_ROLLOUTS
        + NUM_BASIS_ROLLOUTS
    )

    print(f"Input:                  {INPUT_NPZ}")
    print(f"X/U/K:                  {X.shape} / {U.shape} / {K.shape}")
    print(f"Random rollouts:        {NUM_RANDOM_ROLLOUTS}")
    print(f"Adversarial rollouts:   {NUM_ADVERSARIAL_ROLLOUTS}")
    print(f"Basis +/-1 rollouts:    {NUM_BASIS_ROLLOUTS} (= 2 * nx)")
    print(f"Total final rollouts:   {total_rollouts}")
    print(f"RNG seed:               {RNG_SEED}")
    print(f"||w_k|| radius:         {DISTURBANCE_RADIUS}")
    print(f"Random on boundary:     {RANDOM_SAMPLE_ON_BOUNDARY}")
    print(f"Adversarial PGD steps:  {ADVERSARIAL_PGD_STEPS}")
    print(f"Adversarial step size:  {ADVERSARIAL_STEP_SIZE}")
    print(f"Disturbance modes:      {', '.join(DISTURBANCE_MODES)}")
    print(f"JAX devices:            {jax.devices()}")
    print(
        "Basis +/-1 tests:        each rollout has ||w_k||_2 = 1 "
        "for every timestep"
    )

    # ------------------------------------------------------------------
    # Matched random/adversarial mode assignments
    # ------------------------------------------------------------------

    random_masks, random_mode_ids = make_disturbance_mode_masks(
        NUM_RANDOM_ROLLOUTS,
        nx,
    )

    adversarial_masks, adversarial_mode_ids = make_disturbance_mode_masks(
        NUM_ADVERSARIAL_ROLLOUTS,
        nx,
    )

    # ------------------------------------------------------------------
    # Random disturbance sequences
    # ------------------------------------------------------------------

    master_key = jax.random.PRNGKey(RNG_SEED)
    (
        key_random_direction,
        key_random_radius,
        key_adversarial_init_direction,
        key_adversarial_init_radius,
    ) = jax.random.split(master_key, 4)

    random_raw = generate_gaussian_sources(
        key_random_direction,
        NUM_RANDOM_ROLLOUTS,
        N,
        nx,
    )

    random_radius_source = generate_uniform_radii(
        key_random_radius,
        NUM_RANDOM_ROLLOUTS,
        N,
    )

    W_random = random_w_sequences(
        random_raw,
        random_radius_source,
        random_masks,
        on_boundary=RANDOM_SAMPLE_ON_BOUNDARY,
    )

    # ------------------------------------------------------------------
    # Adversarial initial disturbance sequences
    # ------------------------------------------------------------------

    adversarial_raw = generate_gaussian_sources(
        key_adversarial_init_direction,
        NUM_ADVERSARIAL_ROLLOUTS,
        N,
        nx,
    )

    adversarial_radius_source = generate_uniform_radii(
        key_adversarial_init_radius,
        NUM_ADVERSARIAL_ROLLOUTS,
        N,
    )

    # Start every adversarial w_k on the maximum-norm boundary, but PGD is
    # allowed to move inside the <= R ball if that improves the trajectory-level
    # objective.
    W_adversarial_initial = random_w_sequences(
        adversarial_raw,
        adversarial_radius_source,
        adversarial_masks,
        on_boundary=True,
    )

    # ------------------------------------------------------------------
    # Deterministic coordinate-basis +/-1 disturbance sequences
    # ------------------------------------------------------------------

    # Shape: (2*nx, N, nx)
    # Ordering:
    #   0 -> +e_0 for every timestep
    #   1 -> -e_0 for every timestep
    #   2 -> +e_1 for every timestep
    #   3 -> -e_1 for every timestep
    #   ...
    W_basis, basis_coordinate_indices, basis_signs, basis_labels = (
        constant_basis_w_sequences(
            N,
            nx,
            X.dtype,
        )
    )

    try:
        basis_coordinate_names = np.asarray(barrel.state_dimension_names())
        if basis_coordinate_names.shape[0] != nx:
            raise ValueError
    except Exception:
        basis_coordinate_names = np.asarray(
            [f"state_{i}" for i in range(nx)]
        )

    # ------------------------------------------------------------------
    # Rollout function
    # ------------------------------------------------------------------

    rollout_from_w = make_rollout_from_w(
        mpc.dynamics,
        disturbance_model,
        parameter,
    )

    rollout_jit = jax.jit(rollout_from_w)

    # ------------------------------------------------------------------
    # Run the 256 random rollouts
    # ------------------------------------------------------------------

    print()
    print(f"Compiling/executing {NUM_RANDOM_ROLLOUTS} random rollouts...")

    random_outputs = rollout_jit(
        X,
        U,
        K,
        W_random,
    )

    random_outputs[0].block_until_ready()

    print("Random rollout batch complete.")

    random_diag_jax = compute_diagnostics(
        random_outputs[1],
        random_outputs[2],
        tubes,
    )

    # ------------------------------------------------------------------
    # Optimize 256 adversarial disturbance sequences in parallel
    # ------------------------------------------------------------------

    # Record initial adversarial scores for diagnostics.
    adversarial_initial_outputs = rollout_jit(
        X,
        U,
        K,
        W_adversarial_initial,
    )

    adversarial_initial_scores = adversarial_scores_from_error(
        adversarial_initial_outputs[2],
        tubes,
    )

    print()
    print(
        "Compiling/executing adversarial PGD for "
        f"{NUM_ADVERSARIAL_ROLLOUTS} rollouts..."
    )

    optimize_adversarial = make_adversarial_optimizer(
        rollout_from_w,
        X,
        U,
        K,
        tubes,
        adversarial_masks,
    )

    W_adversarial = optimize_adversarial(
        W_adversarial_initial
    )

    W_adversarial.block_until_ready()

    print("Adversarial disturbance optimization complete.")

    adversarial_outputs = rollout_jit(
        X,
        U,
        K,
        W_adversarial,
    )

    adversarial_outputs[0].block_until_ready()

    adversarial_final_scores = adversarial_scores_from_error(
        adversarial_outputs[2],
        tubes,
    )

    print("Adversarial rollout batch complete.")

    adversarial_diag_jax = compute_diagnostics(
        adversarial_outputs[1],
        adversarial_outputs[2],
        tubes,
    )

    # ------------------------------------------------------------------
    # Run the deterministic coordinate-basis +/-1 sweep
    # ------------------------------------------------------------------

    print()
    print(
        f"Executing {NUM_BASIS_ROLLOUTS} constant coordinate-basis +/-1 "
        "rollouts..."
    )

    basis_outputs = rollout_jit(
        X,
        U,
        K,
        W_basis,
    )

    basis_outputs[0].block_until_ready()

    basis_diag_jax = compute_diagnostics(
        basis_outputs[1],
        basis_outputs[2],
        tubes,
    )

    print("Coordinate-basis rollout batch complete.")

    # ------------------------------------------------------------------
    # Device -> host
    # ------------------------------------------------------------------

    (
        X_random,
        U_random,
        Ehist_random,
        dU_random,
        W_random_host,
        D_random,
    ) = to_numpy_outputs(random_outputs)

    (
        X_adversarial,
        U_adversarial,
        Ehist_adversarial,
        dU_adversarial,
        W_adversarial_host,
        D_adversarial,
    ) = to_numpy_outputs(adversarial_outputs)

    (
        X_basis,
        U_basis,
        Ehist_basis,
        dU_basis,
        W_basis_host,
        D_basis,
    ) = to_numpy_outputs(basis_outputs)

    random_diag = to_numpy_diagnostics(random_diag_jax)
    adversarial_diag = to_numpy_diagnostics(adversarial_diag_jax)
    basis_diag = to_numpy_diagnostics(basis_diag_jax)

    W_adversarial_initial_host = np.asarray(
        W_adversarial_initial
    )

    adversarial_initial_scores_host = np.asarray(
        adversarial_initial_scores
    )

    adversarial_final_scores_host = np.asarray(
        adversarial_final_scores
    )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    worst_random_idx = print_batch_summary(
        "Random batch summary",
        random_diag,
    )

    worst_adversarial_idx = print_batch_summary(
        "Adversarial batch summary",
        adversarial_diag,
    )

    worst_basis_idx = print_batch_summary(
        "Coordinate-basis +/-1 batch summary",
        basis_diag,
    )

    print_mode_summary(
        "Random",
        random_mode_ids,
        random_diag,
    )

    print_mode_summary(
        "Adversarial",
        adversarial_mode_ids,
        adversarial_diag,
    )

    print_per_rollout(
        "Random",
        random_mode_ids,
        random_diag,
    )

    print_per_rollout(
        "Adversarial",
        adversarial_mode_ids,
        adversarial_diag,
    )

    print()
    print("Coordinate-basis +/-1 diagnostics")
    print("----------------------------------")
    (
        basis_state_err,
        basis_tube_ratio,
        basis_torque,
        basis_nodes,
        basis_states,
    ) = basis_diag

    for r in range(NUM_BASIS_ROLLOUTS):
        coord = int(basis_coordinate_indices[r])
        sign = int(basis_signs[r])
        coord_name = str(basis_coordinate_names[coord])
        print(
            f"[{r:03d}] w={sign:+d} e_{coord:<2d} "
            f"({coord_name:<18s})  "
            f"max|dx|={basis_state_err[r]:.4e}  "
            f"max(|dx|/tube)={basis_tube_ratio[r]:.6f}  "
            f"worst_node/state={basis_nodes[r]}/{basis_states[r]}  "
            f"torque_violation={basis_torque[r]:.4e}"
        )
    print()
    print("Adversarial PGD objective")
    print("-------------------------")
    print(
        "Mean initial smooth score: "
        f"{np.mean(adversarial_initial_scores_host):.6f}"
    )
    print(
        "Mean final smooth score:   "
        f"{np.mean(adversarial_final_scores_host):.6f}"
    )
    print(
        "Max initial smooth score:  "
        f"{np.max(adversarial_initial_scores_host):.6f}"
    )
    print(
        "Max final smooth score:    "
        f"{np.max(adversarial_final_scores_host):.6f}"
    )

    # ------------------------------------------------------------------
    # Pick overall worst rollout across random, adversarial, and basis +/-1
    # ------------------------------------------------------------------

    (
        random_state_err,
        random_tube_ratio,
        random_torque,
        random_nodes,
        random_states,
    ) = random_diag

    (
        adv_state_err,
        adv_tube_ratio,
        adv_torque,
        adv_nodes,
        adv_states,
    ) = adversarial_diag

    candidate_groups = [
        (
            "random",
            worst_random_idx,
            random_tube_ratio[worst_random_idx],
            random_state_err[worst_random_idx],
            X_random[worst_random_idx],
            U_random[worst_random_idx],
            Ehist_random[worst_random_idx],
            dU_random[worst_random_idx],
            W_random_host[worst_random_idx],
            D_random[worst_random_idx],
        ),
        (
            "adversarial",
            worst_adversarial_idx,
            adv_tube_ratio[worst_adversarial_idx],
            adv_state_err[worst_adversarial_idx],
            X_adversarial[worst_adversarial_idx],
            U_adversarial[worst_adversarial_idx],
            Ehist_adversarial[worst_adversarial_idx],
            dU_adversarial[worst_adversarial_idx],
            W_adversarial_host[worst_adversarial_idx],
            D_adversarial[worst_adversarial_idx],
        ),
        (
            "basis_pm1",
            worst_basis_idx,
            basis_tube_ratio[worst_basis_idx],
            basis_state_err[worst_basis_idx],
            X_basis[worst_basis_idx],
            U_basis[worst_basis_idx],
            Ehist_basis[worst_basis_idx],
            dU_basis[worst_basis_idx],
            W_basis_host[worst_basis_idx],
            D_basis[worst_basis_idx],
        ),
    ]

    def candidate_score(candidate):
        tube_ratio = candidate[2]
        state_err = candidate[3]
        if np.isfinite(tube_ratio):
            return float(tube_ratio)
        return float(state_err)

    overall = max(candidate_groups, key=candidate_score)
    (
        overall_type,
        overall_idx,
        _,
        _,
        X_overall,
        U_overall,
        Ehist_overall,
        dU_overall,
        W_overall,
        D_overall,
    ) = overall

    print()
    print(
        f"Overall worst rollout: {overall_type} [{overall_idx}]"
    )

    # ------------------------------------------------------------------
    # Preserve useful planner metadata
    # ------------------------------------------------------------------

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
            metadata[key_name] = saved[key_name]

    OUTPUT_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------

    np.savez(
        OUTPUT_NPZ,

        # Saved robust plan / controller.
        X=X_np,
        U=U_np,
        K_x_history=K_np,
        trajectory_tubes=tubes_np,

        # -----------------------------------------------------------------
        # 256 random rollouts
        # -----------------------------------------------------------------
        X_random_feedback_rollouts=X_random,
        U_random_feedback_rollouts=U_random,
        x_perturbation_histories_random=Ehist_random,
        feedback_corrections_random=dU_random,
        W_random=W_random_host,
        disturbance_injections_random=D_random,
        random_disturbance_mode_ids=random_mode_ids,

        max_state_errors_random=random_state_err,
        max_tube_ratios_random=random_tube_ratio,
        max_torque_violations_random=random_torque,
        worst_tube_nodes_random=random_nodes,
        worst_tube_states_random=random_states,
        worst_random_rollout_index=np.asarray(worst_random_idx),

        X_worst_random_feedback_rollout=X_random[worst_random_idx],
        U_worst_random_feedback_rollout=U_random[worst_random_idx],
        W_worst_random=W_random_host[worst_random_idx],
        disturbance_injection_worst_random=D_random[worst_random_idx],

        # -----------------------------------------------------------------
        # 256 adversarial rollouts
        # -----------------------------------------------------------------
        X_adversarial_feedback_rollouts=X_adversarial,
        U_adversarial_feedback_rollouts=U_adversarial,
        x_perturbation_histories_adversarial=Ehist_adversarial,
        feedback_corrections_adversarial=dU_adversarial,
        W_adversarial=W_adversarial_host,
        W_adversarial_initial=W_adversarial_initial_host,
        disturbance_injections_adversarial=D_adversarial,
        adversarial_disturbance_mode_ids=adversarial_mode_ids,

        max_state_errors_adversarial=adv_state_err,
        max_tube_ratios_adversarial=adv_tube_ratio,
        max_torque_violations_adversarial=adv_torque,
        worst_tube_nodes_adversarial=adv_nodes,
        worst_tube_states_adversarial=adv_states,
        worst_adversarial_rollout_index=np.asarray(
            worst_adversarial_idx
        ),

        adversarial_initial_smooth_scores=(
            adversarial_initial_scores_host
        ),
        adversarial_final_smooth_scores=(
            adversarial_final_scores_host
        ),

        X_worst_adversarial_feedback_rollout=(
            X_adversarial[worst_adversarial_idx]
        ),
        U_worst_adversarial_feedback_rollout=(
            U_adversarial[worst_adversarial_idx]
        ),
        W_worst_adversarial=(
            W_adversarial_host[worst_adversarial_idx]
        ),
        disturbance_injection_worst_adversarial=(
            D_adversarial[worst_adversarial_idx]
        ),

        # -----------------------------------------------------------------
        # Constant coordinate-basis +e_i / -e_i rollouts
        # -----------------------------------------------------------------
        basis_rollout_labels=basis_labels,
        basis_coordinate_indices=basis_coordinate_indices,
        basis_signs=basis_signs,
        basis_coordinate_names=basis_coordinate_names,

        X_basis_pm1_feedback_rollouts=X_basis,
        U_basis_pm1_feedback_rollouts=U_basis,
        x_perturbation_histories_basis_pm1=Ehist_basis,
        feedback_corrections_basis_pm1=dU_basis,
        W_basis_pm1=W_basis_host,
        disturbance_injections_basis_pm1=D_basis,

        max_state_errors_basis_pm1=basis_state_err,
        max_tube_ratios_basis_pm1=basis_tube_ratio,
        max_torque_violations_basis_pm1=basis_torque,
        worst_tube_nodes_basis_pm1=basis_nodes,
        worst_tube_states_basis_pm1=basis_states,
        worst_basis_rollout_index=np.asarray(worst_basis_idx),

        X_worst_basis_pm1_feedback_rollout=X_basis[worst_basis_idx],
        U_worst_basis_pm1_feedback_rollout=U_basis[worst_basis_idx],
        W_worst_basis_pm1=W_basis_host[worst_basis_idx],
        disturbance_injection_worst_basis_pm1=D_basis[worst_basis_idx],

        basis_w_l2_norm=np.asarray(DISTURBANCE_RADIUS),
        basis_tests_inside_l2_ball=np.asarray(True),

        # -----------------------------------------------------------------
        # Compatibility with older single-rollout plotters/renderers.
        # These fields contain the overall worst across all three groups.
        # -----------------------------------------------------------------
        X_state_feedback_rollout=X_overall,
        U_state_feedback_rollout=U_overall,
        x_perturbation_history=Ehist_overall,
        feedback_corrections=dU_overall,
        W_state_feedback_rollout=W_overall,
        disturbance_injection_state_feedback_rollout=D_overall,
        worst_rollout_type=np.asarray(overall_type),
        worst_rollout_index=np.asarray(overall_idx),

        # Shared metadata/settings.
        disturbance_mode_names=np.asarray(DISTURBANCE_MODES),
        num_random_rollouts=np.asarray(NUM_RANDOM_ROLLOUTS),
        num_adversarial_rollouts=np.asarray(NUM_ADVERSARIAL_ROLLOUTS),
        num_basis_rollouts=np.asarray(NUM_BASIS_ROLLOUTS),
        num_total_rollouts=np.asarray(total_rollouts),
        rng_seed=np.asarray(RNG_SEED),
        disturbance_radius=np.asarray(DISTURBANCE_RADIUS),
        random_sample_on_boundary=np.asarray(
            RANDOM_SAMPLE_ON_BOUNDARY
        ),
        adversarial_pgd_steps=np.asarray(ADVERSARIAL_PGD_STEPS),
        adversarial_step_size=np.asarray(ADVERSARIAL_STEP_SIZE),
        adversarial_softmax_temperature=np.asarray(
            ADVERSARIAL_SOFTMAX_TEMPERATURE
        ),
        clip_controls=np.asarray(CLIP_CONTROLS),
        renormalize_quaternion=np.asarray(
            RENORMALIZE_QUATERNION
        ),

        **metadata,
    )

    print()
    print(f"Saved:\n  {OUTPUT_NPZ}")
    print(
        f"Contains {NUM_RANDOM_ROLLOUTS} random + "
        f"{NUM_ADVERSARIAL_ROLLOUTS} adversarial + "
        f"{NUM_BASIS_ROLLOUTS} coordinate-basis +/-1 closed-loop rollouts."
    )
    print(
        "Every coordinate-basis +/-1 rollout satisfies ||w_k||_2 = 1 "
        "at every timestep."
    )
    print(
        "X_state_feedback_rollout contains the overall worst rollout "
        "by exact max |dx|/tube for compatibility with existing plotters."
    )


if __name__ == "__main__":
    main()