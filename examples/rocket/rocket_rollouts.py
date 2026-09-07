#!/usr/bin/env python3
"""
Robust closed-loop rollout evaluator for the FastSLS rocket experiment.

This mirrors the quadruped robust-rollout evaluator, but uses the feedback
representation actually saved by the rocket experiment:

    u_k = U_k + sum_{j=0}^k Phi_u[k,j] @ history_j,
    history_0 = 0, history_j = w_{j-1} for j >= 1

and the exact state-dependent disturbance callback from the rocket planner:

    x_{k+1} = f(x_k, u_k) + E(x_k) @ w_k
    ||w_k||_2 <= 1.

Test groups
-----------
1) RANDOM
   Random disturbance sequences with one fixed support family per rollout.

2) ADVERSARIAL
   Projected-gradient ascent over the complete disturbance sequence, maximizing
   a smooth approximation of the worst tube-normalized trajectory error.

3) BASIS +/-1
   Deterministic +e_i and -e_i disturbances held for the entire horizon.

The script does NOT rerun the optimizer. It loads rocket_result.npz and
reconstructs only the dynamics/disturbance functions from the planner source.

Example
-------
python3 robust_rollouts_rocket.py \
    rocket_result.npz \
    --planner-script rocket_min_time.py \
    --output disturbance_rollouts_rocket.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

NUM_RANDOM_ROLLOUTS = 256
NUM_ADVERSARIAL_ROLLOUTS = 256
RNG_SEED = 0

DISTURBANCE_RADIUS = 1.0

# Random points on the boundary give a more aggressive stress test.
RANDOM_SAMPLE_ON_BOUNDARY = True

# Projected-gradient ascent.
ADVERSARIAL_PGD_STEPS = 40
ADVERSARIAL_STEP_SIZE = 0.25
ADVERSARIAL_SOFTMAX_TEMPERATURE = 0.05
ADVERSARIAL_ABS_EPS = 1.0e-12

TUBE_EPS = 1.0e-10
ADVERSARIAL_TUBE_FLOOR = 1.0e-8

# Leave False to expose controller demands beyond the nominal actuator box.
CLIP_CONTROLS = False

PRINT_PER_ROLLOUT = False


# Rocket state layout:
# [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r, mass, T_final]
PX, PY, PZ = 0, 1, 2
PHI, THETA, PSI = 3, 4, 5
VX, VY, VZ = 6, 7, 8
P_RATE, Q_RATE, R_RATE = 9, 10, 11
MASS = 12
T_FINAL = 13

PHYSICAL_N = 13

DISTURBANCE_MODES = (
    "vx",
    "vz",
    "vx_vz",
    "all_physical",
)


# -----------------------------------------------------------------------------
# Planner import / saved solution
# -----------------------------------------------------------------------------

def import_planner(planner_path: Path):
    planner_path = planner_path.resolve()
    if not planner_path.is_file():
        raise FileNotFoundError(f"Planner script does not exist: {planner_path}")

    # Let sibling imports such as visualize_experiment resolve normally.
    sys.path.insert(0, str(planner_path.parent))

    spec = importlib.util.spec_from_file_location(
        "_rocket_rollout_planner",
        planner_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import planner from {planner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Result NPZ does not exist: {path}")

    with np.load(path, allow_pickle=False) as f:
        saved = {k: np.asarray(f[k]) for k in f.files}

    # The rocket saver currently writes both plan and X. Prefer X.
    if "X" in saved:
        X = saved["X"]
    elif "plan" in saved:
        X = saved["plan"]
    else:
        raise KeyError(
            f"{path} contains neither 'X' nor 'plan'; keys={list(saved)}"
        )

    if "controls" not in saved:
        raise KeyError(
            f"{path} is missing 'controls'; keys={list(saved)}"
        )
    if "Phi_u" not in saved:
        raise KeyError(
            f"{path} is missing 'Phi_u'; keys={list(saved)}"
        )
    if "Phi_x" not in saved:
        raise KeyError(
            f"{path} is missing 'Phi_x'; keys={list(saved)}"
        )

    U = saved["controls"]
    Phi_u = saved["Phi_u"]
    Phi_x = saved["Phi_x"]

    N = U.shape[0]
    nx = X.shape[1]

    if X.shape != (N + 1, nx):
        raise ValueError(f"Bad X/U horizon: X={X.shape}, U={U.shape}")

    if nx != 14:
        raise ValueError(
            f"Expected rocket state dimension 14, got X.shape={X.shape}"
        )

    if Phi_u.shape[0] != N or Phi_u.shape[1] not in (N, N + 1):
        raise ValueError(
            "Expected Phi_u shape (N, N or N+1, nu, nw); "
            f"with N={N}, got {Phi_u.shape}"
        )

    if Phi_u.shape[2] != U.shape[1]:
        raise ValueError(
            f"Phi_u control dimension {Phi_u.shape[2]} != {U.shape[1]}"
        )

    if Phi_u.shape[3] != nx:
        raise ValueError(
            f"Phi_u disturbance dimension {Phi_u.shape[3]} != nx={nx}"
        )

    # Tube is directly saved by the rocket experiment. If unavailable, reconstruct
    # the usual L2-ball componentwise tube from Phi_x.
    if "tube" in saved:
        tube = saved["tube"]
    else:
        # For ||w_j||_2 <= 1 independently at each stage:
        # tube[k,i] = sum_j ||Phi_x[k,j,i,:]||_2.
        # Handle common Phi_x layouts with N or N+1 state rows.
        phi = np.asarray(Phi_x)
        if phi.ndim != 4:
            raise ValueError(
                "Cannot reconstruct tube: expected rank-4 Phi_x, "
                f"got {phi.shape}"
            )

        tube_core = np.sum(np.linalg.norm(phi, axis=-1), axis=1)

        if tube_core.shape[0] == N:
            tube = np.concatenate(
                [np.zeros((1, tube_core.shape[1]), dtype=tube_core.dtype),
                 tube_core],
                axis=0,
            )
        elif tube_core.shape[0] == N + 1:
            tube = tube_core
        else:
            raise ValueError(
                f"Cannot align reconstructed tube {tube_core.shape} with N={N}"
            )

    if tube.shape[0] == N:
        tube = np.concatenate(
            [np.zeros((1, tube.shape[1]), dtype=tube.dtype), tube],
            axis=0,
        )

    if tube.shape[0] != N + 1:
        raise ValueError(f"Tube has incompatible shape: {tube.shape}")

    return saved, X, U, Phi_x, Phi_u, tube


# -----------------------------------------------------------------------------
# Disturbance support / generation
# -----------------------------------------------------------------------------

def make_disturbance_mode_masks(num_rollouts: int, disturbance_dim: int):
    masks = np.zeros((num_rollouts, disturbance_dim), dtype=np.float32)

    mode_ids = (
        np.arange(num_rollouts, dtype=np.int32)
        % len(DISTURBANCE_MODES)
    )

    for r in range(num_rollouts):
        mode = DISTURBANCE_MODES[int(mode_ids[r])]

        if mode == "vx":
            masks[r, VX] = 1.0
        elif mode == "vz":
            masks[r, VZ] = 1.0
        elif mode == "vx_vz":
            masks[r, [VX, VZ]] = 1.0
        elif mode == "all_physical":
            masks[r, :PHYSICAL_N] = 1.0
        else:
            raise ValueError(mode)

    return jnp.asarray(masks), mode_ids


def masked_unit_directions(raw, rollout_masks):
    masked = raw * rollout_masks[:, None, :]
    norm = jnp.linalg.norm(masked, axis=-1, keepdims=True)
    return masked / jnp.maximum(norm, 1.0e-12)


def random_w_sequences(
    raw_directions,
    radial_uniform,
    rollout_masks,
    *,
    on_boundary: bool,
):
    directions = masked_unit_directions(raw_directions, rollout_masks)

    active_dim = jnp.maximum(
        jnp.sum(rollout_masks > 0.0, axis=1),
        1,
    )

    if on_boundary:
        radius = jnp.full(
            radial_uniform.shape,
            DISTURBANCE_RADIUS,
            dtype=raw_directions.dtype,
        )
    else:
        radius = (
            DISTURBANCE_RADIUS
            * radial_uniform
            ** (
                1.0
                / active_dim[:, None].astype(raw_directions.dtype)
            )
        )

    return directions * radius[..., None]


def project_w_to_ball(W, rollout_masks):
    W = W * rollout_masks[:, None, :]
    norm = jnp.linalg.norm(W, axis=-1, keepdims=True)
    scale = jnp.minimum(
        1.0,
        DISTURBANCE_RADIUS / jnp.maximum(norm, 1.0e-12),
    )
    return W * scale


def detect_active_disturbance_dimensions(disturbance_model, X_nominal, tol=1.0e-10):
    """Return disturbance coordinates whose E(x) columns are nonzero.

    A disturbance coordinate i is considered active if the corresponding column
    E(x_k)[:, i] has norm > tol at any nominal trajectory node x_k.
    """
    E_nodes = jax.vmap(disturbance_model.at_state)(X_nominal)
    column_norms = jnp.linalg.norm(E_nodes, axis=1)
    active = jnp.any(column_norms > tol, axis=0)
    return np.flatnonzero(np.asarray(active)).astype(np.int32)


def constant_basis_w_sequences(horizon, disturbance_dim, active_indices, dtype):
    """Generate constant +/- basis disturbances only in active coordinates.

    Each rollout holds w_k = +/- e_i for the entire horizon, but only for
    disturbance coordinates i that actually affect the plant through E(x).
    """
    active_indices = np.asarray(active_indices, dtype=np.int32).reshape(-1)
    if active_indices.size == 0:
        raise ValueError(
            "No active disturbance dimensions were detected from E(x) along "
            "the nominal trajectory."
        )

    eye = jnp.eye(disturbance_dim, dtype=dtype)[jnp.asarray(active_indices)]
    signed_basis = jnp.stack([eye, -eye], axis=1)
    signed_basis = signed_basis.reshape(2 * active_indices.size, disturbance_dim)

    W = jnp.broadcast_to(
        signed_basis[:, None, :],
        (2 * active_indices.size, horizon, disturbance_dim),
    )

    coordinate_indices = np.repeat(active_indices, 2)
    signs = np.tile(
        np.asarray([1.0, -1.0], dtype=np.float32),
        active_indices.size,
    )
    labels = np.asarray([
        f"w_{i:02d}_{'plus' if s > 0 else 'minus'}"
        for i, s in zip(coordinate_indices, signs)
    ])

    return W, coordinate_indices, signs, labels


# -----------------------------------------------------------------------------
# Closed-loop rollout
# -----------------------------------------------------------------------------

def make_rollout_from_w(planner, disturbance_model, parameter, u_min, u_max):
    """
    Roll out exact nonlinear rocket dynamics with SLS disturbance-history feedback.

        history_0 = 0, history_j = w_{j-1}
        du_k = sum_{j=0}^k Phi_u[k,j] history_j
        u_k  = U[k] + du_k
        x+   = f(x,u) + E(x) w_k
    """

    dynamics = planner.dynamics

    def dynamics_with_parameter(x, u, k):
        return dynamics(x, u, k, parameter=parameter)

    batched_dynamics = jax.vmap(
        dynamics_with_parameter,
        in_axes=(0, 0, None),
        out_axes=0,
    )

    batched_disturbance = jax.vmap(
        disturbance_model.at_state,
        in_axes=0,
        out_axes=0,
    )

    def rollout_from_w(X_nominal, U_nominal, Phi_u, W_sequences):
        R = W_sequences.shape[0]
        N = U_nominal.shape[0]
        nx = X_nominal.shape[1]

        x0_batch = jnp.broadcast_to(X_nominal[0], (R, nx))

        def scan_step(x_batch, k):
            # Phi_u[k]: (Nhist, nu, nw), where Nhist is N or N+1.
            # The rocket solver currently saves Nhist=N+1.
            Phi_k = Phi_u[k]
            n_hist = Phi_k.shape[0]

            history_mask = (
                jnp.arange(n_hist) <= k
            ).astype(X_nominal.dtype)

            Phi_k_causal = (
                Phi_k * history_mask[:, None, None]
            )

            # Match the planner's original disturbance_history convention:
            #
            #   history[0] = 0
            #   history[j] = w_{j-1},  j >= 1
            #
            # Therefore u_k never has access to w_k before that disturbance
            # is injected into x_{k+1}.
            zero = jnp.zeros_like(W_sequences[:, :1, :])

            # Solver convention:
            #   history[0]   = 0
            #   history[j+1] = w_j
            #
            # If Phi_u has N+1 history columns, retain all N disturbances; the
            # causal mask ensures column k+1 (which contains w_k) is NOT used
            # when computing u_k. If an older result has only N columns, drop
            # the final unused history entry.
            disturbance_history_full = jnp.concatenate(
                [zero, W_sequences],
                axis=1,
            )
            disturbance_history = disturbance_history_full[:, :n_hist, :]

            du_batch = jnp.einsum(
                "rjw,jmw->rm",
                disturbance_history,
                Phi_k_causal,
            )

            u_batch = U_nominal[k][None, :] + du_batch

            if CLIP_CONTROLS:
                u_batch = jnp.clip(
                    u_batch,
                    u_min[None, :],
                    u_max[None, :],
                )

            x_det_next = batched_dynamics(
                x_batch,
                u_batch,
                k,
            )

            E_batch = batched_disturbance(x_batch)
            w_batch = W_sequences[:, k, :]

            d_batch = jnp.einsum(
                "rij,rj->ri",
                E_batch,
                w_batch,
            )

            x_next = x_det_next + d_batch
            err_next = x_next - X_nominal[k + 1][None, :]

            return x_next, (
                x_next,
                u_batch,
                du_batch,
                w_batch,
                d_batch,
                err_next,
            )

        _, outputs = jax.lax.scan(
            jax.checkpoint(scan_step),
            x0_batch,
            jnp.arange(N, dtype=jnp.int32),
        )

        X_next_tm, U_tm, dU_tm, W_tm, D_tm, err_tm = outputs

        X_next = jnp.swapaxes(X_next_tm, 0, 1)
        U_all = jnp.swapaxes(U_tm, 0, 1)
        dU_all = jnp.swapaxes(dU_tm, 0, 1)
        W_all = jnp.swapaxes(W_tm, 0, 1)
        D_all = jnp.swapaxes(D_tm, 0, 1)
        err_next = jnp.swapaxes(err_tm, 0, 1)

        X_all = jnp.concatenate(
            [x0_batch[:, None, :], X_next],
            axis=1,
        )

        err0 = x0_batch - X_nominal[0][None, :]
        error_history = jnp.concatenate(
            [err0[:, None, :], err_next],
            axis=1,
        )

        return X_all, U_all, error_history, dU_all, W_all, D_all

    return rollout_from_w


# -----------------------------------------------------------------------------
# Adversarial optimization
# -----------------------------------------------------------------------------

def adversarial_scores_from_error(error_history, tube):
    err = error_history[:, :, :PHYSICAL_N]
    tube_phys = tube[:, :PHYSICAL_N]
    valid = tube_phys > TUBE_EPS

    smooth_abs = jnp.sqrt(err * err + ADVERSARIAL_ABS_EPS)

    ratio = smooth_abs / jnp.maximum(
        tube_phys[None, :, :],
        ADVERSARIAL_TUBE_FLOOR,
    )

    masked = jnp.where(
        valid[None, :, :],
        ratio,
        -1.0e6,
    )

    flat = masked.reshape(masked.shape[0], -1)
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
    Phi_u,
    tube,
    rollout_masks,
):
    def objective_sum(W):
        error_history = rollout_from_w(
            X_nominal,
            U_nominal,
            Phi_u,
            W,
        )[2]

        return jnp.sum(
            adversarial_scores_from_error(
                error_history,
                tube,
            )
        )

    objective_grad = jax.grad(objective_sum)

    @jax.jit
    def optimize(W_initial):
        def body(_, W):
            grad = objective_grad(W)
            grad = grad * rollout_masks[:, None, :]

            grad_norm = jnp.linalg.norm(
                grad,
                axis=-1,
                keepdims=True,
            )
            unit_grad = grad / jnp.maximum(grad_norm, 1.0e-12)

            W_next = W + ADVERSARIAL_STEP_SIZE * unit_grad
            return project_w_to_ball(W_next, rollout_masks)

        return jax.lax.fori_loop(
            0,
            ADVERSARIAL_PGD_STEPS,
            body,
            W_initial,
        )

    return optimize


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def compute_diagnostics(U_all, error_history, tube, u_min, u_max):
    physical_error = jnp.abs(
        error_history[:, :, :PHYSICAL_N]
    )
    tube_phys = jnp.asarray(tube[:, :PHYSICAL_N])
    valid = tube_phys > TUBE_EPS

    ratio = jnp.where(
        valid[None, :, :],
        physical_error
        / jnp.maximum(tube_phys[None, :, :], TUBE_EPS),
        jnp.nan,
    )

    safe_ratio = jnp.where(jnp.isnan(ratio), -jnp.inf, ratio)

    state_err_max = jnp.max(physical_error, axis=(1, 2))
    tube_ratio_max = jnp.max(safe_ratio, axis=(1, 2))

    flat_idx = jnp.argmax(
        safe_ratio.reshape(safe_ratio.shape[0], -1),
        axis=1,
    )
    worst_nodes = flat_idx // PHYSICAL_N
    worst_states = flat_idx % PHYSICAL_N

    upper_violation = U_all - u_max[None, None, :]
    lower_violation = u_min[None, None, :] - U_all

    control_violation = jnp.maximum(
        0.0,
        jnp.maximum(
            jnp.max(upper_violation, axis=(1, 2)),
            jnp.max(lower_violation, axis=(1, 2)),
        ),
    )

    return {
        "state_err_max": state_err_max,
        "tube_ratio_max": tube_ratio_max,
        "worst_node": worst_nodes,
        "worst_state": worst_states,
        "control_violation": control_violation,
        "tube_contained": tube_ratio_max <= (1.0 + 1.0e-5),
        "control_feasible": control_violation <= 1.0e-6,
    }



def compute_terminal_entry_times(
    X_rollouts,
    goal_center,
    goal_half_width,
    dt: float,
):
    """Return first terminal-set entry node/time for each rollout.

    A rollout is counted as having reached the terminal set as soon as ANY
    trajectory node satisfies the planner's 3-D terminal-box condition:

        abs(x[:3] - goal_center[:3]) <= goal_half_width[:3]

    componentwise. Rollouts that never enter receive node=-1 and time=NaN.
    """
    X_rollouts = np.asarray(X_rollouts, dtype=float)
    goal_center = np.asarray(goal_center, dtype=float).reshape(-1)[:3]
    goal_half_width = np.asarray(goal_half_width, dtype=float).reshape(-1)[:3]

    if X_rollouts.ndim != 3 or X_rollouts.shape[2] < 3:
        raise ValueError(
            f"Expected rollout states with shape (R, N+1, nx>=3), got {X_rollouts.shape}"
        )

    inside = np.all(
        np.abs(X_rollouts[:, :, :3] - goal_center[None, None, :])
        <= goal_half_width[None, None, :],
        axis=2,
    )

    reached = np.any(inside, axis=1)
    first_nodes = np.full(X_rollouts.shape[0], -1, dtype=np.int32)
    first_nodes[reached] = np.argmax(inside[reached], axis=1).astype(np.int32)

    first_times = np.full(X_rollouts.shape[0], np.nan, dtype=float)
    first_times[reached] = first_nodes[reached].astype(float) * float(dt)

    return reached, first_nodes, first_times


def summarize_terminal_entries(name, reached, first_nodes, first_times):
    reached = np.asarray(reached, dtype=bool)
    first_nodes = np.asarray(first_nodes, dtype=np.int32)
    first_times = np.asarray(first_times, dtype=float)

    count = int(np.sum(reached))
    total = int(reached.size)

    print()
    print(f"{name} terminal-set arrival")
    print("-" * 72)
    print(f"Reached terminal set:          {count}/{total} ({100.0 * count / max(total, 1):.2f}%)")

    if count:
        valid_times = first_times[reached]
        valid_nodes = first_nodes[reached]
        print(f"Average first-entry time [s]: {np.mean(valid_times):.6f}")
        print(f"Std first-entry time [s]:     {np.std(valid_times):.6f}")
        print(f"Earliest first entry [s]:     {np.min(valid_times):.6f}")
        print(f"Max first-entry time [s]:     {np.max(valid_times):.6f}")
        print(f"Average first-entry node:     {np.mean(valid_nodes):.3f}")
    else:
        print("Average first-entry time [s]: nan (no rollout reached the terminal set)")


def compute_tube_xyz_metrics(tube):
    """Compute position-tube size metrics from componentwise SLS half-widths.

    The saved ``tube[k, i]`` is the componentwise half-width around the nominal
    state. For position coordinates, the full box widths are therefore

        dx_k = 2 * |tube[k, x]|
        dy_k = 2 * |tube[k, y]|
        dz_k = 2 * |tube[k, z]|.

    For a 3-D rectangular tube box, there is no single planar "perimeter".
    Here we report the total length of all 12 box edges,

        P_xyz,k = 4 * (dx_k + dy_k + dz_k).

    We also report the maximum full width in x, y, and z separately.
    """
    tube = np.asarray(tube, dtype=float)
    if tube.ndim != 2 or tube.shape[1] < 3:
        raise ValueError(f"Expected tube shape (N+1, nx>=3), got {tube.shape}")

    half_xyz = np.abs(tube[:, :3])
    widths_xyz = 2.0 * half_xyz
    edge_perimeter_xyz = 4.0 * np.sum(widths_xyz, axis=1)

    max_width_xyz = np.nanmax(widths_xyz, axis=0)
    max_perimeter = float(np.nanmax(edge_perimeter_xyz))
    max_perimeter_node = int(np.nanargmax(edge_perimeter_xyz))

    return {
        "widths_xyz": widths_xyz,
        "edge_perimeter_xyz": edge_perimeter_xyz,
        "max_width_x": float(max_width_xyz[0]),
        "max_width_y": float(max_width_xyz[1]),
        "max_width_z": float(max_width_xyz[2]),
        "max_edge_perimeter_xyz": max_perimeter,
        "max_edge_perimeter_node": max_perimeter_node,
    }


def summarize_tube_xyz_metrics(metrics):
    print()
    print("XYZ SLS tube size")
    print("-" * 72)
    print(f"Max full tube width x [m]:     {metrics['max_width_x']:.6f}")
    print(f"Max full tube width y [m]:     {metrics['max_width_y']:.6f}")
    print(f"Max full tube width z [m]:     {metrics['max_width_z']:.6f}")
    print(
        f"Max XYZ tube edge perimeter [m]: "
        f"{metrics['max_edge_perimeter_xyz']:.6f}"
    )
    print(
        f"Max XYZ perimeter node:          "
        f"{metrics['max_edge_perimeter_node']}"
    )

def state_name(i: int) -> str:
    names = (
        "px", "py", "pz",
        "phi", "theta", "psi",
        "vx", "vy", "vz",
        "p", "q", "r",
        "mass", "T_final",
    )
    return names[i] if 0 <= i < len(names) else f"x{i}"


def summarize_group(name, diagnostics):
    d = {k: np.asarray(v) for k, v in diagnostics.items()}

    ratios = d["tube_ratio_max"]
    finite_ratios = ratios[np.isfinite(ratios)]

    contained = d["tube_contained"].astype(bool)
    controls_ok = d["control_feasible"].astype(bool)

    worst_rollout = int(np.nanargmax(ratios))
    worst_node = int(d["worst_node"][worst_rollout])
    worst_state_idx = int(d["worst_state"][worst_rollout])

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(f"Rollouts:                    {len(ratios)}")
    print(
        f"Tube-contained:              "
        f"{int(np.sum(contained))}/{len(contained)} "
        f"({100.0 * np.mean(contained):.2f}%)"
    )
    print(
        f"Control-feasible:            "
        f"{int(np.sum(controls_ok))}/{len(controls_ok)} "
        f"({100.0 * np.mean(controls_ok):.2f}%)"
    )
    if finite_ratios.size:
        print(f"Worst tube ratio:            {np.max(finite_ratios):.6f}")
        print(f"Mean max tube ratio:         {np.mean(finite_ratios):.6f}")
    print(
        f"Worst absolute state error:  "
        f"{np.max(d['state_err_max']):.6f}"
    )
    print(
        f"Max control violation:       "
        f"{np.max(d['control_violation']):.6f}"
    )
    print(
        f"Worst location:              rollout {worst_rollout}, "
        f"node {worst_node}, state {state_name(worst_state_idx)}"
    )


def print_per_rollout(name, diagnostics, labels=None):
    if not PRINT_PER_ROLLOUT:
        return

    d = {k: np.asarray(v) for k, v in diagnostics.items()}

    print()
    print(f"{name} per-rollout")
    print("-" * 100)
    print(
        f"{'idx':>5} {'label':>20} {'tube ratio':>12} "
        f"{'max |dx|':>12} {'u viol':>12} {'node':>8} {'state':>10}"
    )
    print("-" * 100)

    for i in range(len(d["tube_ratio_max"])):
        label = "-" if labels is None else str(labels[i])
        print(
            f"{i:5d} {label:>20} "
            f"{d['tube_ratio_max'][i]:12.5f} "
            f"{d['state_err_max'][i]:12.5f} "
            f"{d['control_violation'][i]:12.5f} "
            f"{int(d['worst_node'][i]):8d} "
            f"{state_name(int(d['worst_state'][i])):>10}"
        )


# -----------------------------------------------------------------------------
# Side-view rollout visualization
# -----------------------------------------------------------------------------


def plot_rocket_side_view_rollouts_tubes_goal_obstacles(
    *,
    xs,
    plan,
    lower,
    upper,
    centers,
    radii,
    goal_center,
    goal_half_width,
    tube_stride: int = 1,
    ground_height: float = 0.0,
    filename: str = "rocket_side_view_rollouts.png",
    title: str = "Rocket robust rollouts and SLS tubes — side view",
    terminal_visible_fraction: float = 0.35,
    rollout_alpha: float = 0.20,
    rollout_linewidth: float = 0.8,
) -> None:
    """Side-view x-z plot matching plot_rocket_side_view_tubes_goal_obstacles."""
    plan = np.asarray(plan, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    xs = np.asarray(xs, dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    goal_center = np.asarray(goal_center, dtype=float).reshape(-1)
    goal_half_width = np.asarray(goal_half_width, dtype=float).reshape(-1)

    if xs.ndim == 2:
        xs = xs[None, ...]
    if xs.ndim != 3:
        raise ValueError(f"xs must have shape (R,N+1,nx) or (N+1,nx), got {xs.shape}")

    fig, ax = plt.subplots(figsize=(9, 6.2))

    rollout_label_used = False
    for rollout in xs:
        finite = np.isfinite(rollout[:, 0]) & np.isfinite(rollout[:, 2])
        if not np.any(finite):
            continue
        rr = rollout[finite]
        ax.plot(
            rr[:, 0], rr[:, 2],
            linewidth=rollout_linewidth,
            alpha=rollout_alpha,
            label="Robust rollouts" if not rollout_label_used else None,
        )
        rollout_label_used = True

    ax.plot(plan[:, 0], plan[:, 2], linewidth=2.2, label="Nominal plan")
    ax.scatter(plan[0, 0], plan[0, 2], s=42, label="Start")
    ax.scatter(plan[-1, 0], plan[-1, 2], s=55, marker="*", label="Terminal")

    stride = max(1, int(tube_stride))
    for k in range(0, min(len(lower), len(upper)), stride):
        w = upper[k, 0] - lower[k, 0]
        h = upper[k, 2] - lower[k, 2]
        if np.isfinite([lower[k, 0], lower[k, 2], w, h]).all():
            ax.add_patch(Rectangle((lower[k, 0], lower[k, 2]), w, h, alpha=0.06, linewidth=0.0))

    for c, r in zip(centers, radii):
        ax.add_patch(Circle((c[0], c[2]), float(r), alpha=0.22))

    goal_lo = np.array([
        goal_center[0] - goal_half_width[0],
        goal_center[2] - goal_half_width[2],
    ])
    goal_size = np.array([
        2.0 * goal_half_width[0],
        2.0 * goal_half_width[2],
    ])
    ax.add_patch(Rectangle(goal_lo, goal_size[0], goal_size[1], alpha=0.16, label="Goal set"))

    ax.axhline(float(ground_height), linewidth=1.2, linestyle="--", label="Ground")

    finite_rx = xs[:, :, 0][np.isfinite(xs[:, :, 0])]
    finite_rz = xs[:, :, 2][np.isfinite(xs[:, :, 2])]
    x_parts = [lower[:, 0], upper[:, 0], plan[:, 0]]
    z_parts = [lower[:, 2], upper[:, 2], plan[:, 2], np.asarray([ground_height])]
    if finite_rx.size:
        x_parts.append(finite_rx)
    if finite_rz.size:
        z_parts.append(finite_rz)
    x_all = np.concatenate([np.asarray(v).reshape(-1) for v in x_parts])
    z_all = np.concatenate([np.asarray(v).reshape(-1) for v in z_parts])
    x_all = x_all[np.isfinite(x_all)]
    z_all = z_all[np.isfinite(z_all)]

    x_data_min, x_data_max = np.min(x_all), np.max(x_all)
    z_data_min, z_data_max = np.min(z_all), np.max(z_all)
    x_span = max(x_data_max - x_data_min, 1.0)
    z_span = max(z_data_max - z_data_min, 1.0)
    x_margin = 0.05 * x_span
    z_margin = 0.08 * z_span

    goal_x_min = goal_center[0] - goal_half_width[0]
    goal_visible_x = goal_x_min + terminal_visible_fraction * 2.0 * goal_half_width[0]
    x_right = max(x_data_max + x_margin, goal_visible_x)

    ax.set_xlim(x_data_min - x_margin, x_right)
    ax.set_ylim(z_data_min - z_margin, z_data_max + z_margin)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    path = Path(filename)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robust random/adversarial/basis rollouts for rocket_result.npz"
    )
    parser.add_argument(
        "result_npz",
        nargs="?",
        type=Path,
        default=Path("rocket_result.npz"),
        help="Saved robust rocket result.",
    )
    parser.add_argument(
        "--planner-script",
        type=Path,
        required=True,
        help=(
            "Rocket planner Python file containing dynamics(), "
            "rocket_6dof_step(), and make_min_time_disturbance()."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("disturbance_rollouts_rocket.npz"),
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=NUM_RANDOM_ROLLOUTS,
    )
    parser.add_argument(
        "--num-adversarial",
        type=int,
        default=NUM_ADVERSARIAL_ROLLOUTS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for side-view rollout plots; defaults to output NPZ directory.",
    )
    args = parser.parse_args()

    planner = import_planner(args.planner_script)

    saved, X_np, U_np, Phi_x_np, Phi_u_np, tube_np = load_solution(
        args.result_npz
    )

    X = jnp.asarray(X_np)
    U = jnp.asarray(U_np)
    Phi_u = jnp.asarray(Phi_u_np)
    tube = jnp.asarray(tube_np)

    N = U.shape[0]
    nx = X.shape[1]
    nu = U.shape[1]

    # Exact experiment parameter: dynamics() uses dtau * x[-1].
    parameter = 1.0 / N

    if not hasattr(planner, "make_min_time_disturbance"):
        raise AttributeError(
            "Planner is missing make_min_time_disturbance()."
        )

    disturbance_model = planner.make_min_time_disturbance(
        n=nx,
        N=N,
        disturbance_index=VX,
    )

    if not hasattr(disturbance_model, "at_state"):
        raise AttributeError(
            "Planner disturbance callback has no .at_state function."
        )

    # Rebuild actuator limits from the planner constants.
    wet_mass = float(getattr(planner, "WET_MASS", 5.0))
    gravity = float(getattr(planner, "GRAVITY", 9.81))

    T_max = 4.0 * wet_mass * gravity
    gimbal_max = np.deg2rad(60.0)
    roll_torque_max = 2.0

    u_min = jnp.asarray(
        [0.0, -gimbal_max, -gimbal_max, -roll_torque_max],
        dtype=X.dtype,
    )
    u_max = jnp.asarray(
        [T_max, gimbal_max, gimbal_max, roll_torque_max],
        dtype=X.dtype,
    )

    if nu != 4:
        raise ValueError(f"Expected 4 rocket controls, got {nu}")

    rollout_from_w = make_rollout_from_w(
        planner,
        disturbance_model,
        parameter,
        u_min,
        u_max,
    )

    rollout_jit = jax.jit(rollout_from_w)

    key = jax.random.PRNGKey(args.seed)
    key_random_dir, key_random_rad, key_adv_dir, key_adv_rad = jax.random.split(
        key, 4
    )

    # -------------------------------------------------------------------------
    # Random
    # -------------------------------------------------------------------------
    random_masks, random_mode_ids = make_disturbance_mode_masks(
        args.num_random,
        nx,
    )

    random_raw = jax.random.normal(
        key_random_dir,
        (args.num_random, N, nx),
        dtype=X.dtype,
    )
    random_radial = jax.random.uniform(
        key_random_rad,
        (args.num_random, N),
        minval=0.0,
        maxval=1.0,
        dtype=X.dtype,
    )

    W_random = random_w_sequences(
        random_raw,
        random_radial,
        random_masks,
        on_boundary=RANDOM_SAMPLE_ON_BOUNDARY,
    )

    random_outputs = rollout_jit(X, U, Phi_u, W_random)
    random_diag = compute_diagnostics(
        random_outputs[1],
        random_outputs[2],
        tube,
        u_min,
        u_max,
    )

    # -------------------------------------------------------------------------
    # Adversarial PGD
    # -------------------------------------------------------------------------
    adversarial_masks, adversarial_mode_ids = make_disturbance_mode_masks(
        args.num_adversarial,
        nx,
    )

    adv_raw = jax.random.normal(
        key_adv_dir,
        (args.num_adversarial, N, nx),
        dtype=X.dtype,
    )
    adv_radial = jax.random.uniform(
        key_adv_rad,
        (args.num_adversarial, N),
        minval=0.0,
        maxval=1.0,
        dtype=X.dtype,
    )

    W_adv_initial = random_w_sequences(
        adv_raw,
        adv_radial,
        adversarial_masks,
        on_boundary=True,
    )

    adversarial_optimizer = make_adversarial_optimizer(
        rollout_from_w,
        X,
        U,
        Phi_u,
        tube,
        adversarial_masks,
    )

    W_adversarial = adversarial_optimizer(W_adv_initial)
    adversarial_outputs = rollout_jit(X, U, Phi_u, W_adversarial)
    adversarial_diag = compute_diagnostics(
        adversarial_outputs[1],
        adversarial_outputs[2],
        tube,
        u_min,
        u_max,
    )

    # -------------------------------------------------------------------------
    # +/- basis sweep -- only disturbance coordinates that are actually active
    # through E(x) along the nominal trajectory.
    # -------------------------------------------------------------------------
    active_basis_dimensions = detect_active_disturbance_dimensions(
        disturbance_model,
        X,
    )

    print(
        "Active disturbance dimensions for basis sweep: "
        + ", ".join(
            f"{int(i)} ({state_name(int(i))})"
            for i in active_basis_dimensions
        )
    )

    W_basis, basis_coordinate, basis_sign, basis_labels = (
        constant_basis_w_sequences(
            N,
            nx,
            active_basis_dimensions,
            X.dtype,
        )
    )

    basis_outputs = rollout_jit(X, U, Phi_u, W_basis)
    basis_diag = compute_diagnostics(
        basis_outputs[1],
        basis_outputs[2],
        tube,
        u_min,
        u_max,
    )

    # -------------------------------------------------------------------------
    # Terminal-set first-entry times
    # -------------------------------------------------------------------------
    nominal_dt = float(X[0, -1]) / N
    goal_center_metric = np.asarray(
        saved.get("goal_center", np.array([20.0, 0.0, 3.0], dtype=float))
    )
    goal_half_width_metric = np.asarray(
        saved.get("goal_half_width", np.array([8.0, 8.0, 1.5], dtype=float))
    )

    random_reached, random_entry_node, random_entry_time = compute_terminal_entry_times(
        random_outputs[0], goal_center_metric, goal_half_width_metric, nominal_dt
    )
    adversarial_reached, adversarial_entry_node, adversarial_entry_time = compute_terminal_entry_times(
        adversarial_outputs[0], goal_center_metric, goal_half_width_metric, nominal_dt
    )
    basis_reached, basis_entry_node, basis_entry_time = compute_terminal_entry_times(
        basis_outputs[0], goal_center_metric, goal_half_width_metric, nominal_dt
    )

    # Position-tube size metrics. The saved tube values are componentwise
    # half-widths, so full x/y/z widths are 2*tube.
    tube_xyz_metrics = compute_tube_xyz_metrics(tube_np)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print()
    print("=" * 72)
    print("ROCKET ROBUST CLOSED-LOOP ROLLOUTS")
    print("=" * 72)
    print(f"Result:                      {args.result_npz.resolve()}")
    print(f"Planner:                     {args.planner_script.resolve()}")
    print(f"Horizon:                     {N}")
    print(f"Nominal minimum time [s]:    {float(X[0, -1]):.6f}")
    print(f"Nominal dt [s]:              {float(X[0, -1] / N):.6f}")
    print(f"Disturbance radius:          {DISTURBANCE_RADIUS}")
    print(f"Clip controls:               {CLIP_CONTROLS}")

    summarize_tube_xyz_metrics(tube_xyz_metrics)

    summarize_group("RANDOM", random_diag)
    summarize_group("ADVERSARIAL PGD", adversarial_diag)
    summarize_group("BASIS +/-1", basis_diag)

    summarize_terminal_entries(
        "RANDOM", random_reached, random_entry_node, random_entry_time
    )
    summarize_terminal_entries(
        "ADVERSARIAL PGD", adversarial_reached, adversarial_entry_node, adversarial_entry_time
    )
    summarize_terminal_entries(
        "BASIS +/-1", basis_reached, basis_entry_node, basis_entry_time
    )

    all_reached = np.concatenate([
        random_reached, adversarial_reached, basis_reached
    ])
    all_entry_nodes = np.concatenate([
        random_entry_node, adversarial_entry_node, basis_entry_node
    ])
    all_entry_times = np.concatenate([
        random_entry_time, adversarial_entry_time, basis_entry_time
    ])
    summarize_terminal_entries(
        "ALL ROLLOUTS", all_reached, all_entry_nodes, all_entry_times
    )

    random_labels = np.asarray([
        DISTURBANCE_MODES[i] for i in random_mode_ids
    ])
    adversarial_labels = np.asarray([
        DISTURBANCE_MODES[i] for i in adversarial_mode_ids
    ])

    print_per_rollout("RANDOM", random_diag, random_labels)
    print_per_rollout("ADVERSARIAL", adversarial_diag, adversarial_labels)
    print_per_rollout("BASIS", basis_diag, basis_labels)

    # -------------------------------------------------------------------------
    # Save everything needed for visualization / further analysis.
    # -------------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def diag_np(prefix, d):
        return {
            f"{prefix}_{k}": np.asarray(v)
            for k, v in d.items()
        }

    output_payload = {
        "nominal_X": np.asarray(X),
        "nominal_U": np.asarray(U),
        "Phi_x": np.asarray(Phi_x_np),
        "Phi_u": np.asarray(Phi_u),
        "tube": np.asarray(tube),
        "disturbance_radius": np.asarray(DISTURBANCE_RADIUS),
        "random_sample_on_boundary": np.asarray(RANDOM_SAMPLE_ON_BOUNDARY),
        "adversarial_pgd_steps": np.asarray(ADVERSARIAL_PGD_STEPS),
        "adversarial_step_size": np.asarray(ADVERSARIAL_STEP_SIZE),
        "seed": np.asarray(args.seed),

        "random_mode_ids": random_mode_ids,
        "random_mode_labels": random_labels,
        "random_W": np.asarray(W_random),
        "random_X": np.asarray(random_outputs[0]),
        "random_U": np.asarray(random_outputs[1]),
        "random_dU": np.asarray(random_outputs[3]),
        "random_D": np.asarray(random_outputs[5]),
        "random_terminal_reached": np.asarray(random_reached),
        "random_terminal_entry_node": np.asarray(random_entry_node),
        "random_terminal_entry_time": np.asarray(random_entry_time),

        "adversarial_mode_ids": adversarial_mode_ids,
        "adversarial_mode_labels": adversarial_labels,
        "adversarial_W_initial": np.asarray(W_adv_initial),
        "adversarial_W": np.asarray(W_adversarial),
        "adversarial_X": np.asarray(adversarial_outputs[0]),
        "adversarial_U": np.asarray(adversarial_outputs[1]),
        "adversarial_dU": np.asarray(adversarial_outputs[3]),
        "adversarial_D": np.asarray(adversarial_outputs[5]),
        "adversarial_terminal_reached": np.asarray(adversarial_reached),
        "adversarial_terminal_entry_node": np.asarray(adversarial_entry_node),
        "adversarial_terminal_entry_time": np.asarray(adversarial_entry_time),

        "active_basis_dimensions": np.asarray(active_basis_dimensions),
        "basis_coordinate": basis_coordinate,
        "basis_sign": basis_sign,
        "basis_labels": basis_labels,
        "basis_W": np.asarray(W_basis),
        "basis_X": np.asarray(basis_outputs[0]),
        "basis_U": np.asarray(basis_outputs[1]),
        "basis_dU": np.asarray(basis_outputs[3]),
        "basis_D": np.asarray(basis_outputs[5]),
        "basis_terminal_reached": np.asarray(basis_reached),
        "basis_terminal_entry_node": np.asarray(basis_entry_node),
        "basis_terminal_entry_time": np.asarray(basis_entry_time),
        "all_terminal_reached": np.asarray(all_reached),
        "all_terminal_entry_node": np.asarray(all_entry_nodes),
        "all_terminal_entry_time": np.asarray(all_entry_times),
        "terminal_goal_center": np.asarray(goal_center_metric),
        "terminal_goal_half_width": np.asarray(goal_half_width_metric),
        "terminal_dt": np.asarray(nominal_dt),
        "average_terminal_entry_time": np.asarray(
            np.nanmean(all_entry_times) if np.any(all_reached) else np.nan
        ),
        "max_terminal_entry_time": np.asarray(
            np.nanmax(all_entry_times) if np.any(all_reached) else np.nan
        ),
        "tube_xyz_full_widths": np.asarray(tube_xyz_metrics["widths_xyz"]),
        "tube_xyz_edge_perimeter": np.asarray(tube_xyz_metrics["edge_perimeter_xyz"]),
        "max_tube_width_x": np.asarray(tube_xyz_metrics["max_width_x"]),
        "max_tube_width_y": np.asarray(tube_xyz_metrics["max_width_y"]),
        "max_tube_width_z": np.asarray(tube_xyz_metrics["max_width_z"]),
        "max_tube_edge_perimeter_xyz": np.asarray(
            tube_xyz_metrics["max_edge_perimeter_xyz"]
        ),
        "max_tube_edge_perimeter_node": np.asarray(
            tube_xyz_metrics["max_edge_perimeter_node"]
        ),
    }

    output_payload.update(diag_np("random", random_diag))
    output_payload.update(diag_np("adversarial", adversarial_diag))
    output_payload.update(diag_np("basis", basis_diag))

    np.savez_compressed(args.output, **output_payload)

    # Side-view plots matching plot_rocket_side_view_tubes_goal_obstacles.
    plot_dir = args.plot_dir if args.plot_dir is not None else args.output.parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    plan_np = np.asarray(X)
    tube_plot = np.asarray(tube)
    lower_plot = plan_np - tube_plot
    upper_plot = plan_np + tube_plot

    centers_plot = np.asarray(saved.get("centers", np.zeros((0, 3), dtype=float)))
    radii_plot = np.asarray(saved.get("radii", np.zeros((0,), dtype=float)))
    goal_center_plot = goal_center_metric
    goal_half_width_plot = goal_half_width_metric

    plot_specs = (
        ("random", np.asarray(random_outputs[0]), "Random robust rocket rollouts and SLS tubes — side view"),
        ("adversarial", np.asarray(adversarial_outputs[0]), "Adversarial robust rocket rollouts and SLS tubes — side view"),
        ("basis", np.asarray(basis_outputs[0]), "Basis ±1 robust rocket rollouts and SLS tubes — side view"),
    )

    for group_name, rollout_states, plot_title in plot_specs:
        plot_path = plot_dir / f"rocket_{group_name}_side_view_rollouts.png"
        plot_rocket_side_view_rollouts_tubes_goal_obstacles(
            xs=rollout_states,
            plan=plan_np,
            lower=lower_plot,
            upper=upper_plot,
            centers=centers_plot,
            radii=radii_plot,
            goal_center=goal_center_plot,
            goal_half_width=goal_half_width_plot,
            tube_stride=1,
            ground_height=0.0,
            filename=str(plot_path),
            title=plot_title,
        )
        print(f"Saved side-view plot: {plot_path.resolve()}")

    print()
    print(f"Saved rollout data: {args.output.resolve()}")


if __name__ == "__main__":
    main()