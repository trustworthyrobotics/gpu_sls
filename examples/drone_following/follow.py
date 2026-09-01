from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np


# ============================================================
# FATROP receding-horizon tracker for the saved GPU-SLS trajectory
#
# Workflow:
#   1. Load the optimized minimum-time GPU-SLS/JAX trajectory.
#   2. Keep its optimized phase times fixed.
#   3. Build ONE reusable FATROP MPC problem.
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

DEFAULT_MPC_HORIZON = 30
DEFAULT_PROGRESS_SEARCH_WINDOW = 15
DEFAULT_GOAL_TOL = 0.362
DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER = 3

# Closed-loop disturbance settings
DISTURBANCE_MAG = 15.0
DISTURBANCE_W = 1.0

T_HOVER = MASS * GRAVITY
T_MAX = 2.0 * T_HOVER
TAU_MAX = 10.0


# ------------------------------------------------------------
# Tracking weights
#
# These are intentionally much stronger than the offline
# trajectory-planning weights. The online problem is now a
# reference tracking problem, not a minimum-time problem.
# ------------------------------------------------------------
Q_POS = np.array([100.0, 100.0, 100.0])
Q_ANG = np.array([20.0, 20.0, 10.0])
Q_VEL = np.array([10.0, 10.0, 10.0])
Q_RATE = np.array([1.0, 1.0, 1.0])

R_U = np.array([0.20, 0.05, 0.05, 0.05])

TERMINAL_MULTIPLIER = 5.0


# ------------------------------------------------------------
# Same physical bounds as the offline planner
# ------------------------------------------------------------
X_MAX = np.array([
    15.0, 15.0, 15.0,
    np.pi / 2.0,
    np.pi / 2.0,
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
# Optional plant disturbance for closed-loop tests
#
# Default = 0.0, so the plant is nominal.
#
# If --disturbance-mag is nonzero, a deterministic disturbance
# is added to vy after each plant step:
#
#   vy_{k+1} += dt * disturbance_mag * spatial_scale(x, y)
#
# This uses the same spatial window as the robust JAX example.
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
    sphi, cphi = ca.sin(phi), ca.cos(phi)
    tth = ca.tan(theta)
    cth = ca.cos(theta)

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
    sphi, cphi = np.sin(phi), np.cos(phi)
    tth = np.tan(theta)
    cth = np.cos(theta)

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
    disturbance_mag=DISTURBANCE_MAG,
    w=DISTURBANCE_W,
):
    """
    Evaluate the disturbance at the CURRENT state.

    We use

        w = 1

    and apply the disturbance only in the +y velocity direction.

    Since the physical state ordering is

        [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r],

    index 7 is v_y.

    The continuous-time disturbance acceleration is

        d_y(x) = disturbance_mag * spatial_scale(px, py) * w

    which is nonnegative for w = +1.
    """
    spatial_scale = disturbance_spatial_scale(
        x[0],
        x[1],
    )

    d_y = (
        disturbance_mag
        * spatial_scale
        * w
    )

    return float(d_y), float(spatial_scale)


def plant_step(
    x,
    u,
    dt,
    disturbance_mag=DISTURBANCE_MAG,
    w=DISTURBANCE_W,
):
    """
    One actual closed-loop plant step.

    Nominal:
        x_{k+1} = x_k + dt * f(x_k, u_k)

    Disturbed:
        v_y,k+1 += dt * E_y(x_k) * w

    with w = +1, so the disturbance always points in +y.
    """
    x_next = x + dt * quadrotor_xdot_np(x, u)

    d_y, spatial_scale = disturbance_at_state(
        x=x,
        disturbance_mag=disturbance_mag,
        w=w,
    )

    # x[7] = v_y. Positive addition => +y disturbance.
    x_next[7] += dt * d_y

    return x_next, d_y, spatial_scale


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
        Track only terminal position/orientation.  This is used when the
        fixed MPC horizon extends past the end of the offline trajectory so
        the artificial end padding does not make the vehicle brake simply to
        match the final reference velocity/rates.
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
    """
    Build one fixed-size FATROP MPC problem.

    Parameters change at every MPC step, but the NLP structure
    stays fixed, so the solver is constructed only once.
    """
    H = int(horizon)
    K = H + 1

    opti = ca.Opti()

    # Manual FATROP stage dimensions.
    nx = [NX] * K
    nu = [NU] * H + [0]
    ng = [0] * K

    # Stage variables.
    X = []
    U = []

    for j in range(K):
        X.append(
            opti.variable(NX)
        )
        U.append(
            opti.variable(NU if j < H else 0)
        )

    # Time-varying online parameters.
    x_measured = opti.parameter(NX)

    X_ref = [
        opti.parameter(NX)
        for _ in range(K)
    ]

    U_ref = [
        opti.parameter(NU)
        for _ in range(H)
    ]

    dt_ref = [
        opti.parameter()
        for _ in range(H)
    ]

    # 1 for a real offline trajectory interval, 0 for fixed-horizon padding
    # beyond the end of the trajectory.
    stage_active = [
        opti.parameter()
        for _ in range(H)
    ]

    # Full terminal velocity/rate tracking when the prediction horizon lies
    # inside the offline trajectory.  Set to zero when the fixed horizon runs
    # past the final node, so the end of the saved trajectory does not create
    # an artificial braking objective.
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

        # ----------------------------------------------------
        # 1. FATROP dynamics gap MUST be first.
        # ----------------------------------------------------
        if j < H:
            uj = U[j]

            x_next = (
                xj
                + dt_ref[j]
                * quadrotor_xdot_ca(xj, uj)
            )

            opti.subject_to(
                X[j + 1] == x_next
            )

        # ----------------------------------------------------
        # 2. Stage-local constraints after the gap.
        # ----------------------------------------------------
        if j == 0:
            # The current measured state is already fixed and cannot be
            # changed by the optimizer. Do not also impose the nominal
            # state bounds here: a disturbance or round-off-sized overshoot
            # at an active bound would make the entire MPC problem
            # infeasible. Bounds are enforced on predicted states instead.
            opti.subject_to(
                xj == x_measured
            )
            ng[j] += NX
        else:
            add_local_constraint(
                j,
                xj,
                X_MIN,
                X_MAX,
            )

        if j < H:
            add_local_constraint(
                j,
                U[j],
                U_MIN,
                U_MAX,
            )

            # IMPORTANT: padded stages beyond the offline trajectory have
            # stage_active[j] = 0, so they contribute exactly zero tracking
            # cost and therefore cannot make the vehicle brake early.
            objective += stage_active[j] * tracking_stage_cost(
                xj,
                U[j],
                X_ref[j],
                U_ref[j],
            )

        else:
            objective += tracking_terminal_cost(
                xj,
                X_ref[j],
                motion_weight=terminal_motion_weight,
            )

    opti.minimize(objective)

    solver_options = {
        "structure_detection": "manual",
        "nx": nx,
        "nu": nu,
        "ng": ng,
        "N": H,
        "expand": True,
    }

    opti.solver(
        "fatrop",
        solver_options,
    )

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
        raise FileNotFoundError(
            f"Offline GPU-SLS plan not found: {filename}"
        )

    with np.load(filename, allow_pickle=False) as data:
        required = [
            "X",
            "U",
            "phase_times",
            "waypoint_steps",
        ]

        for key in required:
            if key not in data:
                raise ValueError(
                    f"{filename} is missing required key '{key}'"
                )

        X_full = np.asarray(
            data["X"],
            dtype=float,
        )

        U_plan = np.asarray(
            data["U"],
            dtype=float,
        )

        phase_times = np.asarray(
            data["phase_times"],
            dtype=float,
        ).reshape(-1)

        waypoint_steps = np.asarray(
            data["waypoint_steps"],
            dtype=int,
        ).reshape(-1)

        waypoint_centers = (
            np.asarray(
                data["waypoint_centers"],
                dtype=float,
            )
            if "waypoint_centers" in data
            else None
        )

        waypoint_half_widths = (
            np.asarray(
                data["waypoint_half_widths"],
                dtype=float,
            )
            if "waypoint_half_widths" in data
            else None
        )

    if X_full.ndim != 2:
        raise ValueError(
            f"Expected X to be 2D, got {X_full.shape}"
        )

    if X_full.shape[1] < NX:
        raise ValueError(
            f"Expected at least {NX} state columns, got {X_full.shape[1]}"
        )

    # The offline GPU-SLS planner stores
    #   X = [12 physical quadrotor states, T1, ..., T_num_phases].
    # The online tracker only controls/tracks the 12 physical states; the
    # optimized T_i values are converted below into the fixed dt schedule.
    X_plan = X_full[:, :NX]

    N = X_plan.shape[0] - 1

    if U_plan.shape != (N, NU):
        raise ValueError(
            f"U must have shape {(N, NU)}, got {U_plan.shape}"
        )

    if waypoint_steps[-1] != N:
        raise ValueError(
            "Final waypoint step must equal the number "
            "of control intervals."
        )

    segment_lengths = np.concatenate([
        waypoint_steps[:1],
        waypoint_steps[1:] - waypoint_steps[:-1],
    ]).astype(float)

    if len(phase_times) != len(segment_lengths):
        raise ValueError(
            "phase_times and waypoint_steps imply different "
            "numbers of phases."
        )

    # Convert optimized phase times into one dt per control interval.
    dt_plan = np.empty(N, dtype=float)

    for k in range(N):
        phase_idx = int(
            np.searchsorted(
                waypoint_steps,
                k,
                side="right",
            )
        )

        dt_plan[k] = (
            phase_times[phase_idx]
            / segment_lengths[phase_idx]
        )

    return {
        "filename": filename,
        "X_plan": X_plan,
        "U_plan": U_plan,
        "phase_times": phase_times,
        "waypoint_steps": waypoint_steps,
        "waypoint_centers": waypoint_centers,
        "waypoint_half_widths": waypoint_half_widths,
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
    the earlier controls.  The fixed FATROP problem size is preserved.
    """
    H = int(horizon)
    N = len(U_plan)

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
    # Stage 0 is fixed to the measured state and is intentionally not
    # subject to X_MIN/X_MAX.  Do not clip the measurement here: doing so
    # would make the optimizer track a different state than the plant is in.
    x_current = np.asarray(
        x_current,
        dtype=float,
    ).reshape(NX)

    if not np.all(np.isfinite(x_current)):
        raise ValueError("Measured state contains NaN or Inf.")
    opti.set_value(
        mpc["x_measured"],
        x_current,
    )

    for j in range(H + 1):
        opti.set_value(
            mpc["X_ref"][j],
            X_ref_np[j],
        )

    for j in range(H):
        opti.set_value(
            mpc["U_ref"][j],
            U_ref_np[j],
        )
        opti.set_value(
            mpc["dt_ref"][j],
            float(dt_ref_np[j]),
        )
        opti.set_value(
            mpc["stage_active"][j],
            float(stage_active_np[j]),
        )

    opti.set_value(
        mpc["terminal_motion_weight"],
        float(terminal_motion_weight_np),
    )

    # Warm start.
    if X_guess is None:
        X_guess = X_ref_np.copy()

    if U_guess is None:
        U_guess = U_ref_np.copy()

    # Make stage 0 exactly match the measured state in the initial guess.
    X_guess = np.asarray(
        X_guess,
        dtype=float,
    ).copy()
    X_guess[0] = x_current

    for j in range(H + 1):
        opti.set_initial(
            mpc["X"][j],
            X_guess[j],
        )

    for j in range(H):
        opti.set_initial(
            mpc["U"][j],
            U_guess[j],
        )

    tic = time.perf_counter()

    sol = opti.solve()

    solve_time = (
        time.perf_counter()
        - tic
    )

    X_sol = np.stack([
        np.asarray(
            sol.value(mpc["X"][j])
        ).reshape(-1)
        for j in range(H + 1)
    ])

    U_sol = np.stack([
        np.asarray(
            sol.value(mpc["U"][j])
        ).reshape(-1)
        for j in range(H)
    ])

    objective = float(
        sol.value(
            mpc["objective"]
        )
    )

    return (
        X_sol,
        U_sol,
        objective,
        solve_time,
    )


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
    disturbance_mag=DISTURBANCE_MAG,
    disturbance_w=DISTURBANCE_W,
    initial_offset=None,
    progress_search_window=DEFAULT_PROGRESS_SEARCH_WINDOW,
    goal_tol=DEFAULT_GOAL_TOL,
    max_control_steps=None,
):
    """
    Closed-loop FATROP tracking with MONOTONIC nearest-point progress.

    The online controller iteration count and the offline trajectory
    index are deliberately decoupled.

    At every controller update:

        1. Find the nearest point ahead of the previous progress index.
        2. Never allow progress to decrease.
        3. Build the MPC reference from that progress index.
        4. Apply one MPC control.
        5. Repeat until the final path point is reached within goal_tol.

    This means the reference does not keep running away from the
    vehicle simply because wall-clock/controller iterations advance.
    """
    X_plan = plan["X_plan"]
    U_plan = plan["U_plan"]
    dt_plan = plan["dt_plan"]
    N = plan["N"]

    if progress_search_window < 1:
        raise ValueError(
            "progress_search_window must be >= 1"
        )

    if max_control_steps is None:
        max_control_steps = (
            DEFAULT_MAX_CONTROL_STEPS_MULTIPLIER
            * N
        )

    max_control_steps = int(
        max_control_steps
    )

    if max_control_steps < 1:
        raise ValueError(
            "max_control_steps must be >= 1"
        )

    mpc = build_fatrop_mpc(
        horizon=horizon,
    )

    # Allocate for controller iterations, not offline path nodes.
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

    disturbance_history = np.zeros(
        max_control_steps,
        dtype=float,
    )

    disturbance_scale_history = np.zeros(
        max_control_steps,
        dtype=float,
    )

    dt_applied_history = np.zeros(
        max_control_steps,
        dtype=float,
    )

    # One progress index per closed-loop state node.
    progress_idx_history = np.zeros(
        max_control_steps + 1,
        dtype=int,
    )

    nearest_distance_history = np.full(
        max_control_steps + 1,
        np.nan,
        dtype=float,
    )

    # Start at the optimized initial state.
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

    progress_idx_history[0] = (
        progress_idx
    )
    nearest_distance_history[0] = (
        nearest_distance
    )

    X_guess = None
    U_guess = None

    print(
        "\n================ FATROP MPC TRACKING ================"
    )
    print(
        f"Offline GPU-SLS plan: {plan['filename']}"
    )
    print(
        f"Offline plan intervals: {N}"
    )
    print(
        f"MPC horizon: {horizon}"
    )
    print(
        "Progress mode: monotonic nearest point"
    )
    print(
        f"Progress search window: {progress_search_window} nodes"
    )
    print(
        f"Goal tolerance: {goal_tol:.3f} m"
    )
    print(
        f"Max controller steps: {max_control_steps}"
    )
    print(
        f"Plant disturbance magnitude: {disturbance_mag}"
    )
    print(
        f"Disturbance w: {disturbance_w}  (+y direction)"
    )
    print(
        "====================================================="
    )

    num_control_steps = 0
    reached_goal = False
    previous_reference_progress = progress_idx

    for control_step in range(
        max_control_steps
    ):
        # ----------------------------------------------------
        # Determine current monotonic spatial progress.
        # ----------------------------------------------------
        old_progress_idx = progress_idx

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

        # If we are already at the final progress node and physically
        # close enough to the final goal, stop before another solve.
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

        # ----------------------------------------------------
        # Reference horizon starts from SPATIAL progress,
        # not from controller iteration count.
        # ----------------------------------------------------
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
            k=progress_idx,
            horizon=horizon,
        )

        # The physical integration step follows the local reference
        # interval associated with current progress.
        dt_apply = float(
            dt_plan[
                min(
                    progress_idx,
                    N - 1,
                )
            ]
        )

        # ----------------------------------------------------
        # Warm start alignment
        # ----------------------------------------------------
        # A one-node progress advance aligns with the ordinary shifted
        # solution. If progress stalls or jumps by >1, use the new
        # reference window as the warm start instead.
        if (
            control_step > 0
            and (
                progress_idx
                - previous_reference_progress
            ) != 1
        ):
            X_guess = None
            U_guess = None

        try:
            (
                X_mpc,
                U_mpc,
                objective,
                solve_time,
            ) = solve_mpc_step(
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

            u_apply = U_mpc[0]

            objective_history[
                control_step
            ] = objective

            solve_times[
                control_step
            ] = solve_time

            (
                X_guess,
                U_guess,
            ) = shift_warm_start(
                X_mpc,
                U_mpc,
            )

        except RuntimeError as error:
            print(
                f"\nMPC solve failed at controller step "
                f"{control_step}; using offline feedforward "
                f"control at progress node {progress_idx}."
            )
            print(error)

            u_apply = U_ref[0].copy()

            fallback_used[
                control_step
            ] = True

            X_guess = None
            U_guess = None

        previous_reference_progress = (
            progress_idx
        )

        U_closed_loop[
            control_step
        ] = u_apply

        dt_applied_history[
            control_step
        ] = dt_apply

        # ----------------------------------------------------
        # Apply one real plant step with E(x) w, w = +1.
        # ----------------------------------------------------
        (
            x_current,
            disturbance_y,
            disturbance_scale,
        ) = plant_step(
            x=x_current,
            u=u_apply,
            dt=dt_apply,
            disturbance_mag=disturbance_mag,
            w=disturbance_w,
        )

        disturbance_history[
            control_step
        ] = disturbance_y

        disturbance_scale_history[
            control_step
        ] = disturbance_scale

        num_control_steps = (
            control_step + 1
        )

        X_closed_loop[
            num_control_steps
        ] = x_current

        # ----------------------------------------------------
        # Re-evaluate progress after the applied plant step.
        # ----------------------------------------------------
        (
            next_progress_idx,
            next_nearest_distance,
        ) = find_monotonic_progress_index(
            x_current=x_current,
            X_plan=X_plan,
            previous_progress_idx=progress_idx,
            search_window=progress_search_window,
        )

        progress_idx = (
            next_progress_idx
        )

        progress_idx_history[
            num_control_steps
        ] = progress_idx

        nearest_distance_history[
            num_control_steps
        ] = next_nearest_distance

        reference_pos_error = float(
            np.linalg.norm(
                x_current[:3]
                - X_plan[
                    progress_idx,
                    :3,
                ]
            )
        )

        final_pos_error = float(
            np.linalg.norm(
                x_current[:3]
                - X_plan[-1, :3]
            )
        )

        if np.isfinite(
            solve_times[control_step]
        ):
            solve_ms = (
                1000.0
                * solve_times[control_step]
            )
            solve_string = (
                f"{solve_ms:9.3f} ms"
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
            f"d_y={disturbance_y:+.4f}  "
            f"|u|={np.linalg.norm(u_apply):.3f}"
        )

        if (
            progress_idx >= N
            and final_pos_error <= goal_tol
        ):
            reached_goal = True
            break

    # --------------------------------------------------------
    # Trim allocations to actual executed duration.
    # --------------------------------------------------------
    X_closed_loop = X_closed_loop[
        :num_control_steps + 1
    ]

    U_closed_loop = U_closed_loop[
        :num_control_steps
    ]

    objective_history = objective_history[
        :num_control_steps
    ]

    solve_times = solve_times[
        :num_control_steps
    ]

    fallback_used = fallback_used[
        :num_control_steps
    ]

    disturbance_history = disturbance_history[
        :num_control_steps
    ]

    disturbance_scale_history = (
        disturbance_scale_history[
            :num_control_steps
        ]
    )

    dt_applied_history = dt_applied_history[
        :num_control_steps
    ]

    progress_idx_history = (
        progress_idx_history[
            :num_control_steps + 1
        ]
    )

    nearest_distance_history = (
        nearest_distance_history[
            :num_control_steps + 1
        ]
    )

    # Tracking errors are measured against the progress-matched
    # offline state, not against equal controller/path indices.
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

    print(
        "\n================ TRACKING RESULT ====================="
    )
    print(
        f"Reached final goal: {reached_goal}"
    )
    print(
        f"Controller steps executed: {num_control_steps}"
    )
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

    if len(valid_solve_times) > 0:
        print(
            "Mean FATROP MPC solve [ms]: "
            f"{1000.0 * np.mean(valid_solve_times):.3f}"
        )
        print(
            "Max FATROP MPC solve [ms]:  "
            f"{1000.0 * np.max(valid_solve_times):.3f}"
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
        "fallback_used": fallback_used,
        "pos_errors": pos_errors,
        "state_errors": state_errors,
        "disturbance_history": disturbance_history,
        "disturbance_scale_history": disturbance_scale_history,
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
    disturbance_mag=0.0,
    save_path="multiphase_trajectory_tracking_topdown.png",
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    x_min = -6.0
    x_max = 5.0
    y_min = -7.5
    y_max = 8.0

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # Disturbance field background. If disturbance_mag == 0, use
    # 2.5 only for visualization so it still shows the same field
    # geometry as the planner figure.
    display_mag = (
        float(disturbance_mag)
        if disturbance_mag > 0.0
        else 2.5
    )

    xs = np.linspace(
        x_min,
        x_max,
        300,
    )

    ys = np.linspace(
        y_min,
        y_max,
        300,
    )

    XX, YY = np.meshgrid(
        xs,
        ys,
    )

    field = (
        display_mag
        * np.vectorize(
            disturbance_spatial_scale
        )(XX, YY)
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

    contour = ax.contourf(
        XX,
        YY,
        field,
        levels=np.linspace(
            0.0,
            display_mag,
            100,
        ),
        cmap=cmap,
        vmin=0.0,
        vmax=display_mag,
        alpha=0.85,
        zorder=0,
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r"Disturbance magnitude on $v_y$"
    )

    ax.plot(
        X_plan[:, 0],
        X_plan[:, 1],
        "--",
        linewidth=2.0,
        label="Offline GPU-SLS optimized trajectory",
        zorder=4,
    )

    ax.plot(
        X_closed_loop[:, 0],
        X_closed_loop[:, 1],
        "-o",
        markersize=2,
        linewidth=2.0,
        label="Online FATROP MPC rollout",
        zorder=5,
    )

    ax.scatter(
        X_closed_loop[0, 0],
        X_closed_loop[0, 1],
        marker="x",
        s=100,
        linewidths=3,
        label="Start",
        zorder=7,
    )

    if waypoint_centers is not None:
        waypoint_centers = np.asarray(
            waypoint_centers,
            dtype=float,
        )

        ax.scatter(
            waypoint_centers[:, 0],
            waypoint_centers[:, 1],
            marker="s",
            s=90,
            label="Waypoints",
            zorder=7,
        )

        for i, waypoint in enumerate(
            waypoint_centers
        ):
            ax.annotate(
                str(i + 1),
                (
                    waypoint[0],
                    waypoint[1],
                ),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                zorder=8,
            )

    if (
        waypoint_centers is not None
        and waypoint_half_widths is not None
    ):
        waypoint_half_widths = np.asarray(
            waypoint_half_widths,
            dtype=float,
        )

        for center, hw in zip(
            waypoint_centers,
            waypoint_half_widths,
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

    final_error = np.linalg.norm(
        X_closed_loop[-1, :3]
        - X_plan[-1, :3]
    )

    ax.set_title(
        "FATROP Receding-Horizon Tracking\n"
        f"Final position error: {final_error:.3f} m"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.grid(
        True,
        alpha=0.25,
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved top-down tracking plot to: {save_path}"
    )


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
            "Track the saved GPU-SLS/JAX minimum-time trajectory "
            "using online receding-horizon FATROP MPC."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="multiphase_trajectory.npz",
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
            "Magnitude of the always-on spatial disturbance applied "
            "with w=+1 in the +y velocity direction. "
            f"Default is {DISTURBANCE_MAG}."
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
        disturbance_w=DISTURBANCE_W,
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
        objective_history=result["objective_history"],
        fallback_used=result["fallback_used"],
        mpc_horizon=np.array(args.horizon),
        disturbance_mag=np.array(
            args.disturbance_mag
        ),
        disturbance_w=np.array(
            DISTURBANCE_W
        ),
        disturbance_history=result[
            "disturbance_history"
        ],
        disturbance_scale_history=result[
            "disturbance_scale_history"
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