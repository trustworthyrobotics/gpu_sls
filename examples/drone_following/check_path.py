from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ============================================================
# Offline discrete-node gate collision checker
#
# This script:
#   1. Loads the saved offline trajectory NPZ.
#   2. Reconstructs the exact inflated four-bar gate geometry.
#   3. Checks every discrete trajectory node X[k, :3].
#   4. Reports every node lying inside an inflated gate bar.
#
# IMPORTANT:
#   - NO swept-segment collision checking is performed.
#   - Only X[k, :3] is checked.
#   - The gate bars are inflated by drone_radius, exactly like
#     the MPC tracker.
#
# State convention:
#   X[:, 0:3] = [px, py, pz]
# ============================================================


DEFAULT_COLLISION_TOL = 0.0


# ============================================================
# Load offline trajectory
# ============================================================

def load_offline_plan(filename):
    filename = Path(filename).expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(
            f"Offline trajectory not found: {filename}"
        )

    with np.load(filename, allow_pickle=False) as data:
        required = (
            "X",
            "gate_centers",
            "gate_normals",
            "gate_axis_u",
            "gate_axis_v",
            "gate_opening_half_widths",
        )

        for key in required:
            if key not in data:
                raise ValueError(
                    f"{filename} is missing required key '{key}'"
                )

        X = np.asarray(
            data["X"],
            dtype=float,
        )

        gate_centers = np.asarray(
            data["gate_centers"],
            dtype=float,
        )

        gate_normals = np.asarray(
            data["gate_normals"],
            dtype=float,
        )

        gate_axis_u = np.asarray(
            data["gate_axis_u"],
            dtype=float,
        )

        gate_axis_v = np.asarray(
            data["gate_axis_v"],
            dtype=float,
        )

        gate_opening_half_widths = np.asarray(
            data["gate_opening_half_widths"],
            dtype=float,
        )

        gate_bar_thickness = (
            float(np.asarray(data["gate_bar_thickness"]))
            if "gate_bar_thickness" in data
            else 0.10
        )

        gate_depth = (
            float(np.asarray(data["gate_depth"]))
            if "gate_depth" in data
            else 0.10
        )

        drone_radius = (
            float(np.asarray(data["drone_radius"]))
            if "drone_radius" in data
            else 0.10
        )

    if X.ndim != 2:
        raise ValueError(
            f"Expected X to be 2-D, got shape {X.shape}"
        )

    if X.shape[1] < 3:
        raise ValueError(
            f"Expected X to contain XYZ position, got shape {X.shape}"
        )

    return {
        "filename": filename,
        "X": X,
        "gate_centers": gate_centers,
        "gate_normals": gate_normals,
        "gate_axis_u": gate_axis_u,
        "gate_axis_v": gate_axis_v,
        "gate_opening_half_widths": gate_opening_half_widths,
        "gate_bar_thickness": gate_bar_thickness,
        "gate_depth": gate_depth,
        "drone_radius": drone_radius,
    }


# ============================================================
# Construct exact inflated gate-bar geometry
# ============================================================

def build_gate_bar_geometry_np(plan):
    """
    Build the same inflated four-bar gate geometry as the MPC tracker.

    Each physical gate contains four oriented boxes:

        0: top
        1: bottom
        2: left
        3: right

    The boxes are inflated by `drone_radius`, so the drone center can
    be treated as a point.
    """

    gate_centers = np.asarray(
        plan["gate_centers"],
        dtype=float,
    )

    gate_normals = np.asarray(
        plan["gate_normals"],
        dtype=float,
    )

    gate_axis_u = np.asarray(
        plan["gate_axis_u"],
        dtype=float,
    )

    gate_axis_v = np.asarray(
        plan["gate_axis_v"],
        dtype=float,
    )

    opening_half_widths = np.asarray(
        plan["gate_opening_half_widths"],
        dtype=float,
    )

    num_gates = gate_centers.shape[0]

    hu = opening_half_widths[:, 0]
    hv = opening_half_widths[:, 1]

    t_bar = float(
        plan["gate_bar_thickness"]
    )

    half_depth = (
        0.5
        * float(plan["gate_depth"])
    )

    radius = float(
        plan["drone_radius"]
    )

    zeros = np.zeros(
        num_gates,
        dtype=float,
    )

    # --------------------------------------------------------
    # Bar centers in each gate's local [u, v, n] coordinates.
    # --------------------------------------------------------

    top_center_local = np.stack(
        [
            zeros,
            hv + 0.5 * t_bar,
            zeros,
        ],
        axis=1,
    )

    bottom_center_local = np.stack(
        [
            zeros,
            -(hv + 0.5 * t_bar),
            zeros,
        ],
        axis=1,
    )

    left_center_local = np.stack(
        [
            -(hu + 0.5 * t_bar),
            zeros,
            zeros,
        ],
        axis=1,
    )

    right_center_local = np.stack(
        [
            hu + 0.5 * t_bar,
            zeros,
            zeros,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Inflated half sizes.
    #
    # This is the same Minkowski inflation used in the MPC:
    #
    #     physical bar half size + drone_radius
    #
    # along every local box axis.
    # --------------------------------------------------------

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

    top_centers = local_to_world(
        top_center_local
    )

    bottom_centers = local_to_world(
        bottom_center_local
    )

    left_centers = local_to_world(
        left_center_local
    )

    right_centers = local_to_world(
        right_center_local
    )

    # Shape:
    #
    #     centers[gate, bar, xyz]
    #     half_sizes[gate, bar, local_axis]
    #

    centers = np.stack(
        [
            top_centers,
            bottom_centers,
            left_centers,
            right_centers,
        ],
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

    axis_u = np.repeat(
        gate_axis_u[:, None, :],
        4,
        axis=1,
    )

    axis_v = np.repeat(
        gate_axis_v[:, None, :],
        4,
        axis=1,
    )

    normals = np.repeat(
        gate_normals[:, None, :],
        4,
        axis=1,
    )

    return {
        "centers": centers,
        "half_sizes": half_sizes,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "normals": normals,
        "bar_names": (
            "top",
            "bottom",
            "left",
            "right",
        ),
    }


# ============================================================
# Point-vs-oriented-box collision
# ============================================================

def check_gate_point_collision_np(
    pos,
    gate_bar_geometry,
    tol=DEFAULT_COLLISION_TOL,
):
    """
    Check one discrete XYZ position against every inflated gate bar.

    No segment checking is performed.

    For each oriented box, transform the point into the gate's local
    [u, v, n] coordinates.

    A point lies inside the box when

        |q_u| <= h_u
        |q_v| <= h_v
        |q_n| <= h_n

    Define

        penetration_margin =
            min(h - |q|)

    Therefore:

        penetration_margin > 0
            point is inside the inflated obstacle

        penetration_margin = 0
            point is exactly on the boundary

        penetration_margin < 0
            point is outside

    With tol > 0, penetration must exceed tol before the node is
    reported as a collision. This is useful for ignoring very small
    numerical penetrations.
    """

    pos = np.asarray(
        pos,
        dtype=float,
    ).reshape(3)

    centers = gate_bar_geometry["centers"]
    half_sizes = gate_bar_geometry["half_sizes"]
    axis_u = gate_bar_geometry["axis_u"]
    axis_v = gate_bar_geometry["axis_v"]
    normals = gate_bar_geometry["normals"]

    best_gate = -1
    best_bar = -1
    best_margin = -np.inf

    for gate_idx in range(
        centers.shape[0]
    ):
        for bar_idx in range(
            centers.shape[1]
        ):
            delta = (
                pos
                - centers[
                    gate_idx,
                    bar_idx,
                ]
            )

            q = np.array(
                [
                    np.dot(
                        delta,
                        axis_u[
                            gate_idx,
                            bar_idx,
                        ],
                    ),
                    np.dot(
                        delta,
                        axis_v[
                            gate_idx,
                            bar_idx,
                        ],
                    ),
                    np.dot(
                        delta,
                        normals[
                            gate_idx,
                            bar_idx,
                        ],
                    ),
                ],
                dtype=float,
            )

            margins = (
                half_sizes[
                    gate_idx,
                    bar_idx,
                ]
                - np.abs(q)
            )

            penetration_margin = float(
                np.min(margins)
            )

            # Match the tracker collision condition.
            if penetration_margin > float(tol):
                if (
                    best_gate < 0
                    or penetration_margin > best_margin
                ):
                    best_gate = gate_idx
                    best_bar = bar_idx
                    best_margin = penetration_margin

    collision = (
        best_gate >= 0
    )

    return (
        collision,
        best_gate,
        best_bar,
        (
            best_margin
            if collision
            else np.nan
        ),
    )


# ============================================================
# Check every offline node
# ============================================================

def check_offline_trajectory(
    plan,
    collision_tol=DEFAULT_COLLISION_TOL,
):
    X = plan["X"]

    geometry = build_gate_bar_geometry_np(
        plan
    )

    bar_names = geometry[
        "bar_names"
    ]

    collisions = []

    print(
        "\n"
        "================ OFFLINE COLLISION CHECK ================"
    )

    print(
        f"Trajectory: {plan['filename']}"
    )

    print(
        f"Number of discrete states: {len(X)}"
    )

    print(
        f"Number of intervals: {len(X) - 1}"
    )

    print(
        f"Number of gates: "
        f"{geometry['centers'].shape[0]}"
    )

    print(
        f"Gate bar thickness: "
        f"{plan['gate_bar_thickness']:.6f} m"
    )

    print(
        f"Gate depth: "
        f"{plan['gate_depth']:.6f} m"
    )

    print(
        f"Drone radius inflation: "
        f"{plan['drone_radius']:.6f} m"
    )

    print(
        f"Collision tolerance: "
        f"{collision_tol:.6e} m"
    )

    print(
        "Collision mode: DISCRETE NODES ONLY"
    )

    print(
        "Swept segments: DISABLED"
    )

    print(
        "=========================================================="
    )

    for k in range(
        X.shape[0]
    ):
        pos = X[
            k,
            :3,
        ]

        (
            collision,
            gate_idx,
            bar_idx,
            penetration_margin,
        ) = check_gate_point_collision_np(
            pos,
            geometry,
            tol=collision_tol,
        )

        if collision:
            collision_info = {
                "node": k,
                "position": pos.copy(),
                "gate_index": gate_idx,
                "bar_index": bar_idx,
                "bar_name": bar_names[
                    bar_idx
                ],
                "penetration_margin": penetration_margin,
            }

            collisions.append(
                collision_info
            )

            print(
                f"\nCOLLISION:"
                f"\n  node               = {k}"
                f"\n  position           = "
                f"[{pos[0]:+.6f}, "
                f"{pos[1]:+.6f}, "
                f"{pos[2]:+.6f}]"
                f"\n  gate               = {gate_idx + 1}"
                f"\n  bar                = {bar_names[bar_idx]}"
                f"\n  penetration margin = "
                f"{penetration_margin:.9f} m"
            )

    print(
        "\n"
        "====================== RESULT ============================"
    )

    if len(collisions) == 0:
        print(
            "PASS: No discrete offline trajectory nodes collide "
            "with any inflated gate bar."
        )
    else:
        print(
            "FAIL: Discrete offline trajectory contains "
            f"{len(collisions)} colliding node(s)."
        )

        collision_nodes = [
            item["node"]
            for item in collisions
        ]

        collision_gates = sorted(
            set(
                item["gate_index"] + 1
                for item in collisions
            )
        )

        worst_collision = max(
            collisions,
            key=lambda item: item[
                "penetration_margin"
            ],
        )

        print(
            f"Colliding nodes: {collision_nodes}"
        )

        print(
            f"Gates involved: {collision_gates}"
        )

        print(
            "Maximum penetration:"
            f"\n  node   = {worst_collision['node']}"
            f"\n  gate   = {worst_collision['gate_index'] + 1}"
            f"\n  bar    = {worst_collision['bar_name']}"
            f"\n  margin = "
            f"{worst_collision['penetration_margin']:.9f} m"
        )

    print(
        "=========================================================="
    )

    return collisions


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check every discrete node of a saved offline trajectory "
            "against the inflated physical gate-bar obstacles."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="multiphase_trajectory.npz",
        help=(
            "Offline trajectory NPZ "
            "(default: multiphase_trajectory.npz)"
        ),
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_COLLISION_TOL,
        help=(
            "Allowed penetration tolerance. A collision is reported "
            "only when penetration_margin > tol. "
            f"Default: {DEFAULT_COLLISION_TOL}."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.tol < 0.0:
        raise ValueError(
            "--tol must be >= 0"
        )

    plan = load_offline_plan(
        args.input
    )

    collisions = check_offline_trajectory(
        plan,
        collision_tol=args.tol,
    )

    # Exit status:
    #
    #   0 = collision free
    #   1 = at least one discrete-node collision
    #
    # This makes the script easy to use from shell scripts / CI.
    if len(collisions) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()