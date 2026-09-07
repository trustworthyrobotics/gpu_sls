#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon


# ============================================================
# Configuration
# ============================================================

RESULT_DIR = Path("e_mag_sweep")

E_MAGS = [0, 1, 2.5, 5, 7.5, 10, 15]

OUTPUT_PATH = "all_multiphase_topdown.png"


# Same disturbance-field parameters as planner
X_FIELD_MIN = -2.0
X_FIELD_MAX = 1.0

Y_FIELD_MIN = -8.0
Y_FIELD_MAX = 0.0

KX = 1.0
KY = 1.0


# Same plot limits as plot_top_down_trajectory()
X_MIN = -6
X_MAX = 5
Y_MIN = -7.5
Y_MAX = 8


# ============================================================
# Disturbance field
# ============================================================

def disturbance_spatial_scale(px, py):

    x_window = (
        0.5 * (1.0 + np.tanh(KX * (px - X_FIELD_MIN)))
        *
        0.5 * (1.0 - np.tanh(KX * (px - X_FIELD_MAX)))
    )

    y_window = (
        0.5 * (1.0 + np.tanh(KY * (py - Y_FIELD_MIN)))
        *
        0.5 * (1.0 - np.tanh(KY * (py - Y_FIELD_MAX)))
    )

    return x_window * y_window


# ============================================================
# Helpers
# ============================================================

def emag_to_tag(e_mag):
    return str(e_mag).replace(".", "p")


def load_run(e_mag):

    tag = emag_to_tag(e_mag)

    path = (
        RESULT_DIR
        / f"E_mag_{tag}"
        / "multiphase_trajectory.npz"
    )

    if not path.exists():
        print(f"WARNING: missing {path}")
        return None

    data = np.load(path)

    return {
        "e_mag": e_mag,
        "X": np.asarray(data["X"]),
        "reference": np.asarray(data["reference"]),
        "min_time": float(np.asarray(data["min_time"])),

        "waypoint_centers": np.asarray(
            data["waypoint_centers"]
        ),
        "waypoint_half_widths": np.asarray(
            data["waypoint_half_widths"]
        ),
        "waypoint_axis_u": np.asarray(
            data["waypoint_axis_u"]
        ),
        "waypoint_normals": np.asarray(
            data["waypoint_normals"]
        ),

        "gate_centers": np.asarray(
            data["gate_centers"]
        ),
        "gate_normals": np.asarray(
            data["gate_normals"]
        ),
        "gate_axis_u": np.asarray(
            data["gate_axis_u"]
        ),
        "gate_opening_half_widths": np.asarray(
            data["gate_opening_half_widths"]
        ),

        "gate_bar_thickness": float(
            np.asarray(data["gate_bar_thickness"])
        ),
        "gate_depth": float(
            np.asarray(data["gate_depth"])
        ),
        "drone_radius": float(
            np.asarray(data["drone_radius"])
        ),
    }


# ============================================================
# Draw physical gates
# ============================================================

def draw_gates(ax, course):

    gates = course["gate_centers"]
    gate_n = course["gate_normals"]
    gate_u = course["gate_axis_u"]
    gate_openings = course["gate_opening_half_widths"]

    gate_bar_thickness = course["gate_bar_thickness"]
    gate_depth = course["gate_depth"]

    for i, center in enumerate(gates):

        u_xy = np.asarray(
            gate_u[i, :2],
            dtype=float,
        )

        n_xy = np.asarray(
            gate_n[i, :2],
            dtype=float,
        )

        u_xy = u_xy / max(
            np.linalg.norm(u_xy),
            1e-12,
        )

        n_xy = n_xy / max(
            np.linalg.norm(n_xy),
            1e-12,
        )

        hu_open = float(
            gate_openings[i, 0]
        )

        half_outer_width = (
            hu_open
            + gate_bar_thickness
        )

        half_depth = (
            0.5 * gate_depth
        )

        c = np.asarray(
            center[:2],
            dtype=float,
        )

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
            (
                center[0],
                center[1],
            ),
            xytext=(5, -14),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=9,
        )


# ============================================================
# Main
# ============================================================

def main():

    runs = []

    for e_mag in E_MAGS:

        run = load_run(e_mag)

        if run is not None:
            runs.append(run)

            print(
                f"E_MAG={e_mag:<5} "
                f"time={run['min_time']:.4f} s"
            )

    if not runs:
        raise RuntimeError(
            "No trajectory files found."
        )

    # Course geometry should be identical across runs.
    course = runs[0]

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # ========================================================
    # Disturbance field
    #
    # Use the largest E_MAG for the background so the spatial
    # disturbance region remains visible in a single shared plot.
    # ========================================================

    display_e_mag = max(E_MAGS)

    xs = np.linspace(
        X_MIN,
        X_MAX,
        300,
    )

    ys = np.linspace(
        Y_MIN,
        Y_MAX,
        300,
    )

    XX, YY = np.meshgrid(
        xs,
        ys,
    )

    spatial_scale = (
        disturbance_spatial_scale(
            XX,
            YY,
        )
    )

    disturbance_field_values = (
        display_e_mag
        * spatial_scale
    )

    magma = plt.get_cmap(
        "magma"
    )

    disturbance_cmap = (
        LinearSegmentedColormap.from_list(
            "white_to_magma",
            [
                (0.00, "white"),
                (0.08, magma(0.18)),
                (0.35, magma(0.40)),
                (0.65, magma(0.68)),
                (1.00, magma(1.00)),
            ],
        )
    )

    field_levels = np.linspace(
        0.0,
        float(display_e_mag),
        100,
    )

    field = ax.contourf(
        XX,
        YY,
        disturbance_field_values,
        levels=field_levels,
        cmap=disturbance_cmap,
        vmin=0.0,
        vmax=float(display_e_mag),
        alpha=0.85,
        zorder=0,
    )

    cbar = fig.colorbar(
        field,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Disturbance magnitude on $v_y$"
    )

    # ========================================================
    # Reference trajectory
    # ========================================================

    reference = course["reference"]

    ax.plot(
        reference[:, 0],
        reference[:, 1],
        "--",
        linewidth=1.5,
        color="gray",
        label="Reference",
        zorder=3,
    )

    # ========================================================
    # Plot all optimized trajectories
    # ========================================================

    cmap = plt.get_cmap("viridis")

    for idx, run in enumerate(runs):

        X = run["X"]

        if len(runs) > 1:
            color = cmap(
                idx / (len(runs) - 1)
            )
        else:
            color = cmap(0.5)

        ax.plot(
            X[:, 0],
            X[:, 1],
            "-o",
            markersize=2,
            linewidth=2,
            color=color,
            label=(
                f"E_MAG={run['e_mag']}, "
                f"T={run['min_time']:.3f}s"
            ),
            zorder=5,
        )

    # ========================================================
    # Start
    # ========================================================

    X0 = course["X"][0]

    ax.scatter(
        X0[0],
        X0[1],
        s=100,
        marker="x",
        linewidths=3,
        color="black",
        label="Start",
        zorder=8,
    )

    # ========================================================
    # Physical gates
    # ========================================================

    draw_gates(
        ax,
        course,
    )

    # ========================================================
    # Waypoints
    # ========================================================

    waypoints = (
        course["waypoint_centers"]
    )

    ax.scatter(
        waypoints[:, 0],
        waypoints[:, 1],
        s=45,
        marker="s",
        facecolors="none",
        edgecolors="black",
        zorder=7,
    )

    # ========================================================
    # Formatting
    # ========================================================

    ax.set_xlim(
        X_MIN,
        X_MAX,
    )

    ax.set_ylim(
        Y_MIN,
        Y_MAX,
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        "Top-Down Minimum-Time Trajectories "
        "Across Disturbance Levels"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(
        f"Saved combined plot to: "
        f"{OUTPUT_PATH}"
    )

    # plt.show()


if __name__ == "__main__":
    main()