from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def get_trajectory_tubes(Phi_x):
    return jnp.linalg.norm(Phi_x, ord=2, axis=-1).sum(axis=1)


JOINT_NAMES = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)

DEFAULT_PHASE_NAMES = (
    "stance",
    "lateral_launch",
    "flight",
    "prelanding",
    "touchdown",
    "settle",
)


def build_state_labels(n_state: int, phase_names=DEFAULT_PHASE_NAMES):
    """Labels for the Go2 state used by the barrel-roll experiment."""
    labels = [
        "base_x [m]",
        "base_y [m]",
        "base_z [m]",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    ]

    labels += [f"{name}_q [rad]" for name in JOINT_NAMES]
    labels += [
        "base_vx [m/s]",
        "base_vy [m/s]",
        "base_vz [m/s]",
        "base_wx [rad/s]",
        "base_wy [rad/s]",
        "base_wz [rad/s]",
    ]
    labels += [f"{name}_dq [rad/s]" for name in JOINT_NAMES]

    for leg in ("FL", "FR", "RL", "RR"):
        labels += [
            f"{leg}_foot_x [m]",
            f"{leg}_foot_y [m]",
            f"{leg}_foot_z [m]",
        ]

    for leg in ("FL", "FR", "RL", "RR"):
        labels += [
            f"{leg}_grf_x [N]",
            f"{leg}_grf_y [N]",
            f"{leg}_grf_z [N]",
        ]

    labels += [f"T_{name} [s]" for name in phase_names]

    # Keep the script usable if the state layout changes.
    if len(labels) < n_state:
        labels += [f"state_{i}" for i in range(len(labels), n_state)]

    return labels[:n_state]


def sanitize_filename(name: str) -> str:
    keep = []
    for c in name:
        if c.isalnum() or c in ("-", "_"):
            keep.append(c)
        elif c in (" ", "/", "[", "]"):
            keep.append("_")
    return "".join(keep).strip("_")


def load_experiment(path: Path):
    data = np.load(path, allow_pickle=True)

    required = ("X", "Phi_x")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}. "
            "The solve script must save Phi_x in addition to X."
        )

    X = np.asarray(data["X"])
    Phi_x = jnp.asarray(data["Phi_x"])
    tubes = np.asarray(get_trajectory_tubes(Phi_x))

    phase_names = tuple(
        str(x) for x in data["phase_names"]
    ) if "phase_names" in data else DEFAULT_PHASE_NAMES

    if "node_times" in data:
        node_times = np.asarray(data["node_times"], dtype=float)
    else:
        node_times = np.arange(X.shape[0], dtype=float)

    return data, X, tubes, node_times, phase_names


def align_trajectory_and_tubes(X, tubes, node_times):
    """
    Make the nominal trajectory and tube sequence use the same number of nodes.

    Common SLS layouts give either:
      X.shape[0]      tube nodes, or
      X.shape[0] - 1  tube nodes.
    """
    if tubes.ndim != 2:
        raise ValueError(
            "Expected get_trajectory_tubes(Phi_x) to return a 2-D array "
            f"(time, state), got shape {tubes.shape}."
        )

    if tubes.shape[1] != X.shape[1]:
        raise ValueError(
            f"Tube state dimension {tubes.shape[1]} does not match "
            f"trajectory state dimension {X.shape[1]}."
        )

    if tubes.shape[0] == X.shape[0]:
        return X, tubes, node_times[: X.shape[0]]

    if tubes.shape[0] == X.shape[0] - 1:
        return X[:-1], tubes, node_times[: tubes.shape[0]]

    raise ValueError(
        "Tube horizon does not match trajectory horizon: "
        f"X={X.shape}, tubes={tubes.shape}."
    )


def add_phase_boundaries(ax, data, node_times):
    if "phase_end_steps" not in data:
        return

    end_steps = np.asarray(data["phase_end_steps"], dtype=int)
    valid = end_steps[(end_steps >= 0) & (end_steps < len(node_times))]

    for step in valid[:-1]:
        ax.axvline(node_times[step], linestyle="--", linewidth=0.8, alpha=0.45)


def plot_state_tubes(
    result_path: Path,
    output_dir: Path,
    *,
    show: bool = False,
):
    data, X, tubes, node_times, phase_names = load_experiment(result_path)
    X_plot, tube_plot, t_plot = align_trajectory_and_tubes(X, tubes, node_times)

    labels = build_state_labels(X_plot.shape[1], phase_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded X:       {X.shape}")
    print(f"Loaded Phi_x:   {data['Phi_x'].shape}")
    print(f"Computed tubes: {tubes.shape}")
    print(f"Saving {X_plot.shape[1]} state-tube plots to {output_dir}")

    saved_paths = []

    for i, label in enumerate(labels):
        nominal = X_plot[:, i]
        radius = tube_plot[:, i]
        lower = nominal - radius
        upper = nominal + radius

        fig, ax = plt.subplots(figsize=(9, 4.5))

        line, = ax.plot(t_plot, nominal, linewidth=1.8, label="Nominal")
        ax.fill_between(
            t_plot,
            lower,
            upper,
            alpha=0.25,
            color=line.get_color(),
            label="SLS tube",
        )

        add_phase_boundaries(ax, data, t_plot)

        ax.set_xlabel("Time [s]" if "node_times" in data else "Node")
        ax.set_ylabel(label)
        ax.set_title(f"Trajectory tube — {label}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        filename = f"{i:03d}_{sanitize_filename(label)}.png"
        figure_path = output_dir / filename
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        saved_paths.append(figure_path)

        if show:
            plt.show()
        plt.close(fig)

    # Also save one compact diagnostic containing only tube radii.
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(
        tube_plot.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t_plot[0], t_plot[-1], -0.5, X_plot.shape[1] - 0.5],
    )
    ax.set_xlabel("Time [s]" if "node_times" in data else "Node")
    ax.set_ylabel("State dimension")
    ax.set_title("SLS tube radius by state dimension")
    fig.colorbar(image, ax=ax, label="Tube radius")
    fig.tight_layout()

    heatmap_path = output_dir / "tube_radius_heatmap.png"
    fig.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved heatmap: {heatmap_path}")
    print(f"Maximum tube radius: {np.max(tube_plot):.6e}")
    max_index = np.unravel_index(np.argmax(tube_plot), tube_plot.shape)
    print(
        "Largest tube entry: "
        f"time index={max_index[0]}, state={max_index[1]} "
        f"({labels[max_index[1]]}), radius={tube_plot[max_index]:.6e}"
    )

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot SLS trajectory tubes for every state dimension."
    )
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("quadruped_barrel_roll_obstacle_min_time.npz"),
        help="NPZ produced by the barrel-roll experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tube_plots"),
        help="Directory in which individual tube plots are saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each plot interactively as it is generated.",
    )
    args = parser.parse_args()

    plot_state_tubes(
        args.result,
        args.output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
