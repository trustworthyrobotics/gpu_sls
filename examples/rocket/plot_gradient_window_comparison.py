#!/usr/bin/env python3
"""Plot the FastSLS rocket gradient-window ablation.

The input archives are produced by rocket.py. The figure overlays both nominal
trajectories and SLS tubes on the exact altitude-dependent disturbance field,
then shows the optimized minimum times side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


REQUIRED_KEYS = {
    "plan",
    "lower",
    "upper",
    "goal_center",
    "goal_half_width",
    "enable_fastsls",
    "gradient_window",
    "min_time",
    "disturbance_scale",
    "disturbance_slope",
    "disturbance_bias",
    "max_dynamics_defect",
    "max_constraint_violation",
    "max_tightened_constraint_violation",
}


def scalar(data: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(data[key]).reshape(()))


def load_result(path: Path, expected_window: int) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rocket result: {path}")

    with np.load(path) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        data = {key: np.asarray(archive[key]) for key in archive.files}

    if not bool(np.asarray(data["enable_fastsls"]).reshape(())):
        raise ValueError(f"{path} was not generated with FastSLS enabled")
    window = int(np.asarray(data["gradient_window"]).reshape(()))
    if window != expected_window:
        raise ValueError(
            f"{path} records gradient_window={window}; expected {expected_window}"
        )

    for key in ("plan", "lower", "upper"):
        value = data[key]
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError(f"{path}: {key} must have shape (N, >=3), got {value.shape}")
    return data


def shared_disturbance_parameters(
    no_gradient: dict[str, np.ndarray],
    gradient_aware: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    keys = ("disturbance_scale", "disturbance_slope", "disturbance_bias")
    first = tuple(scalar(no_gradient, key) for key in keys)
    second = tuple(scalar(gradient_aware, key) for key in keys)
    if not np.allclose(first, second):
        raise ValueError(
            "The two runs use different disturbance profiles; this is not a "
            "controlled gradient-window ablation."
        )
    return first


def plot_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    x_values: list[np.ndarray] = []
    z_values: list[np.ndarray] = []
    for data in datasets:
        for key in ("plan", "lower", "upper"):
            x_values.append(data[key][:, 0])
            z_values.append(data[key][:, 2])
        center = np.asarray(data["goal_center"]).reshape(-1)
        half_width = np.asarray(data["goal_half_width"]).reshape(-1)
        x_values.append(center[0] + np.array([-half_width[0], half_width[0]]))
        z_values.append(center[2] + np.array([-half_width[2], half_width[2]]))

    x_all = np.concatenate(x_values)
    z_all = np.concatenate(z_values)
    x_span = max(float(np.ptp(x_all)), 1.0)
    z_span = max(float(np.ptp(z_all)), 1.0)
    return (
        float(np.min(x_all) - 0.04 * x_span),
        float(np.max(x_all) + 0.04 * x_span),
        float(np.min(z_all) - 0.08 * z_span),
        float(np.max(z_all) + 0.08 * z_span),
    )


def draw_disturbance_field(
    fig: plt.Figure,
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
    parameters: tuple[float, float, float],
) -> None:
    x_min, x_max, z_min, z_max = limits
    scale, slope, bias = parameters
    z = np.linspace(z_min, z_max, 700)
    magnitude = scale * (np.tanh(slope * z + bias) + 1.0)
    field = np.repeat(magnitude[:, None], 2, axis=1)
    image = ax.imshow(
        field,
        extent=(x_min, x_max, z_min, z_max),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        alpha=0.30,
        interpolation="bilinear",
        zorder=-10,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.05)
    colorbar.set_label(r"Disturbance magnitude $E_{mag}(z)$ before $\Delta t$ scaling")

    if slope != 0.0:
        transition = -bias / slope
        ax.axhline(
            transition,
            color="0.25",
            linestyle=":",
            linewidth=1.3,
            label=f"Disturbance transition (z={transition:.2f} m)",
            zorder=1,
        )


def draw_goal(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    center = np.asarray(data["goal_center"]).reshape(-1)
    half_width = np.asarray(data["goal_half_width"]).reshape(-1)
    goal = Rectangle(
        (center[0] - half_width[0], center[2] - half_width[2]),
        2.0 * half_width[0],
        2.0 * half_width[2],
        facecolor="gold",
        edgecolor="darkgoldenrod",
        hatch="//",
        alpha=0.28,
        linewidth=1.3,
        label="Goal region",
        zorder=1,
    )
    ax.add_patch(goal)


def draw_result(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    *,
    color: str,
    label: str,
) -> None:
    plan = data["plan"]
    lower = data["lower"]
    upper = data["upper"]
    stride = max(1, int(np.asarray(data.get("tube_stride", 2)).reshape(())))

    order = np.argsort(plan[:, 0])
    ax.fill_between(
        plan[order, 0],
        lower[order, 2],
        upper[order, 2],
        color=color,
        alpha=0.13,
        linewidth=0.0,
        zorder=2,
    )

    indices = list(range(0, len(plan), stride))
    if indices[-1] != len(plan) - 1:
        indices.append(len(plan) - 1)
    for k in indices:
        ax.add_patch(
            Rectangle(
                (lower[k, 0], lower[k, 2]),
                max(float(upper[k, 0] - lower[k, 0]), 0.0),
                max(float(upper[k, 2] - lower[k, 2]), 0.0),
                facecolor=color,
                edgecolor=color,
                alpha=0.10,
                linewidth=0.7,
                zorder=3,
            )
        )

    min_time = scalar(data, "min_time")
    ax.plot(
        plan[:, 0],
        plan[:, 2],
        color=color,
        linewidth=2.6,
        label=f"{label}: {min_time:.3f} s",
        zorder=5,
    )
    ax.scatter(plan[0, 0], plan[0, 2], color=color, s=38, zorder=6)
    ax.scatter(plan[-1, 0], plan[-1, 2], color=color, marker="X", s=55, zorder=6)


def make_comparison(
    no_gradient_path: Path,
    gradient_aware_path: Path,
    output_path: Path,
    *,
    show: bool = False,
) -> tuple[float, float]:
    no_gradient = load_result(no_gradient_path, expected_window=0)
    gradient_aware = load_result(gradient_aware_path, expected_window=10)
    parameters = shared_disturbance_parameters(no_gradient, gradient_aware)
    limits = plot_limits(no_gradient, gradient_aware)

    no_gradient_time = scalar(no_gradient, "min_time")
    gradient_aware_time = scalar(gradient_aware, "min_time")
    improvement = no_gradient_time - gradient_aware_time
    improvement_percent = 100.0 * improvement / no_gradient_time

    fig, (trajectory_ax, time_ax) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        gridspec_kw={"width_ratios": (3.2, 1.0)},
    )
    draw_disturbance_field(fig, trajectory_ax, limits, parameters)
    draw_goal(trajectory_ax, no_gradient)
    draw_result(
        trajectory_ax,
        no_gradient,
        color="#d62728",
        label="No gradient (window 0)",
    )
    draw_result(
        trajectory_ax,
        gradient_aware,
        color="#2ca02c",
        label="Gradient-aware (window 10)",
    )
    trajectory_ax.set_xlim(limits[:2])
    trajectory_ax.set_ylim(limits[2:])
    trajectory_ax.set_xlabel("Horizontal position x [m]")
    trajectory_ax.set_ylabel("Altitude z [m]")
    trajectory_ax.set_title("FastSLS path choice in the disturbance field")
    trajectory_ax.grid(True, alpha=0.22)
    handles, labels = trajectory_ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    trajectory_ax.legend(unique.values(), unique.keys(), loc="best")

    bars = time_ax.bar(
        ["Window 0", "Window 10"],
        [no_gradient_time, gradient_aware_time],
        color=["#d62728", "#2ca02c"],
        alpha=0.85,
    )
    time_ax.set_ylabel("Optimized minimum time [s]")
    time_ax.set_title(f"Reduction: {improvement:.3f} s\n({improvement_percent:.1f}%)")
    time_ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, (no_gradient_time, gradient_aware_time)):
        time_ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f} s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle("Rocket gradient-window ablation", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Window 0 minimum time:  {no_gradient_time:.9g} s")
    print(f"Window 10 minimum time: {gradient_aware_time:.9g} s")
    print(f"Minimum-time reduction: {improvement:.9g} s ({improvement_percent:.3f}%)")
    print(f"Saved comparison: {output_path.resolve()}")
    return no_gradient_time, gradient_aware_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gradient", type=Path, required=True)
    parser.add_argument("--gradient-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    make_comparison(
        args.no_gradient,
        args.gradient_aware,
        args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
