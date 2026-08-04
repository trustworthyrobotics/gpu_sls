#!/usr/bin/env python3
"""
Compare side-view trajectory tubes from:

    quadrotor_side_view_data_no_grad.npz
    quadrotor_side_view_data_grad.npz

No-gradient trajectory/tube: red
Gradient trajectory/tube: green
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle, Rectangle


REQUIRED_KEYS = {
    "plan",
    "lower",
    "upper",
    "centers",
    "radii",
    "goal_center",
    "goal_half_width",
}


def disturbance_magnitude(z: np.ndarray) -> np.ndarray:
    """Implemented unscaled disturbance magnitude E_mag(z).

    This matches:
        E_mag = 3 * (tanh(6*z - 2.5) + 1)

    The disturbance matrix entry used by the controller is
    (final_time / N) * E_mag.
    """
    z = np.asarray(z, dtype=float)
    return 3.0 * (np.tanh(6.0 * z - 2.0) + 1.0)


def combined_plot_limits(*datasets: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Compute common x-z limits from trajectories, tubes, goals, and obstacles."""
    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []

    for data in datasets:
        xs.extend([data["plan"][:, 0], data["lower"][:, 0], data["upper"][:, 0]])
        zs.extend([data["plan"][:, 2], data["lower"][:, 2], data["upper"][:, 2]])

        goal_center = np.asarray(data["goal_center"]).reshape(-1)
        goal_half_width = np.asarray(data["goal_half_width"]).reshape(-1)
        xs.append(np.array([goal_center[0] - goal_half_width[0], goal_center[0] + goal_half_width[0]]))
        zs.append(np.array([goal_center[2] - goal_half_width[2], goal_center[2] + goal_half_width[2]]))

        centers = np.atleast_2d(data["centers"])
        radii = np.atleast_1d(data["radii"])
        for center, radius in zip(centers, radii):
            xs.append(np.array([center[0] - radius, center[0] + radius]))
            zs.append(np.array([center[2] - radius, center[2] + radius]))

        ground = scalar_from_data(data, "ground_height", 0.0)
        zs.append(np.array([ground]))

    x_all = np.concatenate(xs)
    z_all = np.concatenate(zs)
    x_span = max(float(np.ptp(x_all)), 1.0)
    z_span = max(float(np.ptp(z_all)), 1.0)

    return (
        float(np.min(x_all) - 0.05 * x_span),
        float(np.max(x_all) + 0.05 * x_span),
        float(np.min(z_all) - 0.10 * z_span),
        float(np.max(z_all) + 0.10 * z_span),
    )


def draw_disturbance_gradient(
    fig: plt.Figure,
    ax: plt.Axes,
    limits: tuple[float, float, float, float],
) -> None:
    """Draw altitude-dependent disturbance magnitude behind the trajectories."""
    x_min, x_max, z_min, z_max = limits
    z_grid = np.linspace(z_min, z_max, 600)
    magnitude = disturbance_magnitude(z_grid)

    # Repeat the 1-D altitude profile horizontally to create a background field.
    field = np.repeat(magnitude[:, None], 2, axis=1)
    image = ax.imshow(
        field,
        extent=[x_min, x_max, z_min, z_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        alpha=0.24,
        interpolation="bilinear",
        zorder=-10,
    )

    # Tie the colorbar axes to the plot axes so equal-aspect and layout changes
    # keep both axes exactly the same height.
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="3%", pad=0.12)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label(r"Disturbance magnitude $E_{\mathrm{mag}}(z)$ before $\Delta t$ scaling")



def load_side_view_data(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find NPZ file: {path}")

    with np.load(path) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise KeyError(
                f"{path} is missing required keys: {sorted(missing)}. "
                f"Available keys: {archive.files}"
            )

        data = {key: np.asarray(archive[key]) for key in archive.files}

    plan = data["plan"]
    lower = data["lower"]
    upper = data["upper"]

    if plan.ndim != 2 or plan.shape[1] < 3:
        raise ValueError(f"{path}: plan must have shape (T, >=3), got {plan.shape}")
    if lower.ndim != 2 or lower.shape[1] < 3:
        raise ValueError(f"{path}: lower must have shape (T, >=3), got {lower.shape}")
    if upper.ndim != 2 or upper.shape[1] < 3:
        raise ValueError(f"{path}: upper must have shape (T, >=3), got {upper.shape}")
    if not (plan.shape[0] == lower.shape[0] == upper.shape[0]):
        raise ValueError(
            f"{path}: plan/lower/upper horizon lengths differ: "
            f"{plan.shape[0]}, {lower.shape[0]}, {upper.shape[0]}"
        )

    return data


def scalar_from_data(data: dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in data:
        return default
    return float(np.asarray(data[key]).reshape(()))


def int_from_data(data: dict[str, np.ndarray], key: str, default: int) -> int:
    if key not in data:
        return default
    return int(np.asarray(data[key]).reshape(()))


def plot_tube(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    *,
    color: str,
    label: str,
    default_stride: int = 2,
) -> None:
    plan = data["plan"]
    lower = data["lower"]
    upper = data["upper"]

    # Side view uses horizontal position x = state 0 and altitude z = state 2.
    x = plan[:, 0]
    z = plan[:, 2]
    x_lower = lower[:, 0]
    x_upper = upper[:, 0]
    z_lower = lower[:, 2]
    z_upper = upper[:, 2]

    # Continuous vertical envelope. Sorting avoids malformed fill_between
    # polygons if the optimized trajectory is not strictly monotone in x.
    order = np.argsort(x)
    ax.fill_between(
        x[order],
        z_lower[order],
        z_upper[order],
        color=color,
        alpha=0.14,
        linewidth=0.0,
        label=f"{label} tube envelope",
        zorder=2,
    )

    # Draw local x-z tube cross-sections at the stored stride.
    stride = max(1, int_from_data(data, "tube_stride", default_stride))
    indices = list(range(0, len(x), stride))
    if indices[-1] != len(x) - 1:
        indices.append(len(x) - 1)

    for k in indices:
        width = max(float(x_upper[k] - x_lower[k]), 0.0)
        height = max(float(z_upper[k] - z_lower[k]), 0.0)
        rect = Rectangle(
            (float(x_lower[k]), float(z_lower[k])),
            width,
            height,
            facecolor=color,
            edgecolor=color,
            alpha=0.10,
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.plot(
        x,
        z,
        color=color,
        linewidth=2.6,
        label=label,
        zorder=5,
    )
    ax.scatter(
        [x[0]],
        [z[0]],
        color=color,
        marker="o",
        s=42,
        zorder=6,
    )
    ax.scatter(
        [x[-1]],
        [z[-1]],
        color=color,
        marker="X",
        s=58,
        zorder=6,
    )


def draw_environment(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    centers = np.atleast_2d(data["centers"])
    radii = np.atleast_1d(data["radii"])

    if centers.shape[0] != radii.shape[0]:
        raise ValueError(
            "Obstacle center/radius counts do not match: "
            f"{centers.shape[0]} centers and {radii.shape[0]} radii"
        )

    # A sphere projects to a circle in the x-z side view.
    # for center, radius in zip(centers, radii):
    #     obstacle = Circle(
    #         (float(center[0]), float(center[2])),
    #         float(radius),
    #         facecolor="0.55",
    #         edgecolor="black",
    #         alpha=0.45,
    #         linewidth=1.2,
    #         zorder=1,
    #     )
    #     ax.add_patch(obstacle)

    goal_center = np.asarray(data["goal_center"]).reshape(-1)
    goal_half_width = np.asarray(data["goal_half_width"]).reshape(-1)
    if goal_center.size < 3 or goal_half_width.size < 3:
        raise ValueError("goal_center and goal_half_width must each contain x, y, z.")

    goal = Rectangle(
        (
            float(goal_center[0] - goal_half_width[0]),
            float(goal_center[2] - goal_half_width[2]),
        ),
        float(2.0 * goal_half_width[0]),
        float(2.0 * goal_half_width[2]),
        facecolor="gold",
        edgecolor="darkgoldenrod",
        alpha=0.28,
        hatch="//",
        linewidth=1.4,
        label="Goal region",
        zorder=1,
    )
    ax.add_patch(goal)

    ground_height = scalar_from_data(data, "ground_height", 0.0)
    ax.axhline(
        ground_height,
        color="black",
        linewidth=1.2,
        linestyle="--",
        label="Ground",
        zorder=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay gradient and no-gradient quadrotor side-view tubes."
    )
    parser.add_argument(
        "--no-grad",
        type=Path,
        default=Path("quadrotor_side_view_data_no_grad.npz"),
        help="Path to the no-gradient NPZ archive.",
    )
    parser.add_argument(
        "--grad",
        type=Path,
        default=Path("quadrotor_side_view_data_grad.npz"),
        help="Path to the gradient NPZ archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("quadrotor_side_view_grad_comparison.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving.",
    )
    args = parser.parse_args()

    no_grad = load_side_view_data(args.no_grad)
    grad = load_side_view_data(args.grad)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    limits = combined_plot_limits(no_grad, grad)
    draw_disturbance_gradient(fig, ax, limits)

    # Draw environment once. It should be identical in both archives.
    draw_environment(ax, no_grad)

    plot_tube(
        ax,
        no_grad,
        color="red",
        label="No gradient",
    )
    plot_tube(
        ax,
        grad,
        color="green",
        label="With gradient",
    )

    ax.set_xlabel("Horizontal position $x$ [m]")
    ax.set_ylabel("Altitude $z$ [m]")
    ax.set_title("Quadrotor Side View: Gradient vs. No Gradient")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")

    # Remove duplicate legend entries while retaining order.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best", frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
