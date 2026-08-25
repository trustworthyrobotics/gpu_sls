#!/usr/bin/env python3
"""Run the two FastSLS rocket gradient-window experiments and compare them.

This launcher holds the dynamics, objective, constraints, initialization, and
disturbance profile fixed. Only SLSConfig.gradient_window changes:

  * no_gradient:   gradient_window = 0
  * gradient_aware: gradient_window = 10
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from plot_gradient_window_comparison import make_comparison


EXPERIMENTS = (
    ("no_gradient", 0),
    ("gradient_aware", 10),
)
MAX_DYNAMICS_DEFECT = 5.0e-2
MAX_CONSTRAINT_VIOLATION = 5.0e-2
MAX_TIGHTENED_CONSTRAINT_VIOLATION = 1.5e-1


def run_experiment(script: Path, output_dir: Path, label: str, window: int) -> Path:
    run_dir = output_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "rocket_result.npz"
    log_path = run_dir / "run.log"
    command = [
        sys.executable,
        str(script),
        "--gradient-window",
        str(window),
        "--output-dir",
        str(run_dir),
    ]
    env = os.environ.copy()
    local_source = script.parents[2] / "src"
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(local_source) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )

    print(f"Running {label}: FastSLS gradient_window={window}")
    print(f"  log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=script.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}; see {log_path}"
        )
    if not result_path.exists():
        raise RuntimeError(f"{label} did not produce {result_path}")
    return result_path


def result_metrics(path: Path) -> dict[str, float]:
    with np.load(path) as result:
        z = np.asarray(result["plan"])[:, 2]
        magnitude = np.asarray(result["disturbance_magnitude"])
        scale = float(np.asarray(result["disturbance_scale"]).reshape(()))
        slope = float(np.asarray(result["disturbance_slope"]).reshape(()))
        bias = float(np.asarray(result["disturbance_bias"]).reshape(()))
        transition = -bias / slope
        return {
            "gradient_window": int(np.asarray(result["gradient_window"]).reshape(())),
            "min_time": float(np.asarray(result["min_time"]).reshape(())),
            "mean_disturbance": float(np.mean(magnitude)),
            "max_disturbance": float(np.max(magnitude)),
            "fraction_below_transition": float(np.mean(z < transition)),
            "disturbance_transition": transition,
            "disturbance_scale": scale,
            "max_dynamics_defect": float(
                np.asarray(result["max_dynamics_defect"]).reshape(())
            ),
            "max_constraint_violation": float(
                np.asarray(result["max_constraint_violation"]).reshape(())
            ),
            "max_tightened_constraint_violation": float(
                np.asarray(result["max_tightened_constraint_violation"]).reshape(())
            ),
        }

def validate_result_metrics(label: str, values: dict[str, float]) -> None:
    tolerances = {
        "max_dynamics_defect": MAX_DYNAMICS_DEFECT,
        "max_constraint_violation": MAX_CONSTRAINT_VIOLATION,
        "max_tightened_constraint_violation": (
            MAX_TIGHTENED_CONSTRAINT_VIOLATION
        ),
    }
    failures = [
        f"{key}={values[key]:.6g} > {limit:.6g}"
        for key, limit in tolerances.items()
        if not np.isfinite(values[key]) or values[key] > limit
    ]
    if failures:
        raise RuntimeError(
            f"{label} is not accurate enough for the ablation: "
            + "; ".join(failures)
        )


def write_summary(
    output_path: Path,
    paths: dict[str, Path],
) -> dict[str, dict[str, float]]:
    metrics = {label: result_metrics(path) for label, path in paths.items()}
    for label, values in metrics.items():
        validate_result_metrics(label, values)
    fieldnames = [
        "experiment",
        "enable_fastsls",
        "gradient_window",
        "min_time_s",
        "mean_disturbance",
        "max_disturbance",
        "fraction_below_transition",
        "max_dynamics_defect",
        "max_constraint_violation",
        "max_tightened_constraint_violation",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for label, _ in EXPERIMENTS:
            values = metrics[label]
            writer.writerow(
                {
                    "experiment": label,
                    "enable_fastsls": True,
                    "gradient_window": int(values["gradient_window"]),
                    "min_time_s": f'{values["min_time"]:.9g}',
                    "mean_disturbance": f'{values["mean_disturbance"]:.9g}',
                    "max_disturbance": f'{values["max_disturbance"]:.9g}',
                    "fraction_below_transition": (
                        f'{values["fraction_below_transition"]:.9g}'
                    ),
                    "max_dynamics_defect": (
                        f'{values["max_dynamics_defect"]:.9g}'
                    ),
                    "max_constraint_violation": (
                        f'{values["max_constraint_violation"]:.9g}'
                    ),
                    "max_tightened_constraint_violation": (
                        f'{values["max_tightened_constraint_violation"]:.9g}'
                    ),
                }
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rocket_gradient_window_experiments"),
        help="Directory containing both runs, the summary CSV, and comparison plot.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Reuse existing NPZ results and only rebuild the summary and plot.",
    )
    parser.add_argument(
        "--allow-no-improvement",
        action="store_true",
        help="Do not fail if window 10 does not achieve a lower minimum time.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    rocket_script = script_dir / "rocket.py"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for label, window in EXPERIMENTS:
        expected_path = output_dir / label / "rocket_result.npz"
        if args.skip_runs:
            if not expected_path.exists():
                raise FileNotFoundError(
                    f"--skip-runs requested, but result is missing: {expected_path}"
                )
            paths[label] = expected_path
        else:
            paths[label] = run_experiment(
                rocket_script,
                output_dir,
                label,
                window,
            )

    summary_path = output_dir / "summary.csv"
    metrics = write_summary(summary_path, paths)
    comparison_path = output_dir / "gradient_window_comparison.png"
    no_gradient_time, gradient_aware_time = make_comparison(
        paths["no_gradient"],
        paths["gradient_aware"],
        comparison_path,
    )

    print(f"Saved summary: {summary_path}")
    print(
        "Low-disturbance fractions: "
        f'window 0 = {metrics["no_gradient"]["fraction_below_transition"]:.1%}, '
        f'window 10 = {metrics["gradient_aware"]["fraction_below_transition"]:.1%}'
    )

    no_gradient_low_fraction = metrics["no_gradient"]["fraction_below_transition"]
    gradient_aware_low_fraction = metrics["gradient_aware"][
        "fraction_below_transition"
    ]
    if (
        gradient_aware_low_fraction <= no_gradient_low_fraction
        and not args.allow_no_improvement
    ):
        raise RuntimeError(
            "The intended path-choice ablation was not reproduced: "
            "gradient_window=10 did not spend more of its trajectory in the "
            "low-disturbance region."
        )

    if gradient_aware_time >= no_gradient_time and not args.allow_no_improvement:
        raise RuntimeError(
            "The intended ablation was not reproduced: gradient_window=10 did "
            f"not reduce minimum time ({gradient_aware_time:.6g} s versus "
            f"{no_gradient_time:.6g} s). Inspect the run logs before presenting "
            "this experiment, or pass --allow-no-improvement for diagnostics."
        )


if __name__ == "__main__":
    main()
