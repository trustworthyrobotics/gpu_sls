"""Render the nominal barrel-roll reference by teleporting the Go2 pose.

This script does not simulate the robot or require the reference to satisfy
the dynamics.  It directly assigns an interpolated reference ``qpos`` and
``qvel`` to MuJoCo for every video frame and runs forward kinematics only.

Example:
    python render_reference.py
    python render_reference.py --output barrel_roll_reference.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Select the offscreen backend before importing MuJoCo through either module.
os.environ.setdefault("MUJOCO_GL", "egl")

import jax.numpy as jnp
import numpy as np

import quadruped_barrel_roll as problem
from render_barrel_roll import render_video, resolve_model_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "quadruped_barrel_roll_reference.mp4"


def build_reference_states() -> tuple[np.ndarray, np.ndarray]:
    """Return reference qpos/qvel states and their nominal physical times."""

    problem.configure_problem()
    reference, _ = problem.build_barrel_roll_reference()
    x0 = jnp.asarray(problem.config.initial_state)
    state_guess = problem.build_initial_guess(x0, reference)

    qpos_qvel_size = 13 + 2 * problem.config.n_joints
    states = np.asarray(state_guess[:, :qpos_qvel_size], dtype=np.float64)
    # The reference generator samples at config.dt; using those node indices
    # avoids carrying float32 roundoff from the optimization constants into
    # the video frame-count calculation.
    node_times = (
        np.arange(state_guess.shape[0], dtype=np.float64)
        * float(problem.config.dt)
    )

    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(node_times)):
        raise ValueError("The generated reference contains non-finite values.")
    return states, node_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output MP4 path (default: {DEFAULT_OUTPUT.name}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Go2 scene XML; defaults to the model supplied by MPX.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=1.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--lookat",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Camera target; defaults to the center of the base trajectory.",
    )
    args = parser.parse_args()

    states, node_times = build_reference_states()
    model_path = resolve_model_path(args.model)
    output_path = args.output.expanduser().resolve()
    frame_count = render_video(
        states,
        node_times,
        model_path,
        output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
    )

    print("Rendered the reference by direct state assignment (no simulation).")
    print(f"Model: {model_path}")
    print(f"Duration: {node_times[-1]:.6f} s")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
