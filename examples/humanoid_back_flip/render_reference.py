"""Render the nominal H1 backflip reference without simulating it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax.numpy as jnp
import numpy as np

import humanoid as problem
from render_backflip import render_video, resolve_model_path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "humanoid_backflip_reference.mp4"


def build_reference_states():
    problem.configure_problem()
    reference, _ = problem.build_backflip_reference()
    state_guess = problem.build_initial_guess(
        jnp.asarray(problem.config.initial_state), reference
    )
    qpos_qvel_size = 13 + 2 * problem.config.n_joints
    states = np.asarray(state_guess[:, :qpos_qvel_size], dtype=np.float64)
    states[:, problem.QVEL_START:problem.QVEL_STOP] *= np.asarray(
        problem.VELOCITY_SCALE
    )
    node_times = problem.phase_node_times(np.asarray(problem.NOMINAL_DURATIONS))
    return states, node_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--lookat", nargs=3, type=float, default=None)
    args = parser.parse_args()
    states, node_times = build_reference_states()
    model_path = resolve_model_path(args.model)
    frame_count = render_video(
        states, node_times, model_path, args.output,
        fps=args.fps, width=args.width, height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
    )
    print("Rendered the nominal reference by direct state assignment.")
    print(f"Duration: {node_times[-1]:.6f} s")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
