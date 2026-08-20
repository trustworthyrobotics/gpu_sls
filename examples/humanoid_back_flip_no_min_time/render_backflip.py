"""Render a fixed-time H1 backflip trajectory to a slow-motion MP4.

Slow motion changes playback timing only; the saved states and fixed phase
durations are not modified.

Examples:
    python render_backflip.py
    python render_backflip.py humanoid_backflip_no_min_time.npz --slowmo 6
    python render_backflip.py --slowmo 4 --output backflip_slowmo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BARREL_RENDERER_DIR = SCRIPT_DIR.parent / "quadruped_barrel_roll"
sys.path.insert(0, str(BARREL_RENDERER_DIR))
from render_barrel_roll import reconstruct_node_times, render_video  # noqa: E402

DEFAULT_INPUT = SCRIPT_DIR / "humanoid_backflip_no_min_time.npz"


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        return path
    try:
        import mpx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import mpx to locate the H1 model; install the project "
            "dependencies or pass --model /path/to/scene.xml."
        ) from error
    model_dir = Path(mpx.__file__).resolve().parent / "data" / "unitree_h1"
    candidates = (
        model_dir / "mjx_scene_h1_walk.xml",
        model_dir / "mjx_h1_walk_real_feet.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find an H1 scene under {model_dir}.")


def load_trajectory(filename: Path):
    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {filename}")
    with np.load(filename, allow_pickle=False) as result:
        if "X" not in result or "phase_times" not in result:
            raise ValueError("The NPZ must contain X and phase_times.")
        states = np.asarray(result["X"], dtype=np.float64)
        phase_times = np.asarray(result["phase_times"], dtype=np.float64).reshape(-1)
        if "node_times" in result:
            node_times = np.asarray(result["node_times"], dtype=np.float64)
        elif "phase_end_steps" in result:
            node_times = reconstruct_node_times(
                phase_times, result["phase_end_steps"], states.shape[0]
            )
        else:
            raise ValueError("The NPZ needs node_times or phase_end_steps.")
    node_times = node_times.reshape(-1)
    if states.ndim != 2 or node_times.size != states.shape[0]:
        raise ValueError("X and node_times have incompatible shapes.")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(node_times)):
        raise ValueError("Trajectory states and node times must be finite.")
    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("node_times must be strictly increasing.")
    return states, node_times, phase_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--slowmo", type=float, default=4.0,
        help="Playback slowdown factor; 4 renders the motion four times slower.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--lookat", nargs=3, type=float, default=None)
    args = parser.parse_args()
    if not np.isfinite(args.slowmo) or args.slowmo <= 0.0:
        parser.error("--slowmo must be a finite positive number.")

    trajectory_path = args.trajectory.expanduser().resolve()
    states, node_times, phase_times = load_trajectory(trajectory_path)
    playback_node_times = node_times * args.slowmo
    model_path = resolve_model_path(args.model)
    output = args.output
    if output is None:
        output = trajectory_path.with_name(
            f"{trajectory_path.stem}_slowmo.mp4"
        )
    frame_count = render_video(
        states, playback_node_times, model_path, output,
        fps=args.fps, width=args.width, height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
    )
    print(f"Model: {model_path}")
    print(f"Fixed phase times: {phase_times}")
    print(f"Physical duration: {node_times[-1]:.6f} s")
    print(f"Slow motion: {args.slowmo:g}x")
    print(f"Playback duration: {playback_node_times[-1]:.6f} s")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
