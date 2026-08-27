"""Render the closed-loop H1 RTI rollout to an MP4.

Playback timing comes directly from the saved RTI rollout ``node_times``.
Slow motion changes playback timing only; the saved rollout states are not
modified.

Examples:
    python3 render_h1_rti_rollout.py
    python3 render_h1_rti_rollout.py humanoid_backflip_rti_h15.npz
    python3 render_h1_rti_rollout.py --slowmo 4
    python3 render_h1_rti_rollout.py --output humanoid_backflip_rti_h15.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BARREL_RENDERER_DIR = SCRIPT_DIR.parent / "quadruped_barrel_roll_obstacle"
sys.path.insert(0, str(BARREL_RENDERER_DIR))
from render_barrel_roll import render_video  # noqa: E402

DEFAULT_INPUT = SCRIPT_DIR / "humanoid_backflip_rti_h15.npz"

# Only used if the RTI rollout file does not contain obstacle geometry.
DEFAULT_OBSTACLE_CENTERS = np.array([
    [-0.40, 0.0, 0.11],
    [-0.40, 0.0, 2.55],
], dtype=np.float64)

DEFAULT_OBSTACLE_SIZES = np.array([
    [0.08, 0.80, 0.22],
    [0.08, 0.80, 0.40],
], dtype=np.float64)


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


def load_rollout(filename: Path):
    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Rollout file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        if "X" not in result:
            raise ValueError("The RTI rollout NPZ must contain X.")
        if "node_times" not in result:
            raise ValueError("The RTI rollout NPZ must contain node_times.")

        states = np.asarray(result["X"], dtype=np.float64)
        node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)

        if "obstacle_centers" in result and "obstacle_sizes" in result:
            obstacle_centers = np.asarray(
                result["obstacle_centers"], dtype=np.float64
            )
            obstacle_sizes = np.asarray(
                result["obstacle_sizes"], dtype=np.float64
            )
        elif "obstacle_center" in result and "obstacle_size" in result:
            obstacle_centers = np.asarray(
                result["obstacle_center"], dtype=np.float64
            ).reshape(1, 3)
            obstacle_sizes = np.asarray(
                result["obstacle_size"], dtype=np.float64
            ).reshape(1, 3)
        else:
            obstacle_centers = DEFAULT_OBSTACLE_CENTERS.copy()
            obstacle_sizes = DEFAULT_OBSTACLE_SIZES.copy()

        solve_times = (
            np.asarray(result["solve_times"], dtype=np.float64).reshape(-1)
            if "solve_times" in result
            else None
        )
        position_error = (
            np.asarray(result["position_error"], dtype=np.float64).reshape(-1)
            if "position_error" in result
            else None
        )
        orientation_error_deg = (
            np.asarray(result["orientation_error_deg"], dtype=np.float64).reshape(-1)
            if "orientation_error_deg" in result
            else None
        )
        horizon = (
            int(np.asarray(result["horizon"]).reshape(()))
            if "horizon" in result
            else None
        )

    if states.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {states.shape}.")

    if node_times.size != states.shape[0]:
        raise ValueError(
            f"X has {states.shape[0]} states but node_times has "
            f"{node_times.size} entries."
        )

    # Render the largest valid prefix instead of rejecting the whole rollout
    # when the closed-loop simulation eventually diverges to NaN/Inf.
    state_row_finite = np.all(np.isfinite(states), axis=1)
    time_finite = np.isfinite(node_times)
    valid_row = state_row_finite & time_finite

    first_bad_index = None
    bad_state_columns = np.array([], dtype=int)

    bad_rows = np.flatnonzero(~valid_row)
    if bad_rows.size:
        first_bad_index = int(bad_rows[0])
        if not state_row_finite[first_bad_index]:
            bad_state_columns = np.flatnonzero(
                ~np.isfinite(states[first_bad_index])
            )

    # A non-increasing timestamp also ends the renderable prefix, even if the
    # values themselves are finite.
    finite_prefix_stop = states.shape[0] if first_bad_index is None else first_bad_index
    if finite_prefix_stop >= 2:
        bad_dt = np.flatnonzero(np.diff(node_times[:finite_prefix_stop]) <= 0.0)
        if bad_dt.size:
            first_bad_index = int(bad_dt[0] + 1)
            finite_prefix_stop = first_bad_index

    if first_bad_index is not None:
        states = states[:finite_prefix_stop]
        node_times = node_times[:finite_prefix_stop]

        # Per-step diagnostic arrays correspond to transitions, so keep only
        # entries associated with the retained state prefix.
        n_valid_steps = max(states.shape[0] - 1, 0)
        if solve_times is not None:
            solve_times = solve_times[:n_valid_steps]
        if position_error is not None:
            position_error = position_error[:n_valid_steps]
        if orientation_error_deg is not None:
            orientation_error_deg = orientation_error_deg[:n_valid_steps]

    if states.shape[0] < 2:
        raise ValueError(
            "The rollout becomes invalid before two finite state nodes are available "
            "to render."
        )

    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(node_times)):
        raise RuntimeError("Internal error while truncating the finite rollout prefix.")

    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("The finite node_times prefix must be strictly increasing.")

    if (
        obstacle_centers.ndim != 2
        or obstacle_sizes.ndim != 2
        or obstacle_centers.shape[1] != 3
        or obstacle_sizes.shape != obstacle_centers.shape
    ):
        raise ValueError("Obstacle centers and sizes must have shape (K, 3).")

    if (
        not np.all(np.isfinite(obstacle_centers))
        or not np.all(np.isfinite(obstacle_sizes))
        or np.any(obstacle_sizes <= 0.0)
    ):
        raise ValueError("Obstacle geometry must be finite and positive.")

    # Remove any nonzero starting offset. render_video only cares about elapsed
    # playback time, and this makes the duration reporting unambiguous.
    node_times = node_times - node_times[0]

    return {
        "states": states,
        "node_times": node_times,
        "obstacle_centers": obstacle_centers,
        "obstacle_sizes": obstacle_sizes,
        "solve_times": solve_times,
        "position_error": position_error,
        "orientation_error_deg": orientation_error_deg,
        "horizon": horizon,
        "first_bad_index": first_bad_index,
        "bad_state_columns": bad_state_columns,
        "original_state_count": int(valid_row.size),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--slowmo",
        type=float,
        default=1.0,
        help="Playback slowdown factor; 1 preserves the physical rollout duration.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--lookat", nargs=3, type=float, default=None)
    args = parser.parse_args()

    if not np.isfinite(args.slowmo) or args.slowmo <= 0.0:
        parser.error("--slowmo must be a finite positive number.")

    if not np.isfinite(args.fps) or args.fps <= 0.0:
        parser.error("--fps must be a finite positive number.")

    trajectory_path = args.trajectory.expanduser().resolve()
    rollout = load_rollout(trajectory_path)

    states = rollout["states"]
    node_times = rollout["node_times"]
    obstacle_centers = rollout["obstacle_centers"]
    obstacle_sizes = rollout["obstacle_sizes"]

    playback_node_times = node_times * args.slowmo
    model_path = resolve_model_path(args.model)

    output = args.output
    if output is None:
        output = trajectory_path.with_suffix(".mp4")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = render_video(
        states,
        playback_node_times,
        model_path,
        output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
        obstacle_center=obstacle_centers,
        obstacle_size=obstacle_sizes,
    )

    print(f"Model: {model_path}")
    print(f"Rendered rollout states: {states.shape[0]}")
    if rollout["first_bad_index"] is not None:
        bad_index = rollout["first_bad_index"]
        print(
            f"WARNING: rollout first becomes invalid at state index {bad_index}; "
            f"rendering only states 0..{states.shape[0] - 1}."
        )
        if rollout["bad_state_columns"].size:
            print(
                "Nonfinite state columns at first bad node: "
                f"{rollout['bad_state_columns'].tolist()}"
            )
    else:
        print(f"Full rollout states: {rollout['original_state_count']}")
    if rollout["horizon"] is not None:
        print(f"RTI horizon: {rollout['horizon']}")
    print(f"Physical duration: {node_times[-1]:.6f} s")
    print(f"Slow motion: {args.slowmo:g}x")
    print(f"Playback duration: {playback_node_times[-1]:.6f} s")

    if rollout["solve_times"] is not None and rollout["solve_times"].size:
        solve_times = rollout["solve_times"]
        print(f"Mean RTI solve time: {1e3 * np.mean(solve_times):.3f} ms")
        print(f"Max RTI solve time:  {1e3 * np.max(solve_times):.3f} ms")

    if rollout["position_error"] is not None and rollout["position_error"].size:
        print(
            "Max position tracking error: "
            f"{np.max(rollout['position_error']):.6f} m"
        )

    if (
        rollout["orientation_error_deg"] is not None
        and rollout["orientation_error_deg"].size
    ):
        print(
            "Max orientation tracking error: "
            f"{np.max(rollout['orientation_error_deg']):.6f} deg"
        )

    print(f"Corridor obstacle centers:\n{obstacle_centers}")
    print(f"Corridor obstacle sizes:\n{obstacle_sizes} m")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output}")


if __name__ == "__main__":
    main()
