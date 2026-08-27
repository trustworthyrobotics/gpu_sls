"""Render the saved zero-disturbance rollout for a minimum-time quadruped barrel roll.

The renderer visualizes X_zero_disturbance_rollout saved by
quadruped_barrel_roll.py. It uses the saved physical node times, interpolates
between shooting nodes, and applies quaternion SLERP to keep the roll smooth.

Example:
    python render_zero_disturbance_rollout.py quadruped_barrel_roll_obstacle_min_time.npz
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# MuJoCo chooses its GL backend when imported.  This script only renders to an
# offscreen framebuffer, so do not let an SSH-forwarded DISPLAY select GLFW.
# An explicit environment setting still wins, e.g. MUJOCO_GL=osmesa.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "quadruped_barrel_roll_obstacle_min_time.npz"
DEFAULT_OBSTACLE_CENTER = np.array([0.0, -0.30, 0.08])
DEFAULT_OBSTACLE_SIZE = np.array([0.80, 0.05, 0.16])


def resolve_model_path(model_path: Path | None) -> Path:
    """Resolve an explicit model or the scene distributed with MPX."""

    if model_path is not None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        return path

    try:
        import mpx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import mpx to locate the Go2 model. Install the "
            "gpu_sls project dependencies or pass --model /path/to/scene.xml."
        ) from error

    go2_dir = Path(mpx.__file__).resolve().parent / "data" / "go2"
    candidates = [go2_dir / "scene_mjx.xml", go2_dir / "go2_mjx.xml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find scene_mjx.xml or go2_mjx.xml under "
        f"{go2_dir}. Pass the model explicitly with --model."
    )


def reconstruct_node_times(
    phase_times: np.ndarray,
    phase_end_steps: np.ndarray,
    state_count: int,
) -> np.ndarray:
    """Reconstruct N+1 physical node times from optimized phase durations."""

    phase_times = np.asarray(phase_times, dtype=np.float64).reshape(-1)
    phase_end_steps = np.asarray(phase_end_steps, dtype=np.int64).reshape(-1)
    if phase_times.size != phase_end_steps.size:
        raise ValueError("phase_times and phase_end_steps must have equal length.")

    segment_lengths = np.diff(np.concatenate([[0], phase_end_steps]))
    if np.any(segment_lengths <= 0):
        raise ValueError("phase_end_steps must be strictly increasing.")
    if int(phase_end_steps[-1]) != state_count - 1:
        raise ValueError(
            "The final phase_end_step must equal X.shape[0] - 1; got "
            f"{phase_end_steps[-1]} and {state_count - 1}."
        )

    transition_dt = np.repeat(phase_times / segment_lengths, segment_lengths)
    return np.concatenate([[0.0], np.cumsum(transition_dt)])


def load_trajectory(
    filename: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load zero-disturbance rollout timing and obstacle geometry from an experiment NPZ."""

    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        if "X_zero_disturbance_rollout" not in result:
            raise ValueError(
                f"{filename} does not contain X_zero_disturbance_rollout."
            )
        states = np.asarray(result["X_zero_disturbance_rollout"], dtype=np.float64)
        if states.ndim != 2 or states.shape[0] < 2:
            raise ValueError(
                "X_zero_disturbance_rollout must have shape (N + 1, nx) with N >= 1."
            )
        if "phase_times" not in result:
            raise ValueError(f"{filename} does not contain phase_times.")
        phase_times = np.asarray(result["phase_times"], dtype=np.float64)
        if phase_times.size != 6:
            raise ValueError(
                "This is a legacy five-phase trajectory with the known "
                "touchdown artifact. Re-run quadruped_barrel_roll.py to "
                "generate the corrected six-phase result before rendering."
            )

        if "node_times" in result:
            node_times = np.asarray(result["node_times"], dtype=np.float64)
        elif "phase_end_steps" in result:
            node_times = reconstruct_node_times(
                phase_times, result["phase_end_steps"], states.shape[0]
            )
        else:
            raise ValueError(
                "The NPZ needs node_times or phase_end_steps to recover timing."
            )
        obstacle_center = np.asarray(
            result.get("obstacle_center", DEFAULT_OBSTACLE_CENTER),
            dtype=np.float64,
        ).reshape(-1)
        obstacle_size = np.asarray(
            result.get("obstacle_size", DEFAULT_OBSTACLE_SIZE),
            dtype=np.float64,
        ).reshape(-1)

    node_times = node_times.reshape(-1)
    if node_times.size != states.shape[0]:
        raise ValueError(
            f"node_times has {node_times.size} entries, but X has "
            f"{states.shape[0]} states."
        )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(node_times)):
        raise ValueError("Trajectory states and node times must be finite.")
    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("node_times must be strictly increasing.")
    if obstacle_center.size != 3 or obstacle_size.size != 3:
        raise ValueError("obstacle_center and obstacle_size must have 3 entries.")
    if not np.all(np.isfinite(obstacle_center)) or not np.all(
        np.isfinite(obstacle_size)
    ):
        raise ValueError("Obstacle geometry must be finite.")
    if np.any(obstacle_size <= 0.0):
        raise ValueError("Obstacle dimensions must be positive.")

    return (
        states,
        node_times,
        phase_times.reshape(-1),
        obstacle_center,
        obstacle_size,
    )


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path SLERP for MuJoCo wxyz quaternions."""

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - alpha) * theta) / sin_theta * q0
        + np.sin(alpha * theta) / sin_theta * q1
    )


def interpolate_state(
    states: np.ndarray,
    node_times: np.ndarray,
    frame_time: float,
    nq: int,
    nv: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate qpos/qvel at one physical video time."""

    if frame_time <= node_times[0]:
        left = right = 0
        alpha = 0.0
    elif frame_time >= node_times[-1]:
        left = right = len(node_times) - 1
        alpha = 0.0
    else:
        right = int(np.searchsorted(node_times, frame_time, side="right"))
        left = right - 1
        alpha = (frame_time - node_times[left]) / (
            node_times[right] - node_times[left]
        )

    qpos = (1.0 - alpha) * states[left, :nq] + alpha * states[right, :nq]
    qpos[3:7] = quaternion_slerp(
        states[left, 3:7], states[right, 3:7], alpha
    )

    qvel_start = nq
    qvel_stop = nq + nv
    qvel = (
        (1.0 - alpha) * states[left, qvel_start:qvel_stop]
        + alpha * states[right, qvel_start:qvel_stop]
    )
    return qpos, qvel


def add_obstacle_to_scene(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray | None = None,
) -> None:
    """Append the optimizer's hurdle box to a rendered MuJoCo scene."""

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo render scene has no room for the obstacle.")
    geom = scene.geoms[scene.ngeom]
    if rgba is None:
        rgba = np.array([0.90, 0.20, 0.08, 1.0], dtype=np.float32)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        0.5 * size,
        center,
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render_video(
    states: np.ndarray,
    node_times: np.ndarray,
    model_path: Path,
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    lookat: np.ndarray | None,
    obstacle_center: np.ndarray,
    obstacle_size: np.ndarray,
) -> int:
    """Render the trajectory with one obstacle or an array of obstacles."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers.")
    obstacle_centers = np.asarray(obstacle_center, dtype=np.float64)
    obstacle_sizes = np.asarray(obstacle_size, dtype=np.float64)
    if obstacle_centers.ndim == 1:
        obstacle_centers = obstacle_centers.reshape(1, -1)
    if obstacle_sizes.ndim == 1:
        obstacle_sizes = obstacle_sizes.reshape(1, -1)
    if (
        obstacle_centers.ndim != 2
        or obstacle_sizes.ndim != 2
        or obstacle_centers.shape[1] != 3
        or obstacle_sizes.shape[1] != 3
        or obstacle_centers.shape[0] != obstacle_sizes.shape[0]
    ):
        raise ValueError("Obstacle centers and sizes must have shape (K, 3).")
    if not np.all(np.isfinite(obstacle_centers)) or not np.all(
        np.isfinite(obstacle_sizes)
    ) or np.any(obstacle_sizes <= 0.0):
        raise ValueError("Obstacle geometry must be finite and positive.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if states.shape[1] < model.nq + model.nv:
        raise ValueError(
            f"X_zero_disturbance_rollout has {states.shape[1]} columns, "
            f"but model nq + nv is "
            f"{model.nq + model.nv}."
        )

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if lookat is None:
        base_positions = states[:, :3]
        lookat = 0.5 * (base_positions.min(axis=0) + base_positions.max(axis=0))
    camera.lookat[:] = np.asarray(lookat, dtype=np.float64)
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    encoder = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    duration = float(node_times[-1])
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    frame_times = np.linspace(0.0, duration, frame_count)
    try:
        for frame_time in frame_times:
            qpos, qvel = interpolate_state(
                states, node_times, frame_time, model.nq, model.nv
            )
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_normalizeQuat(model, data.qpos)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            for obstacle_index, (center, size) in enumerate(zip(
                obstacle_centers, obstacle_sizes
            )):
                color = np.array(
                    [0.90, 0.20 + 0.08 * (obstacle_index % 2), 0.08, 1.0],
                    dtype=np.float32,
                )
                add_obstacle_to_scene(renderer.scene, center, size, color)
            encoder.stdin.write(renderer.render().tobytes())

        encoder.stdin.close()
        error_output = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()
    except Exception:
        encoder.kill()
        encoder.wait()
        raise
    finally:
        renderer.close()

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}: {error_output}"
        )
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path; defaults to the trajectory name with .mp4.",
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

    trajectory_path = args.trajectory.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trajectory_path.with_name(
            trajectory_path.stem + "_zero_disturbance_rollout.mp4"
        )
    )
    states, node_times, phase_times, obstacle_center, obstacle_size = (
        load_trajectory(trajectory_path)
    )
    model_path = resolve_model_path(args.model)
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
        obstacle_center=obstacle_center,
        obstacle_size=obstacle_size,
    )

    print(f"Loaded rollout from: {trajectory_path}")
    print("State key: X_zero_disturbance_rollout")
    print(f"Model: {model_path}")
    print(f"Phase times: {phase_times}")
    print(f"Duration: {node_times[-1]:.6f} s")
    print(f"Obstacle center/size: {obstacle_center} / {obstacle_size} m")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()