"""Render the closed-loop Go2 RTI barrel-roll rollout to MP4.

This expects the NPZ produced by track_barrel_roll_rti.py, i.e. an archive with
at least:

    X_rollout
    node_times

Only qpos/qvel are sent to MuJoCo. Extra rollout-state entries (feet and GRFs)
are ignored by the renderer.

Examples:
    python render_rti_rollout.py
    python render_rti_rollout.py quadruped_barrel_roll_rti_rollout.npz
    python render_rti_rollout.py --output rollout.mp4
    python render_rti_rollout.py --slowmo 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# Offscreen rendering.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "quadruped_barrel_roll_rti_rollout.npz"

# Same obstacle used by the planner/tracker.
DEFAULT_OBSTACLE_CENTER = np.array([0.0, -0.30, 0.08], dtype=np.float64)
DEFAULT_OBSTACLE_SIZE = np.array([0.80, 0.05, 0.16], dtype=np.float64)


def resolve_model_path(model_path: Path | None) -> Path:
    """Resolve an explicit Go2 XML or find the model distributed with MPX."""
    if model_path is not None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        return path

    try:
        import mpx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import mpx to locate the Go2 model. "
            "Install the project dependencies or pass --model /path/to/scene.xml."
        ) from error

    go2_dir = Path(mpx.__file__).resolve().parent / "data" / "go2"
    candidates = (
        go2_dir / "scene_mjx.xml",
        go2_dir / "go2_mjx.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find scene_mjx.xml or go2_mjx.xml under {go2_dir}."
    )


def truncate_to_finite_prefix(
    states: np.ndarray,
    node_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the longest prefix whose state and node time are finite."""
    state_finite = np.all(np.isfinite(states), axis=1)
    time_finite = np.isfinite(node_times)
    valid = state_finite & time_finite

    bad = np.flatnonzero(~valid)
    stop = int(bad[0]) if bad.size else states.shape[0]

    if stop < 2:
        raise ValueError(
            "The rollout becomes non-finite before two renderable states exist."
        )

    if stop < states.shape[0]:
        print(
            f"WARNING: rollout becomes non-finite at node {stop}; "
            f"rendering finite prefix nodes 0..{stop - 1}."
        )

    states = states[:stop]
    node_times = node_times[:stop]

    # The rollout archive should have increasing times. If it does not, keep the
    # longest strictly increasing prefix rather than letting interpolation fail.
    dt = np.diff(node_times)
    bad_dt = np.flatnonzero(dt <= 0.0)
    if bad_dt.size:
        stop2 = int(bad_dt[0]) + 1
        if stop2 < 2:
            raise ValueError("node_times has no usable strictly increasing prefix.")
        print(
            f"WARNING: node_times stops increasing after node {stop2 - 1}; "
            "truncating there."
        )
        states = states[:stop2]
        node_times = node_times[:stop2]

    # Normalize video time to start at zero.
    node_times = node_times - node_times[0]
    return states, node_times


def load_rollout(
    filename: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load X_rollout and timing from an RTI rollout NPZ."""
    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Rollout file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        if "X_rollout" not in result:
            raise ValueError(f"{filename} does not contain X_rollout.")
        if "node_times" not in result:
            raise ValueError(f"{filename} does not contain node_times.")

        states = np.asarray(result["X_rollout"], dtype=np.float64)
        node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)

        obstacle_center = np.asarray(
            result["obstacle_center"]
            if "obstacle_center" in result
            else DEFAULT_OBSTACLE_CENTER,
            dtype=np.float64,
        ).reshape(3)

        obstacle_size = np.asarray(
            result["obstacle_size"]
            if "obstacle_size" in result
            else DEFAULT_OBSTACLE_SIZE,
            dtype=np.float64,
        ).reshape(3)

    if states.ndim != 2 or states.shape[0] < 2:
        raise ValueError("X_rollout must have shape (K, nx) with K >= 2.")

    # Be tolerant if a partially saved archive has one extra time or one extra
    # state. Match the common prefix before finite-prefix truncation.
    common = min(states.shape[0], node_times.size)
    if common < 2:
        raise ValueError(
            f"Incompatible rollout sizes: X_rollout={states.shape}, "
            f"node_times={node_times.shape}."
        )
    if states.shape[0] != node_times.size:
        print(
            "WARNING: X_rollout and node_times lengths differ; "
            f"using first {common} entries."
        )
        states = states[:common]
        node_times = node_times[:common]

    states, node_times = truncate_to_finite_prefix(states, node_times)

    if not np.all(np.isfinite(obstacle_center)):
        raise ValueError("Obstacle center must be finite.")
    if not np.all(np.isfinite(obstacle_size)) or np.any(obstacle_size <= 0.0):
        raise ValueError("Obstacle size must be finite and positive.")

    return states, node_times, obstacle_center, obstacle_size


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path SLERP for MuJoCo wxyz quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1.0e-12 or n1 < 1.0e-12:
        # Fallback for a malformed quaternion in an otherwise finite trajectory.
        q = (1.0 - alpha) * q0 + alpha * q1
        n = np.linalg.norm(q)
        return q / max(n, 1.0e-12)

    q0 = q0 / n0
    q1 = q1 / n1

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - alpha) * theta) / sin_theta * q0
        + np.sin(alpha * theta) / sin_theta * q1
    )


def interpolate_qpos_qvel(
    states: np.ndarray,
    node_times: np.ndarray,
    frame_time: float,
    nq: int,
    nv: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate qpos/qvel from the rollout at one physical time."""
    if frame_time <= node_times[0]:
        left = right = 0
        alpha = 0.0
    elif frame_time >= node_times[-1]:
        left = right = len(node_times) - 1
        alpha = 0.0
    else:
        right = int(np.searchsorted(node_times, frame_time, side="right"))
        left = right - 1
        alpha = float(
            (frame_time - node_times[left])
            / (node_times[right] - node_times[left])
        )

    qpos = (
        (1.0 - alpha) * states[left, :nq]
        + alpha * states[right, :nq]
    )
    qpos[3:7] = quaternion_slerp(
        states[left, 3:7],
        states[right, 3:7],
        alpha,
    )

    qvel_start = nq
    qvel_stop = nq + nv
    qvel = (
        (1.0 - alpha) * states[left, qvel_start:qvel_stop]
        + alpha * states[right, qvel_start:qvel_stop]
    )

    return qpos, qvel


def add_box(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    size: np.ndarray,
) -> None:
    """Append the barrel-roll hurdle to the current render scene."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene has no room for the obstacle geom.")

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        0.5 * np.asarray(size, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.array([0.90, 0.20, 0.08, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def render_video(
    states: np.ndarray,
    node_times: np.ndarray,
    model_path: Path,
    output_path: Path,
    *,
    fps: float,
    slowmo: float,
    width: int,
    height: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    lookat: np.ndarray | None,
    obstacle_center: np.ndarray,
    obstacle_size: np.ndarray,
) -> int:
    """Render the rollout to H.264 MP4."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")
    if fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if slowmo <= 0.0:
        raise ValueError("--slowmo must be positive.")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if states.shape[1] < model.nq + model.nv:
        raise ValueError(
            f"X_rollout has {states.shape[1]} columns, but this MuJoCo model "
            f"requires at least nq + nv = {model.nq + model.nv}."
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
        lookat = 0.5 * (
            np.min(base_positions, axis=0)
            + np.max(base_positions, axis=0)
        )

    camera.lookat[:] = np.asarray(lookat, dtype=np.float64)
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Slow motion changes playback duration only. The rollout itself is not
    # modified. At video time t_video, sample physical rollout time
    # t_video / slowmo.
    physical_duration = float(node_times[-1])
    video_duration = physical_duration * slowmo
    frame_count = max(2, int(np.ceil(video_duration * fps)) + 1)
    video_frame_times = np.linspace(0.0, video_duration, frame_count)

    try:
        for video_time in video_frame_times:
            physical_time = min(video_time / slowmo, physical_duration)

            qpos, qvel = interpolate_qpos_qvel(
                states,
                node_times,
                physical_time,
                model.nq,
                model.nv,
            )

            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_normalizeQuat(model, data.qpos)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            add_box(renderer.scene, obstacle_center, obstacle_size)

            frame = renderer.render()
            encoder.stdin.write(frame.tobytes())

        encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace")
        return_code = encoder.wait()

    except Exception:
        try:
            if encoder.stdin is not None:
                encoder.stdin.close()
        except Exception:
            pass
        encoder.kill()
        encoder.wait()
        raise
    finally:
        renderer.close()

    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code}: {stderr}"
        )

    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rollout",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="RTI rollout NPZ.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4; defaults to <rollout_stem>.mp4.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Go2 MuJoCo XML. Defaults to the MPX Go2 model.",
    )
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--slowmo",
        type=float,
        default=1.0,
        help="Playback slowdown factor. 1 = real rollout timing, 4 = 4x slower.",
    )
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
        help="Camera target. Defaults to the center of the rollout base path.",
    )
    args = parser.parse_args()

    rollout_path = args.rollout.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else rollout_path.with_suffix(".mp4")
    )

    states, node_times, obstacle_center, obstacle_size = load_rollout(
        rollout_path
    )
    model_path = resolve_model_path(args.model)

    frame_count = render_video(
        states,
        node_times,
        model_path,
        output_path,
        fps=args.fps,
        slowmo=args.slowmo,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
        obstacle_center=obstacle_center,
        obstacle_size=obstacle_size,
    )

    print(f"Loaded rollout: {rollout_path}")
    print(f"Model: {model_path}")
    print(f"Renderable nodes: {states.shape[0]}")
    print(f"Physical duration: {node_times[-1]:.6f} s")
    print(f"Slow motion: {args.slowmo:g}x")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
