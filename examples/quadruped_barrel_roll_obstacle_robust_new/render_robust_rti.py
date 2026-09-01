"""Render a saved robust RTI-MPC rollout for the Unitree Go2 barrel roll.

The renderer visualizes ``X_rollout`` saved by ``track_barrel_roll_robust_rti.py``.
It uses the saved physical ``node_times``, linearly interpolates positions and
velocities between RTI nodes, and applies quaternion SLERP for smooth attitude
motion.

The robust RTI output stores the path of the offline minimum-time trajectory in
``source_trajectory``.  When available, this renderer reads the obstacle geometry
from that source NPZ so the hurdle exactly matches the planning experiment.
Otherwise it falls back to the default hurdle geometry.

Examples:
    python render_barrel_roll_robust_rti.py
    python render_barrel_roll_robust_rti.py quadruped_barrel_roll_robust_rti_rollout.npz
    python render_barrel_roll_robust_rti.py rollout.npz --output rollout.mp4
    python render_barrel_roll_robust_rti.py rollout.npz --state-key X_reference
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# MuJoCo chooses its GL backend when imported.  This script renders to an
# offscreen framebuffer, so prefer EGL in headless/SSH environments.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "rti.npz"
DEFAULT_OBSTACLE_CENTER = np.array([0.0, -0.30, 0.08], dtype=np.float64)
DEFAULT_OBSTACLE_SIZE = np.array([0.80, 0.05, 0.16], dtype=np.float64)


def resolve_model_path(model_path: Path | None) -> Path:
    """Resolve an explicit model or the Go2 scene distributed with MPX."""

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


def _npz_scalar_string(value: np.ndarray) -> str:
    """Convert a scalar NumPy string/bytes array to a normal Python string."""

    scalar = np.asarray(value).reshape(()).item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


def load_obstacle_from_source(
    source_trajectory: str | None,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """Load obstacle geometry from the offline trajectory when possible."""

    if not source_trajectory:
        return DEFAULT_OBSTACLE_CENTER.copy(), DEFAULT_OBSTACLE_SIZE.copy(), None

    source_path = Path(source_trajectory).expanduser()
    if not source_path.is_absolute():
        source_path = SCRIPT_DIR / source_path
    source_path = source_path.resolve()

    if not source_path.is_file():
        print(
            "WARNING: source_trajectory does not exist; using default obstacle: "
            f"{source_path}"
        )
        return DEFAULT_OBSTACLE_CENTER.copy(), DEFAULT_OBSTACLE_SIZE.copy(), source_path

    try:
        with np.load(source_path, allow_pickle=False) as result:
            center = np.asarray(
                result.get("obstacle_center", DEFAULT_OBSTACLE_CENTER),
                dtype=np.float64,
            ).reshape(-1)
            size = np.asarray(
                result.get("obstacle_size", DEFAULT_OBSTACLE_SIZE),
                dtype=np.float64,
            ).reshape(-1)
    except Exception as error:
        print(
            "WARNING: could not read obstacle geometry from source trajectory; "
            f"using defaults. ({error})"
        )
        return DEFAULT_OBSTACLE_CENTER.copy(), DEFAULT_OBSTACLE_SIZE.copy(), source_path

    if center.size != 3 or size.size != 3:
        raise ValueError(
            "Source trajectory obstacle_center and obstacle_size must each have 3 entries."
        )
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
        raise ValueError("Source trajectory obstacle geometry must be finite.")
    if np.any(size <= 0.0):
        raise ValueError("Source trajectory obstacle dimensions must be positive.")

    return center, size, source_path


def load_rti_trajectory(
    filename: Path,
    state_key: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Path | None,
    dict[str, object],
]:
    """Load the RTI rollout, physical timing, and scene metadata."""

    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"RTI trajectory file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        if state_key not in result:
            available = ", ".join(sorted(result.files))
            raise ValueError(
                f"{filename} does not contain state key {state_key!r}. "
                f"Available keys: {available}"
            )

        states = np.asarray(result[state_key], dtype=np.float64)
        if states.ndim != 2 or states.shape[0] < 2:
            raise ValueError(
                f"{state_key} must have shape (N + 1, nx) with N >= 1; "
                f"got {states.shape}."
            )

        if "node_times" not in result:
            raise ValueError(f"{filename} does not contain node_times.")
        node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)

        source_trajectory = (
            _npz_scalar_string(result["source_trajectory"])
            if "source_trajectory" in result
            else None
        )

        metadata: dict[str, object] = {}
        if "disturbance_injected" in result:
            metadata["disturbance_injected"] = bool(
                np.asarray(result["disturbance_injected"]).reshape(()).item()
            )
        if "disturbance_scale" in result:
            metadata["disturbance_scale"] = float(
                np.asarray(result["disturbance_scale"]).reshape(()).item()
            )
        if "horizon" in result:
            metadata["horizon"] = int(np.asarray(result["horizon"]).reshape(()).item())
        if "position_errors" in result:
            errors = np.asarray(result["position_errors"], dtype=np.float64)
            if errors.size:
                metadata["max_position_error"] = float(np.max(errors))
        if "orientation_errors_deg" in result:
            errors = np.asarray(result["orientation_errors_deg"], dtype=np.float64)
            if errors.size:
                metadata["max_orientation_error_deg"] = float(np.max(errors))
        if "max_sls_tube_radius" in result:
            radii = np.asarray(result["max_sls_tube_radius"], dtype=np.float64)
            if radii.size:
                metadata["max_sls_tube_radius"] = float(np.max(radii))

        # Prefer geometry stored directly in the RTI result if a future version
        # adds it. Otherwise recover it from source_trajectory below.
        direct_center = (
            np.asarray(result["obstacle_center"], dtype=np.float64).reshape(-1)
            if "obstacle_center" in result
            else None
        )
        direct_size = (
            np.asarray(result["obstacle_size"], dtype=np.float64).reshape(-1)
            if "obstacle_size" in result
            else None
        )

    if node_times.size != states.shape[0]:
        raise ValueError(
            f"node_times has {node_times.size} entries, but {state_key} has "
            f"{states.shape[0]} states."
        )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(node_times)):
        raise ValueError("RTI states and node_times must be finite.")
    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("node_times must be strictly increasing.")

    if direct_center is not None or direct_size is not None:
        if direct_center is None or direct_size is None:
            raise ValueError(
                "RTI result must contain both obstacle_center and obstacle_size, or neither."
            )
        obstacle_center = direct_center
        obstacle_size = direct_size
        source_path = Path(source_trajectory).expanduser().resolve() if source_trajectory else None
    else:
        obstacle_center, obstacle_size, source_path = load_obstacle_from_source(
            source_trajectory
        )

    if obstacle_center.size != 3 or obstacle_size.size != 3:
        raise ValueError("Obstacle center and size must each have 3 entries.")
    if not np.all(np.isfinite(obstacle_center)) or not np.all(np.isfinite(obstacle_size)):
        raise ValueError("Obstacle geometry must be finite.")
    if np.any(obstacle_size <= 0.0):
        raise ValueError("Obstacle dimensions must be positive.")

    return (
        states,
        node_times,
        obstacle_center,
        obstacle_size,
        source_path,
        metadata,
    )


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path SLERP for MuJoCo wxyz quaternions."""

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    norm0 = np.linalg.norm(q0)
    norm1 = np.linalg.norm(q1)
    if norm0 <= 1.0e-12 or norm1 <= 1.0e-12:
        raise ValueError("Cannot interpolate a near-zero quaternion.")

    q0 = q0 / norm0
    q1 = q1 / norm1

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
    """Append the planner hurdle to the rendered MuJoCo scene."""

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
    """Render the RTI rollout to an H.264 MP4."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if states.shape[1] < model.nq + model.nv:
        raise ValueError(
            f"RTI state trajectory has {states.shape[1]} columns, but the MuJoCo "
            f"model requires at least nq + nv = {model.nq + model.nv}."
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
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # node_times are absolute physical times from the offline optimized plan.
    # Shift them to zero so truncated RTI runs also render with the correct
    # physical duration.
    render_times = node_times - node_times[0]
    duration = float(render_times[-1])
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    frame_times = np.linspace(0.0, duration, frame_count)

    try:
        for frame_time in frame_times:
            qpos, qvel = interpolate_state(
                states,
                render_times,
                frame_time,
                model.nq,
                model.nv,
            )
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            mujoco.mj_normalizeQuat(model, data.qpos)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            add_obstacle_to_scene(
                renderer.scene,
                obstacle_center,
                obstacle_size,
            )
            frame = renderer.render()

            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin unexpectedly closed.")
            encoder.stdin.write(frame.tobytes())

        if encoder.stdin is not None:
            encoder.stdin.close()
        error_output = (
            encoder.stderr.read().decode("utf-8", errors="replace")
            if encoder.stderr is not None
            else ""
        )
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
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Robust RTI rollout NPZ produced by track_barrel_roll_robust_rti.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path; defaults to <trajectory>_render.mp4.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="X_rollout",
        help=(
            "NPZ state array to render. Default: X_rollout. "
            "Use X_reference to render the offline reference stored in the RTI result."
        ),
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
            trajectory_path.stem + f"_{args.state_key}_render.mp4"
        )
    )

    (
        states,
        node_times,
        obstacle_center,
        obstacle_size,
        source_path,
        metadata,
    ) = load_rti_trajectory(trajectory_path, args.state_key)

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
        lookat=None if args.lookat is None else np.asarray(args.lookat, dtype=np.float64),
        obstacle_center=obstacle_center,
        obstacle_size=obstacle_size,
    )

    duration = float(node_times[-1] - node_times[0])
    print(f"Loaded RTI rollout from: {trajectory_path}")
    print(f"State key: {args.state_key}")
    print(f"State trajectory shape: {states.shape}")
    print(f"Model: {model_path}")
    if source_path is not None:
        print(f"Offline source trajectory: {source_path}")
    print(f"Duration: {duration:.6f} s")
    print(f"Obstacle center/size: {obstacle_center} / {obstacle_size} m")

    if "horizon" in metadata:
        print(f"RTI horizon: {metadata['horizon']}")
    if "disturbance_injected" in metadata:
        print(f"Plant disturbance injected: {metadata['disturbance_injected']}")
    if "disturbance_scale" in metadata:
        print(f"Disturbance scale: {metadata['disturbance_scale']}")
    if "max_position_error" in metadata:
        print(f"Max RTI position error: {metadata['max_position_error']:.6f} m")
    if "max_orientation_error_deg" in metadata:
        print(
            "Max RTI orientation error: "
            f"{metadata['max_orientation_error_deg']:.3f} deg"
        )
    if "max_sls_tube_radius" in metadata:
        print(f"Max SLS tube radius: {metadata['max_sls_tube_radius']:.6e}")

    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
