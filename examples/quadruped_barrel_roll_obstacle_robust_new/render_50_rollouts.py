#!/usr/bin/env python3
"""
Render 50 saved Go2 feedback rollouts simultaneously in one translucent/ghosted video.

Designed for the NPZ produced by:
    robust_rollouts_random_adversarial_basis_pm1.py

By default, the script chooses the 50 trajectories with the largest maximum XYZ
base-position deviation from the nominal trajectory across all available rollout groups
(random, adversarial, and basis +/- tests), then overlays them in the same MuJoCo
scene.  Each robot is alpha-composited against a robot-free background, so individual
quadrupeds remain see-through while regions where many rollouts overlap become denser.

Examples
--------
    python render_50_rollouts_overlay.py

    python render_50_rollouts_overlay.py \
        quadruped_barrel_roll_random_adversarial_basis_pm1_feedback_rollouts.npz

    # Only random rollouts
    python render_50_rollouts_overlay.py --source random

    # Only adversarial rollouts, with lighter ghosts
    python render_50_rollouts_overlay.py --source adversarial --ghost-alpha 0.06

    # Choose 50 trajectories with largest XYZ position deviation (default)
    python render_50_rollouts_overlay.py --selection position

    # Or choose 50 worst trajectories by saved max tube ratio
    python render_50_rollouts_overlay.py --selection worst
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# MuJoCo chooses its GL backend at import time.  This script renders offscreen.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

EXPERIMENT_DIR = Path(
    "/home/jeff/trustworthroboticsgroup/ICRA2026/min_time/"
    "gpu_sls/examples/quadruped_barrel_roll_obstacle_robust_new"
)

DEFAULT_INPUT = (
    EXPERIMENT_DIR
    / "disturbance_rollouts.npz"
)

DEFAULT_OBSTACLE_CENTER = np.array([0.0, -0.30, 0.08], dtype=np.float64)
DEFAULT_OBSTACLE_SIZE = np.array([0.80, 0.05, 0.16], dtype=np.float64)


# -----------------------------------------------------------------------------
# Model / trajectory loading
# -----------------------------------------------------------------------------


def resolve_model_path(model_path: Path | None) -> Path:
    """Resolve an explicit Go2 XML or the scene distributed with MPX."""

    if model_path is not None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
        return path

    try:
        import mpx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import mpx to locate the Go2 model. Install the gpu_sls "
            "project dependencies or pass --model /path/to/scene.xml."
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
            "The final phase_end_step must equal X.shape[1] - 1; got "
            f"{phase_end_steps[-1]} and {state_count - 1}."
        )

    transition_dt = np.repeat(
        phase_times / segment_lengths,
        segment_lengths,
    )

    return np.concatenate([[0.0], np.cumsum(transition_dt)])



def _append_batch(
    batches: list[np.ndarray],
    scores: list[np.ndarray],
    names: list[str],
    result,
    state_key: str,
    score_key: str,
    label: str,
) -> None:
    if state_key not in result:
        return

    X = np.asarray(result[state_key], dtype=np.float64)

    if X.ndim != 3:
        raise ValueError(
            f"{state_key} must have shape (R, N+1, nx); got {X.shape}."
        )

    batches.append(X)
    names.extend([label] * X.shape[0])

    if score_key in result:
        s = np.asarray(result[score_key], dtype=np.float64).reshape(-1)
        if s.size != X.shape[0]:
            raise ValueError(
                f"{score_key} has {s.size} entries but {state_key} has "
                f"{X.shape[0]} rollouts."
            )
    else:
        s = np.full(X.shape[0], np.nan, dtype=np.float64)

    scores.append(s)



def load_rollouts(
    filename: Path,
    source: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load available rollout batches and combine the requested source groups.

    Returns
    -------
    rollouts : (R, N+1, nx)
    scores : (R,)
        Saved max |dx| / tube score where available.
    source_names : (R,)
    node_times : (N+1,)
    obstacle_center : (3,)
    obstacle_size : (3,)
    """

    filename = filename.expanduser().resolve()

    if not filename.is_file():
        raise FileNotFoundError(f"Rollout NPZ does not exist: {filename}")

    batches: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    names: list[str] = []

    with np.load(filename, allow_pickle=False) as result:
        requested = {source} if source != "mixed" else {
            "random",
            "adversarial",
            "basis",
        }

        if "random" in requested:
            _append_batch(
                batches,
                scores,
                names,
                result,
                "X_random_feedback_rollouts",
                "max_tube_ratios_random",
                "random",
            )

        if "adversarial" in requested:
            _append_batch(
                batches,
                scores,
                names,
                result,
                "X_adversarial_feedback_rollouts",
                "max_tube_ratios_adversarial",
                "adversarial",
            )

        if "basis" in requested:
            _append_batch(
                batches,
                scores,
                names,
                result,
                "X_basis_pm1_feedback_rollouts",
                "max_tube_ratios_basis_pm1",
                "basis",
            )

        if not batches:
            available = list(result.files)
            raise ValueError(
                f"No rollout arrays for source={source!r} were found in {filename}. "
                f"Available keys include: {available}"
            )

        state_count = batches[0].shape[1]
        state_dim = batches[0].shape[2]

        for batch in batches[1:]:
            if batch.shape[1:] != (state_count, state_dim):
                raise ValueError(
                    "All selected rollout batches must use identical trajectory "
                    f"shapes; got {batch.shape[1:]} vs {(state_count, state_dim)}."
                )

        if "node_times" in result:
            node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)
        elif "phase_times" in result and "phase_end_steps" in result:
            node_times = reconstruct_node_times(
                np.asarray(result["phase_times"], dtype=np.float64),
                np.asarray(result["phase_end_steps"], dtype=np.int64),
                state_count,
            )
        else:
            raise ValueError(
                "The NPZ needs node_times or both phase_times and phase_end_steps."
            )

        obstacle_center = np.asarray(
            result.get("obstacle_center", DEFAULT_OBSTACLE_CENTER),
            dtype=np.float64,
        ).reshape(-1)

        obstacle_size = np.asarray(
            result.get("obstacle_size", DEFAULT_OBSTACLE_SIZE),
            dtype=np.float64,
        ).reshape(-1)

    rollouts = np.concatenate(batches, axis=0)
    score_array = np.concatenate(scores, axis=0)
    source_names = np.asarray(names)

    if node_times.size != rollouts.shape[1]:
        raise ValueError(
            f"node_times has {node_times.size} entries but rollouts have "
            f"{rollouts.shape[1]} state nodes."
        )

    if np.any(np.diff(node_times) <= 0.0):
        raise ValueError("node_times must be strictly increasing.")

    if not np.all(np.isfinite(rollouts)):
        raise ValueError("Selected rollout states contain non-finite values.")

    if obstacle_center.size != 3 or obstacle_size.size != 3:
        raise ValueError("Obstacle center/size must each contain three values.")

    return (
        rollouts,
        score_array,
        source_names,
        node_times,
        obstacle_center,
        obstacle_size,
    )


def load_nominal_trajectory(
    filename: Path,
    state_count: int,
    state_dim: int,
) -> tuple[np.ndarray, str]:
    """Load the nominal/reference state trajectory used to define rollout error.

    The rollout NPZs used by related scripts have used a few different key names,
    so try the common variants.  The returned array always has shape
    (N+1, nx).
    """

    candidate_keys = (
        "X_nominal",
        "X_nom",
        "X_ref",
        "X_reference",
        "nominal_X",
        "nominal_states",
        "X",
    )

    with np.load(filename, allow_pickle=False) as result:
        for key in candidate_keys:
            if key not in result:
                continue

            X_nominal = np.asarray(result[key], dtype=np.float64)

            # Expected layout: (N+1, nx).  Accept a transposed saved array too.
            if X_nominal.ndim != 2:
                continue

            if X_nominal.shape == (state_count, state_dim):
                return X_nominal, key

            if X_nominal.shape == (state_dim, state_count):
                return X_nominal.T, key

        available = list(result.files)

    raise ValueError(
        "Could not find a nominal trajectory in the rollout NPZ. Tried keys "
        f"{candidate_keys}. Available keys: {available}"
    )


def max_xyz_position_delta_scores(
    rollouts: np.ndarray,
    nominal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each rollout's maximum 3-D base-position deviation.

    score[r] = max_k || p_rollout[r,k] - p_nominal[k] ||_2,
    where p = [x, y, z].

    Also returns the time index of the maximum and the corresponding xyz error.
    """

    if nominal.shape[0] != rollouts.shape[1]:
        raise ValueError(
            f"Nominal has {nominal.shape[0]} nodes but rollouts have "
            f"{rollouts.shape[1]}."
        )

    if nominal.shape[1] < 3 or rollouts.shape[2] < 3:
        raise ValueError("Need at least x/y/z in the first three state columns.")

    delta_xyz = rollouts[:, :, :3] - nominal[None, :, :3]
    delta_norm = np.linalg.norm(delta_xyz, axis=2)

    max_step = np.argmax(delta_norm, axis=1)
    rollout_ids = np.arange(rollouts.shape[0])
    max_score = delta_norm[rollout_ids, max_step]
    max_delta_xyz = delta_xyz[rollout_ids, max_step]

    return max_score, max_step, max_delta_xyz


# -----------------------------------------------------------------------------
# Rollout selection
# -----------------------------------------------------------------------------


def choose_rollouts(
    rollouts: np.ndarray,
    scores: np.ndarray,
    source_names: np.ndarray,
    count: int,
    selection: str,
    seed: int,
    nominal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Select exactly `count` trajectories from the loaded rollout population.

    For selection="position", rank by
        max_k ||rollout[k, 0:3] - nominal[k, 0:3]||_2.
    """

    total = rollouts.shape[0]
    selection_scores = None

    if count <= 0:
        raise ValueError("--num-rollouts must be positive.")

    if count > total:
        raise ValueError(
            f"Requested {count} rollouts but source contains only {total}."
        )

    if selection == "first":
        indices = np.arange(count, dtype=np.int64)

    elif selection == "worst":
        finite_score = np.where(np.isfinite(scores), scores, -np.inf)
        order = np.argsort(finite_score)[::-1]
        indices = order[:count]
        selection_scores = finite_score

    elif selection == "position":
        if nominal is None:
            raise ValueError(
                "selection='position' requires the nominal trajectory."
            )

        position_scores, _, _ = max_xyz_position_delta_scores(
            rollouts, nominal
        )
        order = np.argsort(position_scores)[::-1]
        indices = order[:count]
        selection_scores = position_scores

    elif selection == "random":
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(total, size=count, replace=False).astype(np.int64)
        )

    else:
        raise ValueError(f"Unknown selection mode: {selection}")

    selected_scores = (
        None if selection_scores is None else selection_scores[indices]
    )

    return rollouts[indices], source_names[indices], indices, selected_scores


# -----------------------------------------------------------------------------
# State interpolation
# -----------------------------------------------------------------------------


def quaternion_slerp(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: float,
) -> np.ndarray:
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



def interpolation_indices(
    node_times: np.ndarray,
    frame_time: float,
) -> tuple[int, int, float]:
    if frame_time <= node_times[0]:
        return 0, 0, 0.0

    if frame_time >= node_times[-1]:
        last = len(node_times) - 1
        return last, last, 0.0

    right = int(np.searchsorted(node_times, frame_time, side="right"))
    left = right - 1
    alpha = float(
        (frame_time - node_times[left])
        / (node_times[right] - node_times[left])
    )

    return left, right, alpha



def interpolate_qpos_qvel(
    states: np.ndarray,
    node_times: np.ndarray,
    frame_time: float,
    nq: int,
    nv: int,
) -> tuple[np.ndarray, np.ndarray]:
    left, right, alpha = interpolation_indices(node_times, frame_time)

    qpos = (
        (1.0 - alpha) * states[left, :nq]
        + alpha * states[right, :nq]
    )

    qpos = qpos.copy()
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


# -----------------------------------------------------------------------------
# MuJoCo scene helpers
# -----------------------------------------------------------------------------


def add_obstacle_to_scene(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    size: np.ndarray,
    rgba: np.ndarray | None = None,
) -> None:
    """Append the optimized hurdle to the rendered scene."""

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo render scene has no room for the obstacle.")

    geom = scene.geoms[scene.ngeom]

    if rgba is None:
        rgba = np.array([0.90, 0.20, 0.08, 1.0], dtype=np.float32)

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        0.5 * np.asarray(size, dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )

    scene.ngeom += 1



def set_robot_scene_alpha(
    scene: mujoco.MjvScene,
    model: mujoco.MjModel,
    alpha: float,
) -> None:
    """
    Set alpha for model geoms attached to non-world bodies.

    The ground/world remains untouched.  The custom obstacle added after
    update_scene is also untouched because it is not a model geom.
    """

    alpha = float(np.clip(alpha, 0.0, 1.0))

    for scene_geom_id in range(scene.ngeom):
        geom = scene.geoms[scene_geom_id]

        try:
            objtype = int(geom.objtype)
            objid = int(geom.objid)
        except Exception:
            continue

        if objtype != int(mujoco.mjtObj.mjOBJ_GEOM):
            continue

        if objid < 0 or objid >= model.ngeom:
            continue

        if int(model.geom_bodyid[objid]) == 0:
            continue

        geom.rgba[3] = alpha



def disable_scene_shadows(scene: mujoco.MjvScene) -> None:
    """Disable shadows when supported, which makes ghost compositing cleaner."""

    try:
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    except Exception:
        # MuJoCo enum/scene APIs vary slightly across releases.  Shadows are only
        # a visual nicety here, so rendering should continue if this is unavailable.
        pass



def update_scene_for_state(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    qpos: np.ndarray,
    qvel: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_size: np.ndarray,
    robot_alpha: float,
) -> np.ndarray:
    """Render one robot state with a requested model-geom alpha."""

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=camera)
    disable_scene_shadows(renderer.scene)

    add_obstacle_to_scene(
        renderer.scene,
        obstacle_center,
        obstacle_size,
    )

    set_robot_scene_alpha(
        renderer.scene,
        model,
        robot_alpha,
    )

    return renderer.render().copy()


# -----------------------------------------------------------------------------
# Ghost compositing
# -----------------------------------------------------------------------------


def composite_ghost(
    composite: np.ndarray,
    background: np.ndarray,
    robot_frame: np.ndarray,
    ghost_alpha: float,
    mask_threshold: float,
    mask_softness: float,
) -> np.ndarray:
    """
    Alpha-composite one robot onto the common background.

    The mask is obtained from the RGB difference between an opaque robot render
    and a robot-free render.  This avoids needing to duplicate the Go2 model 50
    times inside one MuJoCo XML.  Repeatedly overlapping ghosts naturally become
    more opaque, making rollout density visible.
    """

    bg = background.astype(np.float32)
    fg = robot_frame.astype(np.float32)
    out = composite.astype(np.float32)

    diff = np.max(np.abs(fg - bg), axis=2)

    if mask_softness <= 0.0:
        coverage = (diff > mask_threshold).astype(np.float32)
    else:
        coverage = np.clip(
            (diff - mask_threshold) / mask_softness,
            0.0,
            1.0,
        ).astype(np.float32)

    alpha = ghost_alpha * coverage[..., None]
    out = (1.0 - alpha) * out + alpha * fg

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Video renderer
# -----------------------------------------------------------------------------


def render_overlay_video(
    rollouts: np.ndarray,
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
    ghost_alpha: float,
    mask_threshold: float,
    mask_softness: float,
) -> int:
    """Render all selected rollouts into one ghosted MP4."""

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH.")

    if fps <= 0.0:
        raise ValueError("fps must be positive.")

    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("width and height must be positive even integers.")

    if not (0.0 < ghost_alpha <= 1.0):
        raise ValueError("ghost_alpha must lie in (0, 1].")

    model = mujoco.MjModel.from_xml_path(str(model_path))

    if rollouts.shape[2] < model.nq + model.nv:
        raise ValueError(
            f"Rollouts have {rollouts.shape[2]} state columns but model requires "
            f"at least nq+nv={model.nq + model.nv}."
        )

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    if lookat is None:
        base_positions = rollouts[:, :, :3].reshape(-1, 3)
        lookat = 0.5 * (
            base_positions.min(axis=0)
            + base_positions.max(axis=0)
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
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    encoder = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    duration = float(node_times[-1])
    frame_count = max(2, int(np.ceil(duration * fps)) + 1)
    frame_times = np.linspace(0.0, duration, frame_count)

    try:
        # The camera/world/obstacle are static.  Render a robot-free background
        # once and use it for every frame's ghost mask/compositing.
        qpos_bg, qvel_bg = interpolate_qpos_qvel(
            rollouts[0],
            node_times,
            0.0,
            model.nq,
            model.nv,
        )

        background = update_scene_for_state(
            renderer,
            model,
            data,
            camera,
            qpos_bg,
            qvel_bg,
            obstacle_center,
            obstacle_size,
            robot_alpha=0.0,
        )

        for frame_index, frame_time in enumerate(frame_times):
            composite = background.copy()

            for rollout in rollouts:
                qpos, qvel = interpolate_qpos_qvel(
                    rollout,
                    node_times,
                    float(frame_time),
                    model.nq,
                    model.nv,
                )

                opaque_frame = update_scene_for_state(
                    renderer,
                    model,
                    data,
                    camera,
                    qpos,
                    qvel,
                    obstacle_center,
                    obstacle_size,
                    robot_alpha=1.0,
                )

                composite = composite_ghost(
                    composite,
                    background,
                    opaque_frame,
                    ghost_alpha,
                    mask_threshold,
                    mask_softness,
                )

            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin closed unexpectedly.")

            encoder.stdin.write(composite.tobytes())

            if frame_index % max(1, int(round(fps))) == 0:
                print(
                    f"Rendered frame {frame_index + 1}/{frame_count} "
                    f"(t={frame_time:.3f} s)"
                )

        if encoder.stdin is not None:
            encoder.stdin.close()

        error_output = encoder.stderr.read().decode(
            "utf-8",
            errors="replace",
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="NPZ containing saved rollout batches.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4; defaults next to the input NPZ.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Go2 scene XML; defaults to the model supplied by MPX.",
    )

    parser.add_argument(
        "--source",
        choices=("mixed", "random", "adversarial", "basis"),
        default="mixed",
        help=(
            "Which rollout population to sample from. 'mixed' combines all "
            "available random/adversarial/basis rollouts."
        ),
    )

    parser.add_argument(
        "--selection",
        choices=("position", "random", "worst", "first"),
        default="position",
        help=(
            "How to choose displayed rollouts. 'position' chooses the largest "
            "max Euclidean XYZ deviation from the nominal trajectory."
        ),
    )

    parser.add_argument("--num-rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--fps", type=float, default=30.0)
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
        help="Camera target; defaults to center of all selected base trajectories.",
    )

    parser.add_argument(
        "--ghost-alpha",
        type=float,
        default=0.08,
        help=(
            "Opacity contribution of each rollout. Around 0.05-0.12 works well "
            "for 50 trajectories."
        ),
    )

    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=4.0,
        help="RGB difference below this value is treated as background.",
    )

    parser.add_argument(
        "--mask-softness",
        type=float,
        default=20.0,
        help="Soft transition width for the robot/background difference mask.",
    )

    args = parser.parse_args()

    trajectory_path = args.trajectory.expanduser().resolve()

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trajectory_path.with_name(
            trajectory_path.stem
            + f"_{args.num_rollouts}_{args.source}_ghost_rollouts.mp4"
        )
    )

    (
        all_rollouts,
        all_scores,
        all_source_names,
        node_times,
        obstacle_center,
        obstacle_size,
    ) = load_rollouts(
        trajectory_path,
        args.source,
    )

    nominal = None
    nominal_key = None
    if args.selection == "position":
        nominal, nominal_key = load_nominal_trajectory(
            trajectory_path,
            all_rollouts.shape[1],
            all_rollouts.shape[2],
        )

    (
        selected_rollouts,
        selected_sources,
        selected_indices,
        selected_selection_scores,
    ) = choose_rollouts(
        all_rollouts,
        all_scores,
        all_source_names,
        args.num_rollouts,
        args.selection,
        args.seed,
        nominal=nominal,
    )

    model_path = resolve_model_path(args.model)

    unique_sources, source_counts = np.unique(
        selected_sources,
        return_counts=True,
    )

    print(f"Input:             {trajectory_path}")
    print(f"Model:             {model_path}")
    print(f"Source population: {args.source}")
    print(f"Selection:         {args.selection}")
    if nominal_key is not None:
        print(f"Nominal key:       {nominal_key}")
    print(f"Selected rollouts: {selected_rollouts.shape[0]}")
    print(
        "Selected groups:   "
        + ", ".join(
            f"{name}={count}"
            for name, count in zip(unique_sources, source_counts)
        )
    )
    print(f"Selected indices:  {selected_indices.tolist()}")

    if args.selection == "position":
        all_position_scores, all_max_steps, all_max_delta_xyz = (
            max_xyz_position_delta_scores(all_rollouts, nominal)
        )
        print("Selected max XYZ position deltas:")
        for rank, idx in enumerate(selected_indices, start=1):
            dxyz = all_max_delta_xyz[idx]
            print(
                f"  {rank:02d}. rollout={int(idx):4d} "
                f"source={all_source_names[idx]:11s} "
                f"max_norm={all_position_scores[idx]:.6f} m "
                f"step={int(all_max_steps[idx]):3d} "
                f"dxyz=[{dxyz[0]:+.6f}, {dxyz[1]:+.6f}, {dxyz[2]:+.6f}] m"
            )

    print(f"Duration:          {node_times[-1]:.6f} s")
    print(f"Ghost alpha:       {args.ghost_alpha:.3f}")
    print(f"Output:            {output_path}")

    frame_count = render_overlay_video(
        selected_rollouts,
        node_times,
        model_path,
        output_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=(
            None
            if args.lookat is None
            else np.asarray(args.lookat, dtype=np.float64)
        ),
        obstacle_center=obstacle_center,
        obstacle_size=obstacle_size,
        ghost_alpha=args.ghost_alpha,
        mask_threshold=args.mask_threshold,
        mask_softness=args.mask_softness,
    )

    print(f"Frames:            {frame_count} at {args.fps:g} fps")
    print(f"Saved video:       {output_path}")


if __name__ == "__main__":
    main()