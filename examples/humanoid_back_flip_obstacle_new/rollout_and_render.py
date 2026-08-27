"""Roll out the optimized H1 backflip controls through the optimized dynamics
and render the resulting forward simulation to MP4.

This is intentionally different from simply rendering the saved optimized X:
it starts from the saved initial state and repeatedly applies

    x[k+1] = dynamics(x[k], U[k], k, parameter)

using the exact dynamics implementation from the optimization script.

Typical usage
-------------
# If the optimizer script is in the same directory and auto-detected:
python rollout_and_render_backflip.py

# Explicitly specify the optimizer Python file:
python rollout_and_render_backflip.py \
    humanoid_backflip_obstacle_min_time.npz \
    --optimizer-script optimize_backflip.py

# Slow-motion rendering:
python rollout_and_render_backflip.py --slowmo 4

# Save the forward rollout as a separate NPZ too:
python rollout_and_render_backflip.py --save-rollout rollout_backflip.npz
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT = SCRIPT_DIR / "humanoid_backflip_obstacle_min_time.npz"

# Same renderer used by the existing backflip rendering script.
BARREL_RENDERER_DIR = SCRIPT_DIR.parent / "quadruped_barrel_roll_obstacle"
sys.path.insert(0, str(BARREL_RENDERER_DIR))
from render_barrel_roll import render_video  # noqa: E402


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


def import_optimizer(path: Path):
    """Import the optimization script without requiring it to be a package."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Optimizer script does not exist: {path}")

    spec = importlib.util.spec_from_file_location("h1_backflip_optimizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import optimizer script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def auto_find_optimizer_script() -> Path:
    """Try a few likely optimizer filenames in this directory."""
    candidates = (
        "humanoid_backflip_obstacle.py",
        "humanoid_backflip_obstacle_min_time.py",
        "min_time_backflip.py",
        "optimize_backflip.py",
        "backflip_min_time.py",
        "main.py",
    )
    for name in candidates:
        path = SCRIPT_DIR / name
        if path.is_file() and path.resolve() != Path(__file__).resolve():
            return path

    raise FileNotFoundError(
        "Could not auto-detect the optimization script. Pass it explicitly with\n"
        "    --optimizer-script /path/to/your_optimizer.py"
    )


def validate_optimizer_module(opt):
    required = (
        "configure_problem",
        "build_backflip_reference",
        "min_time_backflip_dynamics",
        "PHYSICAL_N",
        "DURATION_SLICE",
        "QVEL_START",
        "QVEL_STOP",
        "GRF_START",
        "GRF_STOP",
        "VELOCITY_SCALE",
        "GRF_SCALE",
        "NUM_PHASES",
        "config",
    )
    missing = [name for name in required if not hasattr(opt, name)]
    if missing:
        raise AttributeError(
            "Optimizer script is missing required symbols: " + ", ".join(missing)
        )


def load_saved_solution(filename: Path):
    filename = filename.expanduser().resolve()
    if not filename.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {filename}")

    with np.load(filename, allow_pickle=False) as result:
        if "X" not in result or "U" not in result:
            raise ValueError("The NPZ must contain both X and U.")

        X_saved = np.asarray(result["X"], dtype=np.float64)
        U = np.asarray(result["U"], dtype=np.float64)

        phase_times = (
            np.asarray(result["phase_times"], dtype=np.float64).reshape(-1)
            if "phase_times" in result
            else None
        )

        duration_logits = (
            np.asarray(result["phase_duration_logits"], dtype=np.float64)
            if "phase_duration_logits" in result
            else None
        )

        if "node_times" in result:
            node_times = np.asarray(result["node_times"], dtype=np.float64).reshape(-1)
        else:
            node_times = None

        if "obstacle_centers" in result and "obstacle_sizes" in result:
            obstacle_centers = np.asarray(result["obstacle_centers"], dtype=np.float64)
            obstacle_sizes = np.asarray(result["obstacle_sizes"], dtype=np.float64)
        elif "obstacle_center" in result and "obstacle_size" in result:
            obstacle_centers = np.asarray(
                result["obstacle_center"], dtype=np.float64
            ).reshape(1, 3)
            obstacle_sizes = np.asarray(
                result["obstacle_size"], dtype=np.float64
            ).reshape(1, 3)
        else:
            obstacle_centers = None
            obstacle_sizes = None

    if X_saved.ndim != 2:
        raise ValueError(f"X must be 2-D; got {X_saved.shape}.")
    if U.ndim != 2:
        raise ValueError(f"U must be 2-D; got {U.shape}.")
    if X_saved.shape[0] != U.shape[0] + 1:
        raise ValueError(
            f"Expected X.shape[0] == U.shape[0] + 1; got "
            f"{X_saved.shape[0]} and {U.shape[0]}."
        )
    if not np.all(np.isfinite(U)):
        raise ValueError("Saved optimized controls contain non-finite values.")

    return (
        X_saved,
        U,
        phase_times,
        duration_logits,
        node_times,
        obstacle_centers,
        obstacle_sizes,
    )


def saved_state_to_internal(opt, X_saved, duration_logits, phase_times):
    """Undo the output-only unit conversion performed by the optimizer.

    The optimizer saves:
      * qvel in physical units,
      * GRFs in physical units,
      * duration coordinates as decoded physical phase durations.

    The dynamics expects:
      * qvel / VELOCITY_SCALE,
      * GRF / GRF_SCALE,
      * unconstrained duration logits.
    """
    X_internal = np.asarray(X_saved, dtype=np.float64).copy()

    velocity_scale = np.asarray(opt.VELOCITY_SCALE, dtype=np.float64)
    X_internal[:, opt.QVEL_START : opt.QVEL_STOP] /= velocity_scale
    X_internal[:, opt.GRF_START : opt.GRF_STOP] /= float(opt.GRF_SCALE)

    if duration_logits is not None:
        duration_logits = np.asarray(duration_logits, dtype=np.float64)
        if duration_logits.shape != (
            X_internal.shape[0],
            int(opt.NUM_PHASES),
        ):
            raise ValueError(
                "phase_duration_logits has unexpected shape: "
                f"{duration_logits.shape}; expected "
                f"({X_internal.shape[0]}, {int(opt.NUM_PHASES)})."
            )
        X_internal[:, opt.DURATION_SLICE] = duration_logits
    elif phase_times is not None:
        # Reconstruct logits from the optimized physical durations.
        phase_times_jax = jnp.asarray(phase_times)
        if hasattr(opt, "_duration_logits"):
            logits = np.asarray(opt._duration_logits(phase_times_jax))
        else:
            min_d = np.asarray(opt.MIN_DURATIONS)
            max_d = np.asarray(opt.MAX_DURATIONS)
            frac = (phase_times - min_d) / (max_d - min_d)
            frac = np.clip(frac, 1.0e-6, 1.0 - 1.0e-6)
            logits = np.log(frac) - np.log1p(-frac)
        X_internal[:, opt.DURATION_SLICE] = logits
    else:
        raise ValueError(
            "NPZ contains neither phase_duration_logits nor phase_times, so the "
            "optimized time coordinates cannot be reconstructed."
        )

    return X_internal


def internal_state_to_saved_units(opt, X_internal):
    """Convert a dynamics rollout to the same physical-unit layout as saved X."""
    X_out = np.asarray(X_internal, dtype=np.float64).copy()

    velocity_scale = np.asarray(opt.VELOCITY_SCALE, dtype=np.float64)
    X_out[:, opt.QVEL_START : opt.QVEL_STOP] *= velocity_scale
    X_out[:, opt.GRF_START : opt.GRF_STOP] *= float(opt.GRF_SCALE)

    # Replace duration logits with decoded physical phase durations, matching
    # the optimizer's NPZ output convention.
    X_jax = jnp.asarray(X_internal)
    durations = np.asarray(jax.vmap(opt.phase_durations)(X_jax))
    X_out[:, opt.DURATION_SLICE] = durations

    return X_out


def build_rollout_dynamics(opt):
    """Construct exactly the dynamics callback used by optimization.

    No MPC/SQP/ADMM object is created here.  We only build the MuJoCo model,
    resolve the same contact geometry/body IDs, and call the optimizer's
    min_time_backflip_dynamics factory directly.
    """
    import mujoco
    from mujoco import mjx

    # configure_problem mutates config.n/config.initial_state and installs the
    # minimum-time dynamics factory, matching the optimization setup.
    opt.configure_problem()
    _, parameter = opt.build_backflip_reference()

    model = mujoco.MjModel.from_xml_path(opt.config.model_path)
    mjx_model = mjx.put_model(model)

    contact_ids = []
    for name in opt.config.contact_frame:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Could not find contact geom {name!r} in model.")
        contact_ids.append(geom_id)

    body_ids = []
    for name in opt.config.body_name:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Could not find body {name!r} in model.")
        body_ids.append(body_id)

    dynamics = opt.min_time_backflip_dynamics(
        model,
        mjx_model,
        tuple(contact_ids),
        tuple(body_ids),
    )
    return dynamics, parameter


def rollout(dynamics, x0, U, parameter):
    """Forward simulate the optimized controls.

    Uses lax.scan so the rollout itself compiles as one JAX computation.
    """
    U_jax = jnp.asarray(U)
    x0_jax = jnp.asarray(x0)
    ts = jnp.arange(U_jax.shape[0], dtype=jnp.int32)

    @jax.jit
    def rollout_jit(x_initial, controls):
        def step(x, inp):
            u, t = inp
            x_next = dynamics(x, u, t, parameter)
            return x_next, x_next

        _, xs = jax.lax.scan(step, x_initial, (controls, ts))
        return jnp.concatenate([x_initial[None, :], xs], axis=0)

    X_roll = rollout_jit(x0_jax, U_jax)
    X_roll.block_until_ready()
    return np.asarray(X_roll)


def first_nonfinite_row(X):
    finite_rows = np.all(np.isfinite(X), axis=1)
    bad = np.flatnonzero(~finite_rows)
    return None if bad.size == 0 else int(bad[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--optimizer-script",
        type=Path,
        default=None,
        help=(
            "Python file containing the optimization problem/dynamics. "
            "If omitted, several common names are searched in this directory."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--save-rollout",
        type=Path,
        default=None,
        help="Optional NPZ path for the forward-rolled trajectory.",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument(
        "--slowmo",
        type=float,
        default=4.0,
        help="Playback slowdown factor only; rollout dynamics are unchanged.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--camera-distance", type=float, default=3.2)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--lookat", nargs=3, type=float, default=None)
    parser.add_argument(
        "--truncate-nonfinite",
        action="store_true",
        help=(
            "If the forward rollout becomes NaN/Inf, render only the finite "
            "prefix instead of raising an exception."
        ),
    )
    args = parser.parse_args()

    if not np.isfinite(args.slowmo) or args.slowmo <= 0.0:
        parser.error("--slowmo must be a finite positive number.")
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        parser.error("--fps must be a finite positive number.")

    optimizer_path = (
        args.optimizer_script.expanduser().resolve()
        if args.optimizer_script is not None
        else auto_find_optimizer_script()
    )
    opt = import_optimizer(optimizer_path)
    validate_optimizer_module(opt)

    trajectory_path = args.trajectory.expanduser().resolve()
    (
        X_saved,
        U,
        phase_times,
        duration_logits,
        saved_node_times,
        obstacle_centers,
        obstacle_sizes,
    ) = load_saved_solution(trajectory_path)

    X_internal_saved = saved_state_to_internal(
        opt, X_saved, duration_logits, phase_times
    )

    dynamics, parameter = build_rollout_dynamics(opt)

    # Roll out only from the optimized initial state. None of X_saved[1:] is
    # used to generate the forward trajectory.
    x0 = X_internal_saved[0]
    print("Compiling/running forward dynamics rollout...")
    X_roll_internal = rollout(dynamics, x0, U, parameter)

    bad_row = first_nonfinite_row(X_roll_internal)
    if bad_row is not None:
        print(f"WARNING: rollout first becomes non-finite at state node {bad_row}.")
        if not args.truncate_nonfinite:
            raise FloatingPointError(
                "Forward rollout contains non-finite values. Re-run with "
                "--truncate-nonfinite to render only the finite prefix."
            )
        if bad_row < 2:
            raise FloatingPointError(
                "Rollout becomes non-finite too early to render a useful video."
            )
        X_roll_internal = X_roll_internal[:bad_row]
        U_render = U[: bad_row - 1]
    else:
        U_render = U

    X_roll = internal_state_to_saved_units(opt, X_roll_internal)

    # Timing comes from the optimized phase durations, not from the saved X path.
    if phase_times is None:
        phase_times = np.asarray(opt.phase_durations(jnp.asarray(x0)))

    full_node_times = np.asarray(opt.phase_node_times(phase_times), dtype=np.float64)
    node_times = full_node_times[: X_roll.shape[0]]
    playback_node_times = node_times * args.slowmo

    # Quantify how much open-loop rollout diverges from the direct-transcription X.
    compare_n = min(X_roll.shape[0], X_saved.shape[0])
    qpos_dim = int(opt.config.n_joints) + 7
    qpos_err = np.linalg.norm(
        X_roll[:compare_n, :qpos_dim] - X_saved[:compare_n, :qpos_dim], axis=1
    )
    max_qpos_err = float(np.max(qpos_err))
    final_qpos_err = float(qpos_err[-1])

    output = args.output
    if output is None:
        output = trajectory_path.with_name(
            f"{trajectory_path.stem}_forward_rollout.mp4"
        )

    model_path = resolve_model_path(args.model)

    render_kwargs = dict(
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat=None if args.lookat is None else np.asarray(args.lookat),
    )
    if obstacle_centers is not None and obstacle_sizes is not None:
        render_kwargs["obstacle_center"] = obstacle_centers
        render_kwargs["obstacle_size"] = obstacle_sizes

    frame_count = render_video(
        X_roll,
        playback_node_times,
        model_path,
        output,
        **render_kwargs,
    )

    if args.save_rollout is not None:
        rollout_path = args.save_rollout.expanduser().resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            rollout_path,
            X=X_roll,
            U=U_render,
            phase_times=np.asarray(phase_times),
            node_times=node_times,
            source_trajectory=str(trajectory_path),
            optimizer_script=str(optimizer_path),
            max_qpos_difference_from_optimized_X=max_qpos_err,
            final_qpos_difference_from_optimized_X=final_qpos_err,
            obstacle_centers=(
                np.asarray(obstacle_centers)
                if obstacle_centers is not None
                else np.empty((0, 3))
            ),
            obstacle_sizes=(
                np.asarray(obstacle_sizes)
                if obstacle_sizes is not None
                else np.empty((0, 3))
            ),
        )
        print(f"Saved forward rollout: {rollout_path}")

    print()
    print(f"Optimizer script: {optimizer_path}")
    print(f"Optimized solution: {trajectory_path}")
    print(f"Model: {model_path}")
    print(f"Rollout nodes: {X_roll.shape[0]}")
    print(f"Applied controls: {U_render.shape[0]}")
    print(f"Optimized phase times: {np.asarray(phase_times)}")
    print(f"Physical rollout duration: {node_times[-1]:.6f} s")
    print(f"Slow motion: {args.slowmo:g}x")
    print(f"Playback duration: {playback_node_times[-1]:.6f} s")
    print(f"Max qpos difference vs optimized X: {max_qpos_err:.6e}")
    print(f"Final qpos difference vs optimized X: {final_qpos_err:.6e}")
    print(f"Frames: {frame_count} at {args.fps:g} fps")
    print(f"Saved video: {output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
