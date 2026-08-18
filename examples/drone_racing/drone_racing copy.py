"""Minimum-time drone racing with GPU-SLS in an RTI-MPC loop."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Select MuJoCo's headless renderer before Crazyflow imports MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")

# Prefer this checkout over unrelated editable gpu_sls installations.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crazyflow.dynamics.core import load_params, parametrize
from crazyflow.dynamics.first_principles import dynamics as crazyflow_dynamics
from drone_sim import DRONE_MODEL, GATES, close, dynamics, get_sim, render, step
from gpu_sls.generic_mpc import GenericMPC, MPCConfig
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig
from gpu_sls.utils.constraint_utils import (
    combine_constraints,
    make_control_box_constraints,
    make_state_box_constraints,
)


# Planner state: [pos(3), quat_xyzw(4), vel(3), body_rates(3),
#                 rotor_rpm(4), total_horizon_time]
NX = 18
NU = 4
HORIZON = 30
MAX_RTI_STEPS = 250
GOAL_TOLERANCE = 0.20
GATE_TOLERANCE = 0.55
MIN_DURATION = 0.005
MAX_DURATION = 8.0
MAX_ROTOR_RPM = 75_000.0
MODEL_SUBSTEPS = 1
INITIAL_SQP_ITERATIONS = 100
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30.0
DEFAULT_VIDEO_PATH = Path(__file__).with_name("drone_race.mp4")

FIRST_PRINCIPLES = parametrize(crazyflow_dynamics, DRONE_MODEL, xp=jnp)
FIRST_PRINCIPLES_PARAMS = load_params(crazyflow_dynamics, DRONE_MODEL)


def _hover_rpm() -> float:
    coefficients = np.asarray(FIRST_PRINCIPLES_PARAMS["rpm2thrust"], dtype=float).copy()
    coefficients[0] -= (
        float(FIRST_PRINCIPLES_PARAMS["mass"]) * 9.81 / 4.0
    )
    roots = np.roots(coefficients[::-1])
    positive_roots = roots[np.isreal(roots) & (roots.real > 0)].real
    return float(positive_roots.min())


HOVER_RPM = _hover_rpm()
HOVER_COMMAND = HOVER_RPM / MAX_ROTOR_RPM


def initialize_hover(position=(0.0, 0.0, 0.75)):
    """Reset Crazyflow to a stationary airborne hover equilibrium."""
    sim = get_sim()
    sim.reset()
    states = sim.data.states
    hover_position = jnp.asarray(position, dtype=states.pos.dtype)
    identity_quaternion = jnp.asarray(
        [0.0, 0.0, 0.0, 1.0], dtype=states.quat.dtype
    )
    states = states.replace(
        pos=states.pos.at[...].set(hover_position),
        quat=states.quat.at[...].set(identity_quaternion),
        vel=jnp.zeros_like(states.vel),
        ang_vel=jnp.zeros_like(states.ang_vel),
        rotor_vel=jnp.full_like(states.rotor_vel, HOVER_RPM),
    )
    sim.data = sim.data.replace(
        states=states,
        core=sim.data.core.replace(
            mjx_synced=jnp.zeros_like(sim.data.core.mjx_synced)
        ),
    )
    dynamics(np.full(NU, HOVER_RPM, dtype=np.float64))
    sim.default_data = sim.data.replace()
    return sim


class H264VideoWriter:
    """Stream RGB frames to FFmpeg as a browser-compatible H.264 MP4."""

    def __init__(self, path: Path):
        self.path = path
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel", "error",
                "-y",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
                "-framerate", str(VIDEO_FPS),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Cannot write to a finalized video")
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        if self._process.stdin is None:
            return
        self._process.stdin.close()
        self._process.stdin = None
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {return_code}")


def open_video_writer(path: str | Path) -> tuple[Path, H264VideoWriter]:
    """Create an H.264 MP4 writer and finalize it if optimization is interrupted."""
    output_path = Path(path)
    writer = H264VideoWriter(output_path)
    atexit.register(writer.release)
    return output_path, writer


def save_video_frame(writer: H264VideoWriter, repeat: int = 1) -> None:
    """Render the race camera off-screen and append it to the video."""
    frame = render(
        mode="rgb_array",
        camera="race_camera",
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
    )
    if frame is None:
        raise RuntimeError("Off-screen rendering did not return an RGB frame")
    frame = np.asarray(frame)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
        raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")
    for _ in range(repeat):
        writer.write(frame)


@dataclass(frozen=True)
class RTIResult:
    reached_goal: bool
    iterations: int
    simulated_time: float
    final_position: np.ndarray


def crossed_gate(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    gate,
    half_width: float = GATE_TOLERANCE,
) -> bool:
    """Return True when the motion segment crosses through the gate opening."""
    roll, pitch, yaw = gate.euler
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Gate body rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )

    center = np.asarray(gate.position, dtype=np.float64)
    normal = rotation[:, 0]  # local +x axis is normal to the gate plane

    previous_distance = float(np.dot(previous_position - center, normal))
    current_distance = float(np.dot(current_position - center, normal))

    # The executed motion must actually cross the plane.
    crossed_plane = (
        (previous_distance < 0.0 <= current_distance)
        or (previous_distance > 0.0 >= current_distance)
    )
    if not crossed_plane:
        return False

    # Interpolate the exact plane-crossing point along the executed segment.
    alpha = previous_distance / (previous_distance - current_distance)
    crossing_position = previous_position + alpha * (current_position - previous_position)

    # Express the crossing point in gate-local coordinates and require it to be
    # inside the opening. The local x coordinate is approximately zero here.
    local = rotation.T @ (crossing_position - center)
    return bool(abs(local[1]) <= half_width and abs(local[2]) <= half_width)


def min_time_dynamics(
    x: jax.Array,
    u: jax.Array,
    _t: jax.Array,
    *,
    parameter: float,
) -> jax.Array:
    """Crazyflow's first-principles model with rotor and quaternion dynamics."""
    dt = parameter * x[-1] / MODEL_SUBSTEPS

    def integrate_substep(_index: int, state: jax.Array) -> jax.Array:
        pos = state[:3]
        quat = state[3:7]
        vel = state[7:10]
        ang_vel = state[10:13]
        rotor_command = u * MAX_ROTOR_RPM
        rotor_vel = state[13:17] * MAX_ROTOR_RPM
        pos_dot, _, vel_dot, ang_vel_dot, rotor_vel_dot = FIRST_PRINCIPLES(
            pos=pos,
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
            cmd=rotor_command,
            rotor_vel=rotor_vel,
        )

        rotation_vector = ang_vel * dt
        # The epsilon keeps the Jacobian finite at zero body rate.
        angle = jnp.sqrt(jnp.dot(rotation_vector, rotation_vector) + 1e-12)
        delta_quat = jnp.concatenate(
            (
                0.5
                * jnp.sinc(angle / (2.0 * jnp.pi))
                * rotation_vector,
                jnp.cos(0.5 * angle)[None],
            )
        )
        q_xyz, q_w = quat[:3], quat[3]
        d_xyz, d_w = delta_quat[:3], delta_quat[3]
        next_quat = jnp.concatenate(
            (
                q_w * d_xyz + d_w * q_xyz + jnp.cross(q_xyz, d_xyz),
                (q_w * d_w - jnp.dot(q_xyz, d_xyz))[None],
            )
        )
        next_quat = next_quat / jnp.sqrt(jnp.dot(next_quat, next_quat) + 1e-12)
        return jnp.concatenate(
            (
                pos + dt * pos_dot,
                next_quat,
                vel + dt * vel_dot,
                ang_vel + dt * ang_vel_dot,
                state[13:17] + dt * rotor_vel_dot / MAX_ROTOR_RPM,
                state[-1:],
            )
        )

    return jax.lax.fori_loop(0, MODEL_SUBSTEPS, integrate_substep, x)


def min_time_cost(
    weights: jax.Array,
    reference: jax.Array,
    x: jax.Array,
    u: jax.Array,
    t: jax.Array,
) -> jax.Array:
    position_error = x[:3] - reference[t, :3]
    quaternion_alignment = jnp.dot(x[3:7], reference[t, 3:7])
    velocity_error = x[7:10] - reference[t, 7:10]
    body_rate_error = x[10:13] - reference[t, 10:13]
    rotor_command_error = u - HOVER_COMMAND
    return (
        weights[0] * jnp.dot(position_error, position_error)
        + weights[1] * (1.0 - quaternion_alignment**2)
        + weights[2] * jnp.dot(velocity_error, velocity_error)
        + weights[3] * jnp.dot(body_rate_error, body_rate_error)
        + weights[4] * jnp.dot(rotor_command_error, rotor_command_error)
        + weights[5] * x[-1]
    )


def make_terminal_constraint(goal: jax.Array, half_width: float):
    goal = jnp.asarray(goal)

    def constraint(x: jax.Array, _u: jax.Array, t: jax.Array) -> jax.Array:
        error = x[:3] - goal
        terminal = jnp.concatenate((error - half_width, -error - half_width))
        return jnp.where(t == HORIZON, terminal, -jnp.ones_like(terminal))

    return constraint

def make_time_constraints(min_duration: float, max_duration: float):
    """Enforce min_duration <= T <= max_duration, where T = x[-1]."""

    def constraint(
        x: jax.Array,
        _u: jax.Array,
        _t: jax.Array,
    ) -> jax.Array:
        T = x[-1]

        # Constraints are represented as g(x, u) <= 0.
        return jnp.array([
            min_duration - T,  # T >= min_duration
            T - max_duration,  # T <= max_duration
        ], dtype=x.dtype)

    return constraint


def disturbance(x_trajectory: jax.Array) -> jax.Array:
    """Small process-noise set used for SLS tube construction."""
    dt = x_trajectory[:, -1] / HORIZON
    identity = jnp.eye(NX, dtype=x_trajectory.dtype)
    matrices = 1e-3 * dt[:, None, None] * identity[None, :, :]
    return matrices.at[:, -1, -1].set(0.0)


def build_reference(
    x0: jax.Array,
    waypoints: np.ndarray,
    duration: float,
) -> jax.Array:
    """Sample a constant-speed polyline through every remaining gate."""
    points = np.vstack((np.asarray(x0[:3]), np.asarray(waypoints)))
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = max(float(cumulative[-1]), 1e-6)
    samples = np.linspace(0.0, total_length, HORIZON + 1)
    positions = np.empty((HORIZON + 1, 3), dtype=np.float64)

    for index, distance in enumerate(samples):
        segment = min(np.searchsorted(cumulative, distance, side="right") - 1, len(segment_lengths) - 1)
        segment = max(segment, 0)
        length = max(float(segment_lengths[segment]), 1e-6)
        alpha = (distance - cumulative[segment]) / length
        positions[index] = points[segment] + alpha * segment_vectors[segment]

    dt = max(duration / HORIZON, 1e-3)
    velocities = np.gradient(positions, dt, axis=0)
    reference = np.zeros((HORIZON + 1, NX), dtype=np.float64)
    reference[:, :3] = positions
    reference[:, 6] = 1.0  # identity quaternion in xyzw convention
    reference[:, 7:10] = velocities
    reference[:, 13:17] = HOVER_COMMAND
    reference[:, -1] = duration
    return jnp.asarray(reference)


def measured_state(duration: float) -> jax.Array:
    sim = get_sim()
    position = np.asarray(sim.data.states.pos[0, 0])
    quaternion = np.asarray(sim.data.states.quat[0, 0])
    velocity = np.asarray(sim.data.states.vel[0, 0])
    angular_velocity = np.asarray(sim.data.states.ang_vel[0, 0])
    rotor_velocity = np.asarray(sim.data.states.rotor_vel[0, 0])
    return jnp.asarray(
        np.concatenate(
            (
                position,
                quaternion,
                velocity,
                angular_velocity,
                rotor_velocity / MAX_ROTOR_RPM,
                [duration],
            )
        ),
        dtype=jnp.float64,
    )


def build_controller(
    reference: jax.Array,
    goal: jax.Array,
    *,
    initialize_nominal: bool,
    X_in: jax.Array | None = None,
    U_in: jax.Array | None = None,
) -> GenericMPC:
    weights = jnp.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1.0], dtype=jnp.float64)
    config = MPCConfig(
        n=NX,
        nu=NU,
        N=HORIZON,
        W=weights,
        u_ref=jnp.full((NU,), HOVER_COMMAND, dtype=jnp.float64),
    )

    u_max = jnp.ones(NU, dtype=jnp.float64)
    u_min = jnp.zeros(NU, dtype=jnp.float64)
    x_min = jnp.array(
        [
            -20.0, -20.0, 0.0,
            -1.0, -1.0, -1.0, -1.0,
            -8.0, -8.0, -5.0,
            -20.0, -20.0, -20.0,
            0.0, 0.0, 0.0, 0.0,
            MIN_DURATION,
        ],
        dtype=jnp.float64,
    )
    x_max = jnp.array(
        [
            30.0, 20.0, 8.0,
            1.0, 1.0, 1.0, 1.0,
            8.0, 8.0, 5.0,
            20.0, 20.0, 20.0,
            1.0, 1.0, 1.0, 1.0,
            MAX_DURATION,
        ],
        dtype=jnp.float64,
    )
    constraints = combine_constraints(
        # make_state_box_constraints(x_min, x_max),
        # make_time_constraints(MIN_DURATION, MAX_DURATION),
        make_control_box_constraints(u_min, u_max),
        make_terminal_constraint(goal, half_width=0.835),
    )

    if X_in is None:
        X_in = reference
    if U_in is None:
        U_in = jnp.full((HORIZON, NU), HOVER_COMMAND, dtype=jnp.float64)

    return GenericMPC(
        SLSConfig(
            max_sls_iterations=1,
            sls_primal_tol=1e-2,
            enable_fastsls=False,
            initialize_nominal=initialize_nominal,
            max_initial_sqp_iterations=(
                INITIAL_SQP_ITERATIONS if initialize_nominal else 0
            ),
            warm_start=True,
            rti=False,
            gradient_window=0,
        ),
        SQPConfig(
            # After the initial nominal solve, RTI uses one SQP step per MPC update.
            max_sqp_iterations=1,
            warm_start=True,
            feas_tol=1e-4,
            step_tol=1e-4,
            line_search=True if initialize_nominal else False,
        ),
        ADMMConfig(
            eps_abs=5e-2,
            eps_rel=1e-3,
            max_iterations=250,
            rho_update_frequency=25,
            initial_rho=1.0,
        ),
        config=config,
        dynamics=min_time_dynamics,
        constraints=constraints,
        obstacles=jnp.zeros((0, 3), dtype=jnp.float64),
        cost=min_time_cost,
        disturbance=disturbance,
        shift=1,
        X_in=X_in,
        U_in=U_in,
    )


def run_rti_mpc(
    max_iterations: int = MAX_RTI_STEPS,
    video_path: str | Path = DEFAULT_VIDEO_PATH,
) -> RTIResult:
    """Run minimum-time optimization and save the simulation as an MP4."""
    sim = initialize_hover()
    gate_centers = np.asarray([gate.position for gate in GATES], dtype=np.float64)
    if not len(gate_centers):
        raise ValueError("At least one gate must be defined in drone_sim.GATES")

    # Finish one metre beyond the final gate so the vehicle flies through it.
    exit_point = gate_centers[-1] + np.array([1.0, 0.0, 0.0], dtype=np.float64)
    waypoints = np.vstack((gate_centers, exit_point))
    goal = jnp.asarray(exit_point)
    duration = min(MAX_DURATION, max(2.0, float(np.linalg.norm(exit_point)) / 1.5))
    x = measured_state(duration)
    reference = build_reference(x, waypoints, duration)
    # First controller: perform the expensive nominal initialization once.
    controller = build_controller(
        reference,
        goal,
        initialize_nominal=True,
    )
    gate_index = 0
    simulated_time = 0.0
    output_path, video_writer = open_video_writer(video_path)
    save_video_frame(video_writer)
    print(f"Recording race video to {output_path}")

    executed_times = []
    executed_actions = []

    for iteration in range(max_iterations):
        x = measured_state(duration)
        remaining = waypoints[gate_index:]
        reference = build_reference(x, remaining, duration)

        solve_start = perf_counter()
        _, predicted_x, predicted_u, _, _, _, _ = controller.run(
            x0=x,
            reference=reference,
            parameter=1.0 / HORIZON,
        )
        solve_ms = 1e3 * (perf_counter() - solve_start)

        if iteration == 0:
            # The first controller's initialize_nominal path is executed inside
            # controller.run(). Rebuild once with nominal initialization disabled
            # so all subsequent MPC calls perform only the configured 1 SQP step.
            #
            # Seed the new RTI controller with the trajectory just optimized by
            # the 100-step initialization so we do not throw that solution away.
            controller = build_controller(
                reference,
                goal,
                initialize_nominal=False,
                X_in=predicted_x,
                U_in=predicted_u,
            )

        optimized_duration = float(np.asarray(predicted_x[0, -1]))
        # A single RTI SQP step can be infeasible during initial warm-up.
        # Bound the duration update so one poor iterate cannot create a very
        # large or tiny physics interval.
        duration = float(
            np.clip(
                optimized_duration,
                max(MIN_DURATION, 0.8 * duration),
                min(MAX_DURATION, 1.2 * duration),
            )
        )
        dt = duration / HORIZON

        command = np.clip(
            np.asarray(predicted_u[0]),
            0.0,
            1.0,
        ).astype(np.float64) * MAX_ROTOR_RPM

        # Log the action that is ACTUALLY applied.
        executed_times.append(simulated_time)
        executed_actions.append(command.copy())

        physics_steps = max(1, round(dt * sim.freq))
        state = step(command, n_steps=physics_steps)
        elapsed_time = physics_steps / sim.freq
        frame_count = max(1, round(elapsed_time * VIDEO_FPS))
        save_video_frame(video_writer, repeat=frame_count)
        simulated_time += elapsed_time
        position = np.asarray(state.pos[0, 0])

        previous_position = np.asarray(x[:3], dtype=np.float64)
        if gate_index < len(GATES) and crossed_gate(
            previous_position,
            position,
            GATES[gate_index],
        ):
            print(f"Passed gate {gate_index + 1}")
            gate_index += 1

        goal_error = float(np.linalg.norm(position - exit_point))
        print(
            f"RTI {iteration:03d} | solve {solve_ms:7.2f} ms | "
            f"T {duration:5.2f} s | goal error {goal_error:5.2f} m"
        )

        # Success is now defined purely by physically crossing the final gate.
        if gate_index == len(GATES):
            video_writer.release()
            print(f"Saved race video to {output_path}")
            plot_executed_actions(executed_times, executed_actions)
            return RTIResult(True, iteration + 1, simulated_time, position)

    position = np.asarray(sim.data.states.pos[0, 0])
    video_writer.release()
    print(f"Saved race video to {output_path}")
    plot_executed_actions(executed_times, executed_actions)
    return RTIResult(False, max_iterations, simulated_time, position)

def plot_executed_actions(
    times: list[float],
    actions: list[np.ndarray],
    path: str | Path = "executed_actions.png",
) -> None:
    """Save a plot of the rotor commands actually executed by the simulator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray(times)
    actions = np.asarray(actions)

    if len(actions) == 0:
        return

    path = Path(path)

    plt.figure(figsize=(10, 5))

    for motor in range(NU):
        plt.step(
            times,
            actions[:, motor],
            where="post",
            label=f"Motor {motor + 1}",
        )

    plt.axhline(
        HOVER_RPM,
        linestyle="--",
        label="Hover RPM",
    )

    plt.xlabel("Simulated time [s]")
    plt.ylabel("Rotor command [RPM]")
    plt.title("Executed Control Actions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved executed action plot to {path}")

def main() -> None:
    try:
        result = run_rti_mpc()
        print(result)
    except KeyboardInterrupt:
        print("\nStopping RTI MPC...")
    finally:
        close()


if __name__ == "__main__":
    main()
