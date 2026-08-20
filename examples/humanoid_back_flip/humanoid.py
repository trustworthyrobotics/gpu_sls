import argparse
import os
import sys
import time
from timeit import default_timer as timer
from typing import Callable

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, "..")))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from pathlib import Path
import mpx
import config_h1 as config
import mpx.utils.sim as sim_utils

import gpu_sls.legged_mpc as mpc_wrapper
from gpu_sls.gpu_admm import ADMMConfig
from gpu_sls.gpu_sls import SLSConfig
from gpu_sls.gpu_sqp import SQPConfig

jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def make_constant_disturbance(
    n: int,
    alpha: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a horizon-length disturbance acting on planar base position."""

    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        horizon = X_prefix.shape[0]
        diagonal = jnp.zeros(n, dtype=X_prefix.dtype).at[:2].set(alpha)
        return jnp.broadcast_to(jnp.diag(diagonal), (horizon, n, n))

    return disturbance


def _build_solve_fn(mpc):
    @jax.jit
    def solve_mpc(mpc_data, qpos, qvel, foot, command, contact):
        x0 = (
            mpc.initial_state
            .at[mpc.qpos_slice].set(qpos)
            .at[mpc.qvel_slice].set(qvel)
            .at[mpc.foot_slice].set(foot)
        )
        return mpc.run(mpc_data, x0, command, contact)

    return solve_mpc


def main(headless=False, steps=500):
    mpx_root = Path(mpx.__file__).parent
    model = mujoco.MjModel.from_xml_path(
        str(mpx_root / "data" / "unitree_h1" / "mjx_scene_h1_walk.xml")
    )
    data = mujoco.MjData(model)
    sim_frequency = 500.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    # Obstacles: each row is [x, y, radius].
    obstacles = jnp.array(
        [[2.0, 0.3, 0.43], [2.0, -0.9, 0.43], [2.7, 1.5, 0.43]]
    )

    # Cylinder dimensions are visual only.
    OBSTACLE_HEIGHT = 2  # meters
    obstacle_visual_radius = 0.1
    obstacle_visual_half_height = OBSTACLE_HEIGHT / 2.0

    def outside_circle_constraint(x, u, t):
        del u, t
        distances = jnp.linalg.norm(x[:2] - obstacles[:, :2], axis=1)
        return obstacles[:, 2] - distances

    disturbance_magnitude = 0.05
    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=SLSConfig(),
        sqp_config=SQPConfig(),
        admm_config=ADMMConfig(max_iterations=200,
                               initial_rho=30.0),
        constraints=outside_circle_constraint,
        num_constraints=obstacles.shape[0],
        disturbance=make_constant_disturbance(
            config.n,
            alpha=disturbance_magnitude * config.dt,
        ),
    )
    forward_velocity = 0.3
    command_handle = sim_utils.KeyboardVelocityCommand(vx=forward_velocity)

    def make_command():
        command = jnp.asarray(command_handle.mpc_input(config.robot_height))
        return command.at[0].set(forward_velocity)

    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    # Warm up the jitted MPC call so the printed timings are steady-state.
    warm_command = make_command()
    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        warm_command,
        warm_contact,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)

    period = int(sim_frequency / config.mpc_frequency)
    counter = 0
    tau = jnp.zeros(config.n_joints)
    steps_mpc = 0
    total = 0

    def step_controller():
        nonlocal counter, tau, mpc_data, total, steps_mpc

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
            command = make_command()
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))

            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact,
            )
            tau.block_until_ready()
            stop = timer()

            tau = jnp.clip(tau, config.min_torque, config.max_torque)
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau - 3.0 * qvel[6 : 6 + config.n_joints])
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=command_handle.key_callback,
    ) as viewer:
        with viewer.lock():
            for obstacle in np.asarray(obstacles):
                obstacle_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    obstacle_geom,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array(
                        [obstacle_visual_radius, obstacle_visual_half_height, 0.0]
                    ),
                    np.array(
                        [obstacle[0], obstacle[1], obstacle_visual_half_height]
                    ),
                    np.eye(3).ravel(),
                    np.array([0.9, 0.1, 0.1, 0.35], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1
        viewer.sync()
        while viewer.is_running():
            tic = timer()
            overlay_text = command_handle.consume_overlay_text()
            if overlay_text is not None:
                viewer.set_texts((None, None, *overlay_text))
            step_controller()
            elapsed = timer() - tic
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless, steps=args.steps)