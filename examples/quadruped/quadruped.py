# Adapted from https://github.com/iit-DLSLab/mpx/blob/main/mpx/examples/mjx_quad.py

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
import config_go2 as config
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
    """
    Returns a constant disturbance E with shape (T, n, nc),
    where E[t] = alpha * I for all t.
    """
    def disturbance(X_prefix: jnp.ndarray) -> jnp.ndarray:
        T = X_prefix.shape[0]

        diag = jnp.zeros(n, dtype=X_prefix.dtype)
        diag = diag.at[:2].set(alpha)   # first two entries = alpha

        E0 = jnp.diag(diag)              # (n, n)
        return jnp.broadcast_to(E0, (T, n, n))

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
        str(mpx_root / "data" / "go2" / f"scene_mjx.xml")
    )

    data = mujoco.MjData(model)
    sim_frequency = 200.0
    model.opt.timestep = 1 / sim_frequency

    contact_ids = sim_utils.geom_ids(model, config.contact_frame)

    obstacle_center = np.array([1.0, 0.1])
    obstacle_radius = 0.43
    obstacle_visual_radius = 0.15
    obstacle_visual_half_height = 1.0
    command = jnp.array([
        0.1, 0.0, 0.0,
        0.0, 0.0, 0.0,
        config.robot_height,
    ])

    def outside_circle_constraint(x, u, t):
        del u, t
        distance = jnp.linalg.norm(x[:2] - obstacle_center)
        return jnp.array([obstacle_radius - distance])

    E_mag = 0.05

    mpc = mpc_wrapper.MPCWrapper(
        config,
        sls_config=SLSConfig(),
        sqp_config=SQPConfig(),
        admm_config=ADMMConfig(),
        constraints=outside_circle_constraint,
        num_constraints=1,
        disturbance=make_constant_disturbance(config.n, alpha=E_mag * config.dt),
    )
    solve_mpc = _build_solve_fn(mpc)
    reset_mpc = jax.jit(mpc.reset)

    data.qpos = jnp.concatenate([config.p0, config.quat0, config.q0])
    mujoco.mj_forward(model, data)

    foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
    mpc_data = reset_mpc(mpc.make_data(), data.qpos.copy(), data.qvel.copy(), foot)

    warm_contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
    mpc_data, tau = solve_mpc(
        mpc_data,
        data.qpos.copy(),
        data.qvel.copy(),
        foot,
        command,
        warm_contact,
    )
    tau.block_until_ready()
    mpc_data = reset_mpc(mpc_data, data.qpos.copy(), data.qvel.copy(), foot)

    period = int(sim_frequency / config.mpc_frequency)
    print(f"Controller period: {period} steps at {sim_frequency} Hz simulation frequency.")
    counter = 0
    tau = jnp.zeros(config.n_joints)
    q_ref = config.q0.copy()

    def step_controller():
        nonlocal counter, tau, q_ref, mpc_data

        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        
        if counter % period == 0:
            foot = jnp.asarray(sim_utils.geom_positions(data, contact_ids))
           
            contact = jnp.asarray(sim_utils.estimate_contacts(data, contact_ids))
            print(f"Contact: {contact}")
            print(foot)
            print(f"Command: {command}")
            
            start = timer()
            mpc_data, tau = solve_mpc(
                mpc_data,
                qpos,
                qvel,
                foot,
                command,
                contact*0.0,
            )
            tau.block_until_ready()
            stop = timer()

            # tau = jnp.clip(tau, config.min_torque, config.max_torque)
            # The shifted warm start is the next joint target used by the PD stabilizer.
            q_ref = mpc_data.X0[0, 7 : 7 + config.n_joints]
            print(f"MPC time: {1e3 * (stop - start):.2f} ms")

        data.ctrl = np.asarray(tau)
        mujoco.mj_step(model, data)
        counter += 1

    if headless:
        for _ in range(steps):
            step_controller()
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        # This marker is deliberately narrower than the MPC safety radius so
        # the forbidden boundary remains visible around it.
        with viewer.lock():
            obstacle_geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                obstacle_geom,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([
                    obstacle_visual_radius,
                    obstacle_visual_half_height,
                    0.0,
                ]),
                np.array([
                    obstacle_center[0],
                    obstacle_center[1],
                    obstacle_visual_half_height,
                ]),
                np.eye(3).ravel(),
                np.array([0.9, 0.1, 0.1, 0.35], dtype=np.float32),
            )
            viewer.user_scn.ngeom += 1
        viewer.sync()
        while viewer.is_running():
            tic = timer()
            step_controller()
            toc = timer()
            if toc - tic < model.opt.timestep:
                sleep_time = model.opt.timestep - (toc - tic)
                time.sleep(sleep_time)
            viewer.sync()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(
        headless=args.headless,
        steps=args.steps,
    )
