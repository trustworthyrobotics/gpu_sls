import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import jax
import glfw
import numpy as np

from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim


HERE = Path(__file__).resolve().parent
WORLD_SCENE = HERE / "drone_racing_world.xml"
GATE_MESH = HERE / "georgia_tech_gate" / "georgia_tech_gate_textured.obj"
GATE_TEXTURE = HERE / "georgia_tech_gate" / "gate_design.png"
GATE_MESH_NAME = "georgia_tech_gate"
GATE_TEXTURE_NAME = "georgia_tech_gate_design"
GATE_MATERIAL_NAME = "georgia_tech_gate_material"
GATE_MESH_EULER = (0.0, 0.0, 1.5707963267948966)
DRONE_MODEL = "cf21B_500"
CONTROL_DIM = 4

_sim: Sim | None = None
_render_enabled: bool | None = None


@dataclass(frozen=True)
class Gate:
    """Gate position in metres and Euler angles in radians."""

    position: tuple[float, float, float]
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


# Define the course here. Each pose is (x, y, z), (roll, pitch, yaw).
GATES = [
    Gate(position=(2.0, 0.0, 1.2)),
    # Gate(
    #     position=(4.0, 1.0, 1.25),
    #     euler=(0.0, 0.0, 0.35),
    # ),
    # Gate(
    #     position=(6.0, -1.2, 0.95),
    #     euler=(0.0, 0.0, -0.45),
    # ),
    # Gate(
    #     position=(8.0, 0.5, 1.4),
    #     euler=(0.12, 0.0, 0.20),
    # ),
    # Gate(
    #     position=(10.0, -0.5, 1.05),
    #     euler=(0.0, 0.0, -0.25),
    # ),
]


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(str(value) for value in values)


def can_open_viewer() -> bool:
    """Check GLFW before Crazyflow enters its renderer's abort-prone path."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        available = bool(glfw.init())
    if available:
        glfw.terminate()
    return available


def add_gate(worldbody: ET.Element, gate: Gate, index: int) -> None:
    """Add one instance of the Georgia Tech gate mesh at the requested pose."""
    body = ET.SubElement(
        worldbody,
        "body",
        name=f"gate{index}",
        pos=_numbers(gate.position),
        euler=_numbers(gate.euler),
    )
    ET.SubElement(
        body,
        "geom",
        name=f"gate{index}_georgia_tech_mesh",
        type="mesh",
        mesh=GATE_MESH_NAME,
        euler=_numbers(GATE_MESH_EULER),
        material=GATE_MATERIAL_NAME,
        contype="0",
        conaffinity="0",
    )


def build_scene(gates: list[Gate]) -> ET.ElementTree:
    """Load our world XML and apply the gate poses defined in Python."""
    tree = ET.parse(WORLD_SCENE)
    model = tree.getroot()
    mesh = model.find(f"./asset/mesh[@name='{GATE_MESH_NAME}']")
    if mesh is None:
        raise ValueError(f"{WORLD_SCENE} does not declare {GATE_MESH_NAME}")
    # Crazyflow changes meshdir while attaching its drone. An absolute path
    # keeps this custom asset valid through that model-replication step.
    mesh.set("file", str(GATE_MESH))
    texture = model.find(f"./asset/texture[@name='{GATE_TEXTURE_NAME}']")
    if texture is None:
        raise ValueError(f"{WORLD_SCENE} does not declare {GATE_TEXTURE_NAME}")
    texture.set("file", str(GATE_TEXTURE))

    worldbody = model.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{WORLD_SCENE} has no worldbody")
    for body in list(worldbody.findall("body")):
        if body.get("name", "").startswith("gate"):
            worldbody.remove(body)
    for index, gate in enumerate(gates, start=1):
        add_gate(worldbody, gate, index)

    ET.indent(model)
    return tree


def create_sim(gates: list[Gate]) -> Sim:
    """Create Crazyflow using a scene generated entirely by this script."""
    with tempfile.TemporaryDirectory(prefix="crazyflow_drone_racing_") as temp_dir:
        scene_path = Path(temp_dir) / "scene.xml"
        if not WORLD_SCENE.is_file():
            raise FileNotFoundError(f"World scene not found: {WORLD_SCENE}")
        if not GATE_MESH.is_file():
            raise FileNotFoundError(f"Gate mesh not found: {GATE_MESH}")
        if not GATE_TEXTURE.is_file():
            raise FileNotFoundError(f"Gate texture not found: {GATE_TEXTURE}")
        build_scene(gates).write(scene_path, encoding="utf-8", xml_declaration=True)

        # Sim reads and compiles the XML during construction.
        try:
            jax.devices("gpu")
            device = "gpu"
        except RuntimeError:
            device = "cpu"
            print("No JAX GPU detected; using CPU instead")

        sim = Sim(
            n_worlds=1,
            n_drones=1,
            dynamics=Dynamics.first_principles,
            control=Control.rotor_vel,
            drone=DRONE_MODEL,
            freq=500,
            attitude_freq=500,
            state_freq=100,
            device=device,
            xml_path=scene_path,
        )
        validate_gate_meshes(sim, len(gates))
        return sim


def validate_gate_meshes(sim: Sim, gate_count: int) -> None:
    """Ensure Crazyflow retained the requested mesh after attaching the drone."""
    import mujoco

    mesh_id = mujoco.mj_name2id(
        sim.mj_model,
        mujoco.mjtObj.mjOBJ_MESH,
        GATE_MESH_NAME,
    )
    if mesh_id < 0:
        raise RuntimeError(f"Crazyflow did not load {GATE_MESH}")

    for index in range(1, gate_count + 1):
        geom_name = f"gate{index}_georgia_tech_mesh"
        geom_id = mujoco.mj_name2id(
            sim.mj_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        if geom_id < 0 or sim.mj_model.geom_dataid[geom_id] != mesh_id:
            raise RuntimeError(f"{geom_name} does not reference {GATE_MESH_NAME}")

    print(f"Verified {gate_count} gates use {GATE_MESH}")


def get_sim() -> Sim:
    """Return the shared simulator, creating it on first use."""
    global _sim
    if _sim is None:
        _sim = create_sim(GATES)
        _sim.reset()
    return _sim


def _control_command(command: np.ndarray) -> jax.Array:
    """Normalize motor RPM commands to [world, drone, 4]."""
    sim = get_sim()
    command_array = np.asarray(command, dtype=np.float32)
    expected_shape = (sim.n_worlds, sim.n_drones, CONTROL_DIM)

    if command_array.shape == (CONTROL_DIM,):
        command_array = np.broadcast_to(command_array, expected_shape)
    elif command_array.shape == (sim.n_drones, CONTROL_DIM):
        command_array = np.broadcast_to(command_array, expected_shape)
    elif command_array.shape != expected_shape:
        raise ValueError(
            f"Motor command must have shape ({CONTROL_DIM},), "
            f"({sim.n_drones}, {CONTROL_DIM}), or {expected_shape}; "
            f"received {command_array.shape}"
        )

    return jax.device_put(command_array, sim.device)


def dynamics(command: np.ndarray):
    """Stage four motor RPM commands for the first-principles dynamics."""
    sim = get_sim()
    sim.rotor_vel_control(_control_command(command))
    return sim.data


def step(command: np.ndarray | None = None, n_steps: int | None = None):
    """Advance the simulator and return the updated drone state.

    When supplied, ``command`` is staged before stepping. By default this
    advances one controller interval rather than a single 500 Hz substep.
    """
    sim = get_sim()
    if command is not None:
        dynamics(command)
    if n_steps is None:
        n_steps = sim.freq // sim.control_freq
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive; received {n_steps}")
    sim.step(n_steps)
    return sim.data.states


def render(
    mode: str = "human",
    camera: int | str = "race_camera",
    **kwargs,
):
    """Render the current state, or return ``None`` on a headless machine."""
    global _render_enabled
    sim = get_sim()
    if _render_enabled is None:
        _render_enabled = can_open_viewer() if mode == "human" else True
        if not _render_enabled:
            print("No graphical display detected; render skipped")
    if not _render_enabled:
        return None
    return sim.render(mode=mode, camera=camera, **kwargs)


def close() -> None:
    """Close and discard the shared simulator."""
    global _sim, _render_enabled
    if _sim is not None:
        _sim.close()
    _sim = None
    _render_enabled = None


def main():
    print(f"Building a race scene with {len(GATES)} gates")
    sim = get_sim()

    # Approximate hover command for the cf21B_500 motors, in RPM.
    cmd = np.full(
        (sim.n_worlds, sim.n_drones, CONTROL_DIM),
        16_000.0,
        dtype=np.float32,
    )

    # Keep viewer/simulation alive indefinitely.
    try:
        while True:
            dynamics(cmd)
            step()
            render()

            # Rendering only; physics timing does not need to run flat-out.
            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        print("\nClosing Crazyflow...")

    finally:
        close()


if __name__ == "__main__":
    main()
