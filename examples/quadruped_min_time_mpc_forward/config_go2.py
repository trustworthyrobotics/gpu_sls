# Adapted from https://github.com/iit-DLSLab/mpx/blob/main/mpx/config/config_go2.py

import jax.numpy as jnp
import jax 
import mpx.utils.models as mpc_dyn_model
import mpx.utils.mpc_utils as mpc_utils
import mpx.utils.objectives as mpc_objectives
import os 
from functools import partial
from pathlib import Path
import mpx
from mujoco import mjx
from mujoco.mjx._src import math


dir_path = os.path.dirname(os.path.realpath(__file__))
mpx_root = Path(mpx.__file__).parent
model_path = str(mpx_root / "data" / "go2" / "go2_mjx.xml")  # Path to the MuJoCo model XML file
# Contact frame names and body names for feet (or calves)
contact_frame = ['FL', 'FR', 'RL', 'RR']
body_name = ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']

# Time and stage parameters
dt = 0.02  # Time step in seconds
N = 25         # Number of stages
mpc_frequency = 100  # Frequency of MPC updates in Hz

# Timer values (make sure the values match your intended configuration)
timer_t =  jnp.array([0.5, 0.0, 0.0, 0.5])  # Timer values for each leg galop jnp.array([0.25, 0.5, 0.75, 0.0]) crawl jnp.array([0.25, 0.75, 0.0, 0.5])
duty_factor = 0.65 #0.65  # Duty factor for the gait
step_freq = 1.35 #1.4   # Step frequency in Hz
step_height = 0.065 # Step height in meters
initial_height = 0.1  # Initial height of the robot's base in meters
robot_height = 0.27  # Height of the robot's base in meters
clearance_speed = 0.2

# Initial positions, orientations, and joint angles
p0 = jnp.array([0, 0, robot_height])  # Initial position of the robot's base
quat0 = jnp.array([1, 0, 0, 0])  # Initial orientation of the robot's base (quaternion)   
q0 = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])  # Initial joint angles
q0_init = jnp.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])

p_legs0 = jnp.array([
    0.192, 0.142, .0,  # Initial position of the front left leg
    0.192, -0.142, .0, # Initial position of the front right leg
   -0.195, 0.142, .0,  # Initial position of the rear left leg
   -0.195, -0.142, .0  # Initial position of the rear right leg
])

# Determine number of joints and contacts from the lists
n_joints = 12  # Number of joints
n_contact = len(contact_frame)  # Number of contact points
n =  13 + 2*n_joints + 6*n_contact  # Number of states (theta1, theta1_dot, theta2, theta2_dot)
m = n_joints  # Number of controls (F)
grf_as_state = True
foot_slice = slice(13 + 2 * n_joints, 13 + 2 * n_joints + 3 * n_contact)
leg_slice = foot_slice
# Reference torques and controls (using n_joints)
u_ref = jnp.zeros(m)  # Reference controls (concatenated torques)

# Cost matrices (diagonal matrices created using jnp.diag)
Qp    = jnp.diag(jnp.array([0, 0, 1e4]))  # Cost matrix for position
Qrot  = jnp.diag(jnp.array([1000, 1000, 0]))  # Cost matrix for rotation
Qq    = jnp.diag(jnp.ones(n_joints)) * 1e-1 # Cost matrix for joint angles
Qdp   = jnp.diag(jnp.array([1, 1, 1])) * 5e3  # Cost matrix for position derivatives
Qomega= jnp.diag(jnp.array([1, 1, 1])) * 1e2  # Cost matrix for angular velocity
Qdq   = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for joint angle derivatives
Qtau  = jnp.diag(jnp.ones(n_joints)) * 1e-1  # Cost matrix for torques
Q_grf = jnp.diag(jnp.ones(3*n_contact)) * 1e-2  # Cost matrix for ground reaction forces

# For the leg contact cost, repeat the unit cost for each contact point.
Qleg = jnp.diag(jnp.tile(jnp.array([1e4,1e4,1e5]),n_contact))

W = {"pos": Qp, "rot": Qrot, "q": Qq, "vel": Qdp, "omega": Qomega, "dq": Qdq, "contact": Qleg, "tau": Qtau, "grf": Q_grf}

use_terrain_estimation = True  # Flag to use terrain estimation

_state_extra = n - (13 + 2 * n_joints + 3 * n_contact)
initial_state = jnp.concatenate(
    [p0, quat0, q0, jnp.zeros(6 + n_joints), p_legs0, jnp.zeros(_state_extra)]
)

cost = partial(mpc_objectives.quadruped_wb_obj, True, n_joints, n_contact, n_contact, N)
hessian_approx = None
reference_generator = partial(
    mpc_utils.reference_generator,
    use_terrain_estimation,
    N,
    dt,
    n_joints,
    n_contact,
    foot0=p_legs0,
    q0=q0,
    clearence_speed=clearance_speed,
)

def quadruped_wb_dynamics(model, mjx_model, contact_id, body_id, n_joints, dtau, x, u, t, parameter):
    """
    Compute the whole-body dynamics of a quadruped robot using forward dynamics and contact forces.

    Args:
        model: The MuJoCo model object.
        mjx_model: The MuJoCo XLA model object for the simulation.
        contact_id (list): List of contact point IDs for each leg. [FL, FR, RL, RR]
        body_id (list): List of body IDs for each leg. [FL, FR, RL, RR]
        n_joints (int): Number of joints in the quadruped. 
        dt (float): Time step for the simulation.
        x (jnp.ndarray): Current state vector [position, orientation, joint positions, velocities].
        u (jnp.ndarray): Control input vector (torques for the joints).
        t (int): Current time step index.
        parameter (jnp.ndarray): Contact parameters for each time step.

    Returns:
        jnp.ndarray: The updated state vector after applying dynamics and contact forces.
    """
    T = x[-1]

    # Actual physical timestep
    dt = dtau * T
    # Create a new data object for the simulation
    mjx_data = mjx.make_data(model)
    # Update the position and velocity in the data object
    mjx_data = mjx_data.replace(qpos=x[:n_joints+7], qvel=x[n_joints+7:2*n_joints+13])

    # Perform forward kinematics and dynamics computations
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

    # Extract the mass matrix and bias forces
    M = mjx_data.qLD
    D = mjx_data.qfrc_bias

    # Get the contact parameters for the current time step
    contact = parameter[t, :4]

    # Create the torque vector, with zeros for the base and control inputs for the joints
    tau = jnp.concatenate([jnp.zeros(6), u])

    # Get the positions of the contact points on the legs
    FL_leg = mjx_data.geom_xpos[contact_id[0]]
    FR_leg = mjx_data.geom_xpos[contact_id[1]]
    RL_leg = mjx_data.geom_xpos[contact_id[2]]
    RR_leg = mjx_data.geom_xpos[contact_id[3]]

    # Compute the Jacobians for each leg
    J_FL, _ = mjx.jac(mjx_model, mjx_data, FL_leg, body_id[0])
    J_FR, _ = mjx.jac(mjx_model, mjx_data, FR_leg, body_id[1])
    J_RL, _ = mjx.jac(mjx_model, mjx_data, RL_leg, body_id[2])
    J_RR, _ = mjx.jac(mjx_model, mjx_data, RR_leg, body_id[3])

    # Concatenate the Jacobians into a single matrix
    J = jnp.concatenate([J_FL, J_FR, J_RL, J_RR], axis=1)
    # Concatenate the positions of the legs into a single vector
    
    if len(contact_id) > 4 :
        current_leg = jnp.asarray([mjx_data.geom_xpos[int(geom_id)] for geom_id  in contact_id]).flatten()
    else:
        current_leg = jnp.concatenate([FL_leg, FR_leg, RL_leg, RR_leg], axis=0)
    alpha = 25
    # Compute the velocity-level constraint violation
    g_dot = J.T @ x[n_joints+7:13+2*n_joints]
    # Compute the stabilization term
    baumgarte_term = -2 * alpha * g_dot

    # Compute the inverse of the mass matrix projected onto the constraint Jacobian
    JT_M_invJ = J.T @ jax.scipy.linalg.cho_solve((M, False), J)
    # Compute the right-hand side of the constraint force equation
    rhs = -J.T @ jax.scipy.linalg.cho_solve((M, False), tau - D) + baumgarte_term
    # Solve for the ground reaction forces
    cho_JT_M_invJ = jax.scipy.linalg.cho_factor(JT_M_invJ)
    grf = jax.scipy.linalg.cho_solve(cho_JT_M_invJ, rhs)
    # Apply the contact forces only to the legs that are in contact
    grf = jnp.concatenate([grf[:3]*contact[0], grf[3:6]*contact[1], grf[6:9]*contact[2], grf[9:12]*contact[3]])
    # Update the velocity using the computed forces
    v = x[n_joints+7:13+2*n_joints] + jax.scipy.linalg.cho_solve((M, False), tau - D + J @ grf) * dt
    # Perform semi-implicit Euler integration to update the position and orientation
    p = x[:3] + v[:3] * dt
    quat = math.quat_integrate(x[3:7], v[3:6], dt)
    q = x[7:7+n_joints] + v[6:6+n_joints] * dt
    # Concatenate the updated state variables into a single vector
    x_next = jnp.concatenate([p, quat, q, v, current_leg, grf, jnp.asarray([T])])

    return x_next

def dynamics(model, mjx_model, contact_id, body_id):
    return partial(
        quadruped_wb_dynamics,
        model,
        mjx_model,
        contact_id,
        body_id,
        n_joints,
        1/ N,
    )
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_learned_contact_model
# dynamics = mpc_dyn_model.quadruped_wb_dynamics_explicit_contact
max_torque = 25
min_torque = -25