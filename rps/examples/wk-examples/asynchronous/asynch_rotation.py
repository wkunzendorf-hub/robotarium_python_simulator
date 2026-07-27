# Creating Robotarium simulation using asynchronous rotation cycle sheaf

import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

# Asynch sheaf submodule package
from asynchsheaves.asynch import initialize_asynchronous_robot_algorithm
import asynchsheaves.synch as syn
from asynchsheaves.sheaf import CellularSheaf

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 4
iterations = 2000

# =========================================================
# GRAPH TOPOLOGY & CONTROLLER SETUP (SHEAF SETUP)
# =========================================================
# Generate a connected cyclic graph Laplacian
nodes = [1, 2, 3, 4]
edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
shf = CellularSheaf(nodes, edges, dim=2)

# Making and setting restriction maps
eye2 = np.eye(2)
rot = np.array([[0, -1], [1, 0]]) # 90 Degree rotation

# Setting restriction maps
# Sheaf rotation restriction maps ([[0, 1], [1, 0]]) will be applied in a counterclockwise manner (4->1, 1->2, 2->3, 3->4)
# Edge (1, 2)
shf.set_res_map(1, 2, rot)
shf.set_res_map(2, 1, eye2)
# Edge (2, 3)
shf.set_res_map(2, 3, rot)
shf.set_res_map(3, 2, eye2)
# Edge (3, 4)
shf.set_res_map(3, 4, rot)
shf.set_res_map(4, 3, eye2)
# Edge (1, 4)
shf.set_res_map(4, 1, eye2)
shf.set_res_map(1, 4, rot)

# Defining initial state values
x1 = [-1.0, 0.5]
x2 = [1.0, 0.5]
x3 = [1.0, -0.5]
x4 = [-1.0, -0.5]
shf.set_node_loc_sect(1, x1)
shf.set_node_loc_sect(2, x2)
shf.set_node_loc_sect(3, x3)
shf.set_node_loc_sect(4, x4)

# Setting phase and comm rate
shf.set_phase_and_comms(1, 2, 3_000)
shf.set_phase_and_comms(2, 4, 6_000)
shf.set_phase_and_comms(3, 8, 9_000)
shf.set_phase_and_comms(4, 16, 12_000)

# Important simulation variables
alpha = 0.3
n = 100
tuning = 1
v_max = 0.2

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_conditions = np.array([[x1[0], x2[0], x3[0], x4[0]], [x1[1], x2[1], x3[1], x4[1]], [0, -np.pi, -np.pi, 0]], dtype=float)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=False, initial_conditions=initial_conditions)

# Initializing cellular sheaf dynamics
asynchronous_robot_algorithm = initialize_asynchronous_robot_algorithm(sheaf=shf, alpha=alpha, n=n, offset=0)

# Get the SI/UNI mapping functions
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for k in range(iterations):
    # Retrieve current poses
    x = r.get_poses()
    xi = uni_to_si_states(x)

    # Initialize SI velocity vector
    dxi = np.zeros((2, N))

    # Asynchronous consensus algorithm
    output_values = asynchronous_robot_algorithm()

    assert xi[:2, :].shape[0] == output_values.shape[0], "Dimension mismatch."
    assert xi[:2, :].shape[1] == output_values.shape[1], "Dimension mismatch."

    dxi = tuning * (output_values - xi)

    # Velocity maximum control
    for robot in range(dxi.shape[1]):
        velocity = dxi[:, robot]
        v_mag = np.linalg.norm(velocity)
        velocity = min(1, v_max / v_mag) * velocity
        dxi[:, robot] = velocity

    # Convert to unicycle velocities
    dxu = si_to_uni_dyn(dxi, x)
    dxu = uni_barrier_cert(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

r.debug()