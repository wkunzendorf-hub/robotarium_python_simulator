# Creating a Robotarium experiment using an asynchronous sheaf with long communication delays

import numpy as np
import rps.robotarium as robotarium
from rps.utilities.graph import cycleGL, topological_neighbors
from rps.utilities.transformations import create_si_to_uni_mapping

# Asynch sheaf submodule package
from asynchsheaves.asynch import initialize_asynchronous_robot_algorithm
import asynchsheaves.synch as syn
from asynchsheaves.sheaf import CellularSheaf

# Activation: py rps\examples\wk-examples\asynch_linear.py

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 4
iterations = 3000

# =========================================================
# GRAPH TOPOLOGY & CONTROLLER SETUP (SHEAF SETUP)
# =========================================================
# Generate a connected cyclic graph Laplacian
nodes = [1, 2, 3, 4]
edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
shf = CellularSheaf(nodes, edges, dim=2)

assert len(nodes) == N, "Sheaf node amount does not equal expected robot amount."

# Restriction maps
res_map = np.eye(2)
shf.set_all_res_maps(res_map)

# Initial state values
rng = np.random.default_rng()
func = rng.uniform(-1.6, 1.6, size=2)
x1 = [-1.0, 1]
x2 = [1, 1]
x3 = [1, -1]
x4 = [-1, -1]
shf.set_node_loc_sect(1, x1)
shf.set_node_loc_sect(2, x2)
shf.set_node_loc_sect(3, x3)
shf.set_node_loc_sect(4, x4)

# Phase and communication rate values
shf.set_phase_and_comms(1, 2, 1000)
shf.set_phase_and_comms(2, 2, 1200)
shf.set_phase_and_comms(3, 2, 1400)
shf.set_phase_and_comms(4, 2, 1600)

# Important simulation variables
alpha = 0.3
n = 2

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_conditions = np.array([[x1[0], x2[0], x3[0], x4[0]], [x1[1], x2[1], x3[1], x4[1]], [0, 0, 0, 0]], dtype=float)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=False, initial_conditions=initial_conditions)

# Initializing cellular sheaf dynamics
asynchronous_robot_algorithm = initialize_asynchronous_robot_algorithm(sheaf=shf, alpha=alpha, n=n)

# Get the SI/UNI mapping functions
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

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

    dxi = output_values - xi

    # Convert to unicycle velocities
    dxu = si_to_uni_dyn(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

r.debug()


