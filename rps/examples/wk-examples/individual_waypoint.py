# Creating experiment where each robot has an individual waypoint

# modules/packages
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_positions, generate_random_poses, create_at_position

# Activation: py rps\examples\wk-examples\individual_waypoint.py

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 6

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_positions = generate_random_poses(N, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_positions)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
si_barrier_certificate = create_si_barrier_certificate_with_boundary()
si_position_controller = create_si_position_controller()
si_to_uni = create_si_to_uni_dynamics()

position_error = 0.05
at_position = create_at_position(position_error=position_error)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
arena_width = r.BOUNDARIES[1] - r.BOUNDARIES[0] - 3*r.ROBOT_DIAMETER
arena_height = r.BOUNDARIES[3] - r.BOUNDARIES[2] - 3*r.ROBOT_DIAMETER

# Each waypoint position does not block other robot
waypoints_initialization = generate_random_positions(N=18, width=arena_width, height=arena_height, spacing=0.3) 
waypoint_set_one = waypoints_initialization[:, 0:N]
waypoint_set_two = waypoints_initialization[:, N:2*N]
waypoint_set_three = waypoints_initialization[:, 2*N:3*N]

waypoints = np.vstack((waypoint_set_one, waypoint_set_two, waypoint_set_three))
final_points = waypoint_set_three

assert waypoints.ndim == 2, "Waypoints dimensions not as expected."
assert waypoints.shape[0] == 6, "Waypoints rows not as expected."
assert waypoints.shape[1] == 6, "Waypoint columns not as expected."

assert final_points.shape[0] == 2, "Final rows not as expected."
assert final_points.shape[1] == 6, "Final columns not as expected."

x = r.get_poses()
r.step()

# =========================================================
# MAIN LOOP
# =========================================================
goal_points = waypoints[[0, 1], :]
already_reported = np.zeros(N, dtype=bool)
waypoint_set_splices = {1: np.array([0, 1]),
                        2: np.array([2, 3]),
                        3: np.array([4, 5])
                        } # Splices for waypoint sets
robot_set_numbers = {i: 1 for i in [1, 2, 3, 4, 5, 6]}

# at_position returns (all_done, per_robot_done_array)
while not at_position(x, final_points)[0]:
    x = r.get_poses()

    dxi = si_position_controller(x[:2, :], goal_points)
    dxi = si_barrier_certificate(dxi, x)
    dxu = si_to_uni(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Reporting logic
    _, converged = at_position(x, goal_points)
    newly_arrived = converged & ~already_reported
    for i in np.where(newly_arrived)[0]:
        robot_set_number = robot_set_numbers[i+1]
        print(f"Robot {i+1} has reached its goal position for set {robot_set_number}.")
        
        if robot_set_number != 3:
            robot_set_number += 1
            robot_set_numbers[i+1] = robot_set_number
            goal_points[:, i] = waypoints[waypoint_set_splices[robot_set_number], i]
        else: 
            already_reported[i] = True







