# Creating a waypoint example for Robotarium

# modules/packages
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_positions, generate_random_poses, create_at_position

# Activation: py rps\examples\wk-examples\way_point.py

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

waypoints_set_one = generate_random_positions(N, width=arena_width, height=arena_height, spacing=0.5)
waypoints_set_two = generate_random_positions(N, width=arena_width, height=arena_height, spacing=0.5)
waypoints_set_three = generate_random_positions(N, width=arena_width, height=arena_height, spacing=0.5)

waypoints = np.vstack((waypoints_set_one, waypoints_set_two, waypoints_set_three))
final_points = waypoints[4:6, :]

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
waypoint_set_number = 1
waypoint_set_splices = {1: np.array([0, 1]),
                        2: np.array([2, 3]),
                        3: np.array([4, 5])
                        } # Splices for waypoint sets
already_reported = np.zeros(N, dtype=bool)

# at_position returns (all_done, per_robot_done_array)
while not at_position(x, final_points)[0]:
    x = r.get_poses()

    # Getting waypoints
    waypoint_rows = waypoint_set_splices[waypoint_set_number]
    goal_points = waypoints[waypoint_rows, :]

    dxi = si_position_controller(x[:2, :], goal_points)
    dxi = si_barrier_certificate(dxi, x)
    dxu = si_to_uni(dxi, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Reporting logic
    _, converged = at_position(x, goal_points)
    newly_arrived = converged & ~already_reported
    for i in np.where(newly_arrived)[0]:
        print(f"Robot {i+1} has reached its goal position for set {waypoint_set_number}.")
    already_reported |= converged

    if waypoint_set_number != 3 and np.all(already_reported) == True:
        print(f"Waypoint set {waypoint_set_number} has been completed.")
        waypoint_set_number += 1
        already_reported = np.zeros(N, dtype=bool)

r.debug()


