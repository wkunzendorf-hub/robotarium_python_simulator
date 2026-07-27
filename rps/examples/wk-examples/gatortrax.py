# Creating GatorTRAX function and code for waypoint robot experiment

# modules/packages
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_positions, generate_random_poses, create_at_position

# Creating the function
def initialize_robot_waypoints(robot_count: int, starting_points: np.ndarray, *waypoints: np.ndarray):

    # Make sure to add a feature so that waypoints FOR EACH ROBOT are decided in define_robot_waypoint_order

    # Assertions
    assert isinstance(robot_count, int) and robot_count > 0, "Robot count is a positive integer."

    assert isinstance(starting_points, np.ndarray) and starting_points.size == 2, "Starting points must be a 2D Numpy array."
    assert starting_points.shape[0] == 3, "Starting points must contain x, y, and theta rows."
    assert starting_points.shape[1] == robot_count, "Starting point size is inconsistent with the number of robots."

    for waypoint in waypoints:
        assert isinstance(waypoint, np.ndarray) and isinstance.size == 2, "Waypoint must be a 2D Numpy array."
        assert waypoint.shape[0] == 2, "Waypoint size must contain x and y rows."
        assert waypoint.shape[1] == robot_count, "Waypoint size is inconsistent with the number of robots."

    # Maybe include assertion for starting points and waypoints to show that they are within bounds of the Robotarium
    # Include assertion that shows that waypoints are not too close

    def define_robot_waypoint_order(order: list[int]):

        # Make assertion for waypoint amount equaling order list
        waypoint_amount = len(waypoints)
        assert set(order) == set(range(1, waypoint_amount+1)), "Order is not in a viable format."

        waypoint_order = [x - 1 for x in order]
        waypoints_arranged = (waypoints[i] for i in waypoint_order)

        waypoints_set = create_waypoint_array(robot_count, starting_points, waypoints_arranged)

        return waypoints_set



def create_waypoint_array(robot_count, starting_points, waypoints):
    # Making waypoint sets
    waypoint_rows = len(waypoints) * 2 + 2
    waypoint_columns = robot_count
    waypoints_set = np.empty((waypoint_rows, waypoint_columns))
    count = 0
    for waypoint in waypoints:
        waypoints_set[count:count+2, :] = waypoint
        count += 2

    waypoints_set[-2:, :] == starting_points[:2, :]

    return waypoints_set




    




# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 6

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
initial_positions = generate_random_poses(N, spacing=0.5)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_positions)