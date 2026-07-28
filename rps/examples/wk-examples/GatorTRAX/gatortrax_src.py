# Source code for GatorTRAX waypoint file

import numpy as np

def initialize_robot_waypoints(*waypoints: np.ndarray, robot_count: int, starting_points: np.ndarray, width: float=2.87, height: float=1.67, spacing: float=0.3):
    """Closure function for determining the order of waypoints for each robot."""
    # Assertions
    assert isinstance(robot_count, int) and robot_count > 0, "Robot count is a positive integer."

    assert isinstance(starting_points, np.ndarray) and starting_points.ndim == 2, "Starting points must be a 2D Numpy array."
    assert starting_points.shape[0] == 3, "Starting points must contain x, y, and theta rows."
    assert starting_points.shape[1] == robot_count, "Starting point size is inconsistent with the number of robots."

    for waypoint in waypoints:
        assert isinstance(waypoint, np.ndarray) and waypoint.ndim == 2, "Waypoint must be a 2D Numpy array."
        assert waypoint.shape[0] == 2, "Waypoint size must contain x and y rows."
        assert waypoint.shape[1] == robot_count, "Waypoint size is inconsistent with the number of robots."

    # Asserting width, height, and spacing
    assert isinstance(width, float) and width > 0, "Width must be a positive, non-zero float."
    assert isinstance(height, float) and height > 0, "Height must be a positive, non-zero float."
    assert isinstance(spacing, float) and spacing > 0, "Spacing must be a positive, non-zero float."

    # Assertion for showing points aren't too close to one another
    _assert_point_proximity(starting_points, waypoints, spacing)
    _assert_points_within_bounds(starting_points, waypoints, width, height)

    # Maybe include assertion for starting points and waypoints to show that they are within bounds of the Robotarium

    # Include assertion that shows that waypoints are not too close

    waypoints_set = _create_waypoint_array(robot_count, waypoints)

    def create_ordered_waypoints(*orders: list[int]):
        """Internal function for determining the order of waypoints for each robot."""
        # Assertions
        assert  robot_count == len(orders), "Number of orders must be equal to the number of robots."
        waypoint_amount = len(waypoints)
        for order in orders:
            assert set(order) == set(range(1, waypoint_amount + 1)), "Order numbers are not in a viable format."

        # Main algorithm
        ordered_waypoints = _order_waypoint_array(orders, waypoints_set)

        ending_points = starting_points[:2, :]
        all_waypoints = np.vstack((ordered_waypoints, ending_points))

        return all_waypoints

    return create_ordered_waypoints

# Helper Functions

def _create_waypoint_array(robot_count, waypoints):
    """Helper function for creating waypoints set; does not include starting points at the end."""

    waypoint_rows = len(waypoints) * 2
    waypoint_columns = robot_count
    waypoints_set = np.empty((waypoint_rows, waypoint_columns))
    count = 0
    for waypoint in waypoints:
        waypoints_set[count:count+2, :] = waypoint
        count += 2

    return waypoints_set

def _order_waypoint_array(orders, waypoints_set):
    """Helper function for creating ordered set of waypoints for each robot."""
    for index, order in enumerate(orders):
        robot_points = waypoints_set[:, index]
        waypoint_order_ranges = [range(2*(x - 1), 2*(x-1)+2) for x in order]
        waypoint_order = [i for x in waypoint_order_ranges for i in x]
        ordered_points = robot_points[waypoint_order]
        waypoints_set[:, index] = ordered_points

    return waypoints_set

def _assert_point_proximity(starting_points, waypoints, spacing):
    """Helper function for making sure the starting points and waypoints aren't too close together."""
    list_of_points = starting_points[:2, :]
    for waypoint in waypoints:
        list_of_points = np.hstack((list_of_points, waypoint))

    mask = np.ones(list_of_points.shape[1], dtype=np.bool)

    for index in range(list_of_points.shape[1]):
        point = list_of_points[:, index]
        mask[index] = False
        other_points = list_of_points[:, mask]

        for column in range(other_points.shape[1]):
            assert np.linalg.norm(point - other_points[:, column]) >= spacing, "A point is too close to another point."

def _assert_points_within_bounds(starting_points, waypoints, width, height):
    list_of_points = starting_points[:2, :]
    for waypoint in waypoints:
        list_of_points = np.hstack((list_of_points, waypoint))

    lower_bounds = np.array([-width/2, -height/2])
    upper_bounds = np.array([width/2, height/2])

    for index in range(list_of_points.shape[1]):
        point = list_of_points[:, index]
        within_bounds = (point >= lower_bounds) & (point <= upper_bounds)

        assert np.all(within_bounds), "Starting point or waypoint is not within bounds."
