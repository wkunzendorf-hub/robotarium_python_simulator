# Unit testing for GatorTRAX source code

import pytest
import numpy as np
from gatortrax_src import initialize_robot_waypoints, _create_waypoint_array, _order_waypoint_array, _assert_point_proximity, _assert_points_within_bounds

# Unit tests

def test_create_waypoint_array():
    """Test that helper function creates overall array correctly for alterations."""
    robot_count = 2
    waypoints = (np.array([[1, 2], [3, 4]], dtype=float), np.array([[5, 6], [7, 8]], dtype=float))

    res = _create_waypoint_array(robot_count, waypoints)
    exp = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)

    assert np.array_equal(res, exp), "Created waypoint array is not as expected."

def test_order_waypoint_array():
    """Test that helper function orders waypoint array correctly."""
    orders = ([2, 1], [1, 2])
    waypoints_set = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)

    res = _order_waypoint_array(orders, waypoints_set)
    exp = np.array([[5, 2], [7, 4], [1, 6], [3, 8]], dtype=float)

    assert np.array_equal(res, exp), "Ordered waypoint array is not as expected."

def test_assert_point_proximity():
    """Test that helper function correctly lets these points through."""
    starting_points = np.array([[9, 10], [11, 12], [13, 14]], dtype=float)
    waypoints1 = np.array([[1, 2], [3, 4]], dtype=float)
    waypoints2 = np.array([[5, 6], [7, 8]], dtype=float)
    waypoints = (waypoints1, waypoints2)
    spacing=0.3

    # Testing that it doesn't activate
    _assert_point_proximity(starting_points, waypoints, spacing)

def test_assert_point_proximity_activates():
    """Test that helper function shows that these points are too close."""
    starting_points = np.array([[9, 10], [11, 12], [13, 14]], dtype=float)
    waypoints1 = np.array([[1, 2], [3, 4]], dtype=float)
    waypoints2 = np.array([[5, 6], [7, 8]], dtype=float)
    waypoints = (waypoints1, waypoints2)
    spacing = 2.0

    # Testing that it does activate
    with pytest.raises(AssertionError):
        _assert_point_proximity(starting_points, waypoints, spacing)

def test_assert_points_within_bounds():
    """Test that helper function lets within-bounds parameters through."""
    starting_points = np.array([[9, 10], [11, 12], [13, 14]], dtype=float)
    waypoints1 = np.array([[1, 2], [3, 4]], dtype=float)
    waypoints2 = np.array([[5, 6], [7, 8]], dtype=float)
    waypoints = (waypoints1, waypoints2)
    width = 20
    height = 24

    _assert_points_within_bounds(starting_points, waypoints, width, height)

def test_assert_points_within_bounds_activate():
    """Test that helper function stops out-of-bounds parameters."""
    starting_points = np.array([[9, 10], [11, 12], [13, 14]], dtype=float)
    waypoints1 = np.array([[1, 2], [3, 4]], dtype=float)
    waypoints2 = np.array([[5, 6], [7, 8]], dtype=float)
    waypoints = (waypoints1, waypoints2)
    width = 20
    height = 20

    with pytest.raises(AssertionError):
        _assert_points_within_bounds(starting_points, waypoints, width, height)

def test_initialize_robot_waypoints():
    """Test for initializing function for creating ordered waypoint array."""
    robot_count = 2
    starting_points = np.array([[9, 10], [11, 12], [13, 14]], dtype=float)
    waypoints1 = np.array([[1, 2], [3, 4]], dtype=float)
    waypoints2 = np.array([[5, 6], [7, 8]], dtype=float)
    order1 = [2, 1]
    order2 = [1, 2]

    create_ordered_waypoints = initialize_robot_waypoints(waypoints1, waypoints2, robot_count=robot_count, starting_points=starting_points, width=20.0, height=24.0)

    res = create_ordered_waypoints(order1, order2)
    exp = np.array([[5, 2], [7, 4], [1, 6], [3, 8], [9, 10], [11, 12]], dtype=float)

    print("Result: \n", res)

    assert np.array_equal(res, exp), "Final ordered waypoint array is not as expected."






