# Creating GatorTRAX function and code for waypoint robot experiment

# modules/packages
import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_positions, generate_random_poses, create_at_position

# GatorTRAX source code
from gatortrax_src import initialize_robot_waypoints

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N = 4

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
starting_points = np.array([[-1.25, -1.25, 1.25, 1.25], [-0.75, 0.75, 0.75, -0.75], [0.0, 0.0, -np.pi, -np.pi]], dtype=float)
r = robotarium.Robotarium(number_of_robots=N, 
                          show_figure=True, 
                          sim_in_real_time=False, 
                          skip_initialization=True, 
                          initial_conditions=starting_points)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
uni_barrier_certificate = create_uni_barrier_certificate_with_boundary()
si_position_controller = create_si_position_controller()
si_to_uni = create_si_to_uni_dynamics()

position_error = 0.05
at_position = create_at_position(position_error=position_error)

# =========================================================
# GOAL INITIALIZATION
# =========================================================
waypoint_set_one = np.array([[0.5, -0.5, 0.0, -1.0], [-0.5, -0.5, -0.5, 0.0]], dtype=float)
waypoint_set_two = np.array([[1.0, 1.0, -0.25, 0.5], [0.25, -0.25, 0.5, 0.0]], dtype=float)
waypoint_set_three = np.array([[-0.75, 0.5, 0.0, -0.5], [0.5, 0.5, 0.0, 0.0]], dtype=float)

create_ordered_waypoints = initialize_robot_waypoints(waypoint_set_one, 
                                                      waypoint_set_two, 
                                                      waypoint_set_three,
                                                      robot_count=N, 
                                                      starting_points=starting_points, 
                                                      width=2.5,
                                                      height=1.5,
                                                      spacing=0.3
                                                      )

# =========================================================
# USER INTERFACE
# =========================================================
robot_one_order = [1, 2, 3]
robot_two_order = [1, 2, 3]
robot_three_order = [1, 3, 2]
robot_four_order = [2, 3, 1]

# =========================================================
# SIMULATION LOOP
# =========================================================
waypoints = create_ordered_waypoints(robot_one_order, 
                                     robot_two_order, 
                                     robot_three_order, 
                                     robot_four_order)
v_max = 0.18
w_max = 3.6
waypoint_amount = 4 # Important
final_points = waypoints[-2:, :]
goal_points = waypoints[:2, :]
already_reported = np.zeros(N, dtype=bool)
waypoint_set_splices = {i: np.array([(i-1)*2, (i-1)*2+1]) for i in range(1, waypoint_amount+1)}
robot_set_numbers = {i: 1 for i in range(1, N+1)}

assert v_max < r.MAX_LINEAR_VELOCITY, "V max must be below maximum linear velocity." # 0.2 m/s
assert v_max < 2 * r.WHEEL_RADIUS * r.MAX_WHEEL_VELOCITY / (2 + r.BASE_LENGTH), "V max must not cause wheel velocity to exceed max." # 0.1895 s^-1
assert w_max < r.MAX_ANGULAR_VELOCITY, "Omega max must be below maximum angular velocity." # 3.636 s^-1

x = r.get_poses()
r.step()

while not at_position(x, final_points)[0] or any(v != 4 for v in robot_set_numbers.values()):
    x = r.get_poses()

    dxi = si_position_controller(x[:2, :], goal_points)
    dxu = si_to_uni(dxi, x)

    # Linear, wheel, and angular velocity maximum control
    for robot in range(dxu.shape[1]):
        # linear velocity
        velocity = dxu[:2, robot]
        v_mag = np.linalg.norm(velocity)
        velocity = min(1, v_max / v_mag) * velocity
        dxu[:2, robot] = velocity

        # angular velocity
        angular = dxu[-1, robot]
        w_mag = np.abs(angular)
        angular = min(1, w_max / w_mag) * angular
        dxu[-1, robot] = angular

    dxu = uni_barrier_certificate(dxu, x)

    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Reporting logic
    _, converged = at_position(x, goal_points)
    newly_arrived = converged & ~already_reported
    for i in np.where(newly_arrived)[0]:
        robot_set_number = robot_set_numbers[i+1]
        print(f"Robot {i+1} has reached its goal position for set {robot_set_number}.")
        
        if robot_set_number != waypoint_amount:
            robot_set_number += 1
            robot_set_numbers[i+1] = robot_set_number
            goal_points[:, i] = waypoints[waypoint_set_splices[robot_set_number], i]
        else: 
            already_reported[i] = True

r.debug()





