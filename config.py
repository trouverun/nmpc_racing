import numpy as np
from scipy.spatial.transform import Rotation
from utils import TransInv, RpToTrans


# Simulator
max_fsds_client_attempts = 10
max_failed_mpc_solves = 10
max_stuck_steps = 100
lap_speed_increase = 1.25
controller_no_cones_attempts = 5
max_collisions = 0


# Camera params
camera_R = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
camera_p = np.array([0, -0.8, -0.3])
camera_T = TransInv(RpToTrans(camera_R, camera_p))
mapping_cam_resolution = 1280
human_cam_resolution = 360
fov = 90
f = mapping_cam_resolution / (2 * np.tan(fov * np.pi / 360))
c = mapping_cam_resolution / 2
K = np.array([
    [f, 0, c],
    [0, f, c],
    [0, 0, 1]
])


# Cone detection / keypoint regression
kp_resize_w, kp_resize_h = 80, 80
kp_min_w, kp_min_h = 20, 20
cone_3d_points = np.array([
    [0, 0, 0.305],
    [-0.042, 0, 0.18],
    [-0.058, 0, 0.1275],
    [-0.075, 0, 0.03],
    [0.042, 0, 0.18],
    [0.058, 0, 0.1275],
    [0.075, 0, 0.03]
])
min_valid_cone_distance = 2
max_valid_cone_distance = 18
min_bbox_conf = 0.85


# Mapping
cone_position_variance = 1
mapping_vision_adjuster = 0.85
delete_threshold = 1
max_cone_separation = 6.2
max_midpoint_distance = 10
max_path_length = 25


# Car dynamics:
car_max_speed = 17.5
car_max_acceleration = 12.5
car_drag = 0.025
car_rolling_resistance = 0.025
car_lr = 0.78
car_lf = 0.41
car_max_steer = np.deg2rad(25)
u_steer_max = 1.5
u_throttle_max = 1.5
car_max_slip = np.deg2rad(9)
u_theta_max = car_max_speed

blend_min_speed = 3
blend_max_speed = 5

# Identified params
car_mass = 190
car_inertia = 231.79156494140625
wheel_Bf = -17.375244140625
wheel_Cf = 0.8026883602142334
wheel_Df = 1545.7764892578125
wheel_Br = -5.899516582489014
wheel_Cr = 5.130153656005859
wheel_Dr = 786.07373046875
car_Tm = 2467.453369140625
car_Tr0 = 44.012840270996094
car_Tr2 = 3.43312931060791


# MPC
spline_deg = 2

track_radius = 1.25
n_bspline_points = 30
bspline_point_distance = 1
bspline_max_distance = n_bspline_points*bspline_point_distance
b_spline_points = np.arange(0, bspline_max_distance, bspline_point_distance)
assert len(b_spline_points) == n_bspline_points
mpc_horizon = 30
mpc_first_lap_hz = 10
mpc_first_lap_dt = 1/mpc_first_lap_hz
mpc_fast_lap_hz = 15
mpc_fast_lap_dt = 1/mpc_fast_lap_hz
car_initial_max_speed = 4.5
# Weights for nonlinear lsq cost:
lag_weight = 100
contour_weight = 25
theta_weight = 0.1
u_steer_weight = 2
u_throttle_weight = 2
steer_weight = 0.1
throttle_weight = 0.1
# Soft constraint violation weights:
soft_u_theta_weight = 5
soft_state_v_weight = 5
soft_state_slip_weight = 5
soft_state_theta_weight = 5
soft_nl_track_circle_weight = 100
soft_nl_max_v_weight = 10
# solver params
solver_max_iter = 10
solver_tolerance = 1e-3


# PETS
pets_track_width = 1
pets_horizon = 20
pets_dt = 0.1
pets_hidden_size = 64
pets_n_hidden = 2
pets_discounting_factor = 0.95
pets_tightening_steps = 0
pets_path_dev_cost = 10
pets_path_progress_reward = 10
pets_track_exit_cost = 5
pets_u_steer_cost = 0.1
pets_u_throttle_cost = 0.1
pets_max_speed_violation_cost = 5
pets_u_max = 5