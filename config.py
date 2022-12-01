import numpy as np
from scipy.spatial.transform import Rotation
from utils import TransInv, RpToTrans


dynamics_data_folder = "/home/aleksi/PycharmProjects/nmpc_racing/car_data/"

# Simulator
max_fsds_client_attempts = 10
max_failed_mpc_solves = 10
max_stuck_steps = 10
lap_speed_increase = 1.25
controller_no_cones_attempts = 5
max_collisions = 5


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
variance_increase_distance = 10
additional_cone_pos_variance = 0.1
mapping_vision_adjuster = 0.85
delete_threshold = 1
max_cone_separation = 6.2
max_midpoint_distance = 10
max_path_length = 25


# Car dynamics:
car_max_speed = 15
car_max_acceleration = 11
car_drag = 0.03
car_rolling_resistance = 0.015
car_lr = 0.78
car_lf = 0.41
car_max_steer = np.deg2rad(25)
u_steer_max = 1.5
u_throttle_max = 1.5
car_max_slip = np.deg2rad(180)
u_theta_max = car_max_speed

blend_min_speed = 3
blend_max_speed = 5

# Identified params
car_mass = 190
car_inertia = 227.11926270
wheel_Bf = -13.20129013
wheel_Cf = -1.48318338
wheel_Df = -1180.37243652
wheel_Br = -5.21678352
wheel_Cr = 6.60196829
wheel_Dr = 674.78057861
car_Tm = 2334.06518555
car_Tr0 = 40.22737122
car_Tr2 = 3.24289775

# prevs:
# car_inertia = 222.60449219
# wheel_Bf = -12.82018471
# wheel_Cf = -1.46732950
# wheel_Df = -1185.33837891
# wheel_Br = -4.38556051
# wheel_Cr = 7.58791828
# wheel_Dr = 681.89630127
# car_Tm = 2222.82739258
# car_Tr0 = 38.54015732
# car_Tr2 = 3.26377106


# MPC
spline_deg = 1
track_radius = 1.25
n_bspline_points = 30
bspline_point_distance = 1
bspline_max_distance = n_bspline_points*bspline_point_distance
b_spline_points = np.arange(0, bspline_max_distance, bspline_point_distance)
assert len(b_spline_points) == n_bspline_points
mpc_horizon = 30
mpc_first_lap_hz = 15
mpc_first_lap_dt = 1/mpc_first_lap_hz
mpc_fast_lap_hz = 15
mpc_fast_lap_dt = 1/mpc_fast_lap_hz
car_initial_max_speed = 4.5
# Weights for nonlinear lsq cost:
lag_weight = 100
contour_weight = 25
theta_weight = 0.25
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