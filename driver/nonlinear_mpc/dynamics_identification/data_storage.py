import time

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
import os


class DataStorage:
    def __init__(self, data_dir, dataset_name):
        self.initialized = False
        self.current_data = []
        self.dataset_dir = "%s/%s" % (data_dir, dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        try:
            self.original_start_indices = np.load("%s/original_start_indices.npy" % self.dataset_dir).tolist()
            self.original_odometry_data = np.load("%s/original_odometry_data.npy" % self.dataset_dir).tolist()
            self.filtered_start_indices = np.load("%s/filtered_start_indices.npy" % self.dataset_dir).tolist()
            self.filtered_odometry_data = np.load("%s/filtered_odometry_data.npy" % self.dataset_dir).tolist()
        except:
            self.original_start_indices = []
            self.filtered_start_indices = []
            self.original_odometry_data = []
            self.filtered_odometry_data = []

    def _get_data_array(self, sim_out):
        return np.array([
            sim_out['timestamp'],                  # 0
            sim_out['car_pos'][0],                 # 1
            sim_out['car_pos'][1],                 # 2
            sim_out['car_hdg'],                    # 3
            sim_out['car_linear_vel'][0],          # 4
            sim_out['car_linear_vel'][1],          # 5
            sim_out['car_angular_vel'],            # 6
            sim_out['car_linear_acc'][0],          # 7
            sim_out['car_linear_acc'][1],          # 8
            sim_out['car_angular_acc'],            # 9
            sim_out['car_steer'],                  # 10
            sim_out['car_steer_cmd'],              # 11
            sim_out['car_rpm'],                    # 12
            sim_out['car_throttle'],               # 13
            sim_out['fl_rpm'],                     # 14
            sim_out['fr_rpm'],                     # 15
            sim_out['rl_rpm'],                     # 16
            sim_out['rr_rpm'],                     # 17
            sim_out['world_linear_velocity'][0],   # 18
            sim_out['world_linear_velocity'][1],   # 19
            sim_out['car_speed'],                  # 20
            sim_out['car_slip'],                   # 21
            sim_out['car_steer_cmd'],              # 22
        ])

    def record_data(self, sim_out):
        self.current_data.append(self._get_data_array(sim_out))
        if not self.initialized:
            self.original_start_indices.append(len(self.original_odometry_data))
            self.filtered_start_indices.append(len(self.filtered_odometry_data))
            self.initialized = True

    def save_dataset(self, p=None):
        self.initialized = False

        if len(self.current_data) > 500:
            heading_i = 3
            angular_vel_i = 6
            angular_acc_i = 9
            steer_i = 11
            steer_d_i = 22

            odometry_array = np.asarray(self.current_data)[1:-1]
            # shift to start from zero, ns to s:
            timestamps = (odometry_array[:, 0] - odometry_array[0, 0]) / 1e9
            # resample with 2x sample rate:
            new_timestamps, dt = np.linspace(timestamps[0], timestamps[-2], num=2*(len(timestamps)-1), retstep=True)

            if p is not None:
                window_size = int(p / dt)
            else:
                window_size = 75

            original_result = np.c_[new_timestamps[:-1]]
            filtered_result = np.c_[new_timestamps[:-1]]
            for j in range(1, odometry_array.shape[1]):
                if j in [angular_acc_i, steer_d_i]:
                    if j == angular_acc_i:
                        sel_i = angular_vel_i
                    elif j == steer_d_i:
                        sel_i = steer_i
                    f = interpolate.interp1d(timestamps, odometry_array[:, sel_i])
                    resampled = f(new_timestamps)
                    filtered = savgol_filter(
                        resampled, window_length=window_size, polyorder=3, deriv=1
                    )[:-1] / dt
                else:
                    f = interpolate.interp1d(timestamps, odometry_array[:, j])
                    resampled = f(new_timestamps)
                    if j != heading_i:
                        filtered = savgol_filter(resampled, window_length=window_size, polyorder=4, deriv=0)[:-1]
                    else:
                        filtered = resampled[:-1]
                original_result = np.c_[original_result, resampled[:-1]]
                filtered_result = np.c_[filtered_result, filtered]

            invalid = int(np.ceil(window_size/2))
            original_result[invalid:-invalid, 0] -= original_result[invalid, 0]
            filtered_result[invalid:-invalid, 0] -= filtered_result[invalid, 0]
            self.original_odometry_data.extend(original_result[invalid:-invalid].tolist())
            self.filtered_odometry_data.extend(filtered_result[invalid:-invalid].tolist())
            self.current_data = []
            np.save("%s/original_start_indices" % self.dataset_dir, np.asarray(self.original_start_indices))
            np.save("%s/original_odometry_data" % self.dataset_dir, np.asarray(self.original_odometry_data))
            np.save("%s/filtered_start_indices" % self.dataset_dir, np.asarray(self.filtered_start_indices))
            np.save("%s/filtered_odometry_data" % self.dataset_dir, np.asarray(self.filtered_odometry_data))

