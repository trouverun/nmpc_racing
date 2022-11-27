import os
import signal
import subprocess
import time
import fsds
import config
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import interpolate


class SimWrapper:
    def __init__(self, sim_executable_path):
        self.executable_path = sim_executable_path
        self.client = None
        self.current_map_i = 0
        self.steer = 0
        self.throttle = 0
        self.failed_solves = 0
        self.laps_driven = 0
        self.max_speed = config.car_initial_max_speed
        self.steer = 0
        self.throttle = 0
        self.prev_pos = None
        self.not_moved_steps = 0
        self.known_track = False
        self.start_time = None
        self.max_samples = 50
        self.times = np.zeros(self.max_samples)
        self.ws = np.zeros(self.max_samples)
        self.samples = 0

    def start_sim(self, map_name, known_track=False):
        try:
            self.simulator_launcher_process = subprocess.Popen([self.executable_path, map_name],  shell=False)
            time.sleep(5)
            self.simulator_pid = int(subprocess.check_output(['pidof', "-s", 'Blocks-Linux-Test']))
        except Exception:
            raise Exception("Failed to start sim")

        time.sleep(2.5)

        failed_attempts = 0
        while failed_attempts < config.max_fsds_client_attempts:
            try:
                self.client = fsds.FSDSClient()
                self.client.confirmConnection()
                self.client.enableApiControl(True)
                break
            except Exception:
                failed_attempts += 1
                time.sleep(1)

        if self.client is None:
            self.stop_sim()
            raise Exception("Failed to connect client")

        self._init_state(known_track)

    def _init_state(self, known_track=False):
        self.steer = 0
        self.throttle = 0
        self.failed_solves = 0
        self.max_speed = config.car_initial_max_speed
        self.steer = 0
        self.throttle = 0
        self.prev_pos = None
        self.not_moved_steps = 0
        self.known_track = known_track
        referee_state = self.client.getRefereeState()
        self.laps_driven = len(referee_state.laps)
        self.start_time = None
        self.times = np.zeros(self.max_samples)
        self.ws = np.zeros(self.max_samples)
        self.samples = 0

    def stop_sim(self, wait=True):
        os.kill(self.simulator_pid, signal.SIGKILL)
        time.sleep(2.5)
        self.simulator_launcher_process.kill()
        self.simulator_launcher_process.wait()
        if wait:
            time.sleep(2.5)

    def step(self, steer, throttle, solver_success, using_mapping_camera, disable_camera=False, disable_referee=False):
        if self.failed_solves > config.max_failed_mpc_solves:
            print("FAILED SOLVES ------------------------------------------")
            return None, True

        if not disable_referee:
            referee_state = self.client.getRefereeState()
            # If we have collided with a cone, we are done:
            if referee_state.doo_counter > config.max_collisions:
                print("COLLISION ------------------------------------------")
                return None, True
            # After each lap increase the max speed:
            if len(referee_state.laps) > self.laps_driven:
                print("LAPPED -------------------------------------------")
                self.laps_driven += 1
                self.max_speed *= config.lap_speed_increase

        using_mapping_camera = using_mapping_camera and not (self.laps_driven > 0 or self.known_track)
        iteration_timestamp = time.time_ns()
        if self.start_time is None:
            self.start_time = iteration_timestamp

        car_info = self.client.getCarState()
        rpm = car_info.rpm

        state = self.client.simGetGroundTruthKinematics()
        car_pos = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        quats = np.array([state.orientation.x_val, state.orientation.y_val, state.orientation.z_val, state.orientation.w_val])
        car_hdg = np.asarray(Rotation.from_quat(quats).as_euler("yxz"))[2]

        world_linear_velocity = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val])
        gt_angular_velocity = state.angular_velocity.z_val
        gt_angular_acc = state.angular_acceleration.z_val

        car_angular_velocity = gt_angular_velocity
        car_angular_acceleration = gt_angular_acc

        car_linear_acceleration = np.array([state.linear_acceleration.x_val, state.linear_acceleration.y_val])

        R = np.array([
            [np.cos(-car_hdg), -np.sin(-car_hdg)],
            [np.sin(-car_hdg), np.cos(-car_hdg)]
        ])
        car_linear_velocity = R @ world_linear_velocity
        car_linear_acceleration = R @ car_linear_acceleration

        if not disable_referee:
            if self.prev_pos is not None:
                if np.sqrt(np.square(car_pos[:2] - self.prev_pos).sum()) < 1e-2:
                    self.not_moved_steps += 1
                    if self.not_moved_steps > config.max_stuck_steps:
                        print("STUCK ------------------------------------------")
                        return None, True
                else:
                    self.not_moved_steps = 0
            self.prev_pos = car_pos[:2]

        # Apply controls
        if not solver_success:
            self.failed_solves += 1
        else:
            self.failed_solves = 0
        controls = fsds.CarControls(steering=-steer)
        if throttle >= 0:
            controls.throttle = throttle
        else:
            controls.brake = -throttle
        self.client.setCarControls(controls)
        self.throttle = throttle
        # Read actual steering angle:
        wheel_states = self.client.simGetWheelStates()
        actual_steer = [-wheel_states.fl_steering_angle/config.car_max_steer, -wheel_states.fr_steering_angle/config.car_max_steer]
        self.steer = actual_steer[np.argmax(np.abs(actual_steer))]

        fl_rpm = wheel_states.fl_rpm
        fr_rpm = wheel_states.fr_rpm
        rl_rpm = wheel_states.rl_rpm
        rr_rpm = wheel_states.rr_rpm

        camera_frame = None
        frame_time_ms = 0
        if not disable_camera:
            if not using_mapping_camera:
                cam = 'human_cam'
                res = config.human_cam_resolution
            else:
                cam = 'mapping_cam'
                res = config.mapping_cam_resolution

            # Read camera frame
            camera_t1 = time.time_ns()
            images = self.client.simGetImages(
                [fsds.ImageRequest(camera_name=cam, image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=False)],
                vehicle_name='FSCar'
            )
            camera_frame = fsds.string_to_uint8_array(images[0].image_data_uint8).reshape([res, res, 3])[:, :, ::-1].astype(np.uint8)
            frame_time_ms = (time.time_ns() - camera_t1) / 1e6

        speed = np.sqrt(np.square(car_linear_velocity).sum())
        slip = np.arctan2(car_linear_velocity[1], car_linear_velocity[0])

        outputs = {
            'timestamp': iteration_timestamp, 'car_pos': car_pos, 'car_hdg': car_hdg,
            'car_linear_vel': car_linear_velocity, 'car_angular_vel': car_angular_velocity,
            'car_linear_acc': car_linear_acceleration, 'car_angular_acc': car_angular_acceleration,
            'car_speed': speed, 'car_slip': slip, 'world_linear_velocity': world_linear_velocity,
            'car_steer': self.steer, 'car_steer_cmd': steer, 'car_rpm': rpm, 'car_throttle': self.throttle,
            'camera_frame': camera_frame, 'frame_time_ms': frame_time_ms,
            'disable_camera': disable_camera, 'using_mapping_camera': using_mapping_camera,
            'known_track': self.known_track, 'laps_done': self.laps_driven, 'max_speed': self.max_speed,
            'fl_rpm': fl_rpm, 'fr_rpm': fr_rpm, 'rl_rpm': rl_rpm, 'rr_rpm': rr_rpm,
            'gt_angular_vel': gt_angular_velocity, 'gt_angular_acc': gt_angular_acc,
            'time_passed_s': (iteration_timestamp - self.start_time) / 1e9
        }

        return outputs, False