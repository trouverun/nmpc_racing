import time
import numpy as np
from driver.nonlinear_mpc.solver import Solver


class Controller:
    def __init__(self, model_type):
        self.model_type = model_type
        self.solver = Solver(model_type)

    def reset(self):
        self.solver.reset()

    def extract_common_info(self, state):
        info = {}
        if len(state.shape) == 1:
            if self.model_type == 'kinematic_bicycle':
                info['car_pos'] = state[:2]
                info['car_hdg'] = state[2]
                info['steer'] = state[5]
                info['throttle'] = state[6]
            else:
                info['car_pos'] = state[:2]
                info['car_hdg'] = state[2]
                info['steer'] = state[6]
                info['throttle'] = state[7]
        else:
            if self.model_type == 'kinematic_bicycle':
                info['car_pos'] = state[:, :2]
                info['car_hdg'] = state[:, 2]
                info['steer'] = state[:, 5]
                info['throttle'] = state[:, 6]
            else:
                info['car_pos'] = state[:, :2]
                info['car_hdg'] = state[:, 2]
                info['steer'] = state[:, 6]
                info['throttle'] = state[:, 7]
        return info

    def _get_state(self, sim_out):
        if self.model_type == 'kinematic_bicycle':
            return np.array([
                sim_out['car_pos'][0],
                sim_out['car_pos'][1],
                sim_out['car_hdg'],
                sim_out['car_speed'],
                sim_out['car_slip'],
                sim_out['car_steer'],
                sim_out['car_throttle'],
            ])
        else:
            return np.array([
                sim_out['car_pos'][0],
                sim_out['car_pos'][1],
                sim_out['car_hdg'],
                sim_out['car_linear_vel'][0],
                sim_out['car_linear_vel'][1],
                sim_out['car_angular_vel'],
                sim_out['car_steer'],
                sim_out['car_throttle'],
            ])

    def get_control(self, sim_out, midpoints, mpc_dt, no_delay_comp=False):
        state = self._get_state(sim_out)

        if not no_delay_comp:
            dt_s = (time.time_ns() - sim_out['timestamp']) / 1e9
            state = self.solver.delay_compensation(state, dt_s)

        predicted_states = np.zeros([self.solver.N, self.solver.n_states])

        if len(midpoints) < 2:
            return 0, -1, predicted_states, False

        self.solver.initialize(state, midpoints, sim_out['max_speed'], mpc_dt)

        try:
            steer, throttle, predicted_states = self.solver.solve()
            solver_success = True
        except Exception:
            solver_success = False
            steer = 0
            throttle = -1

        return steer, throttle, predicted_states, solver_success