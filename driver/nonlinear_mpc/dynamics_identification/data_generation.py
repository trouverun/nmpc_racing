import numpy as np
from driver.sim_wrapper.sim import SimWrapper
from driver.nonlinear_mpc.dynamics_identification.data_collector import DataCollector
import time


sim_wrapper = SimWrapper("/home/aleksi/Formula-Student-Driverless-Simulator/FSDS.sh")
data_storage = DataCollector("dataset1")

for map_name, experiment in [('Null_a', 'a'), ('Null_a', 's')]:
    throttle_amplitudes = [0]
    if experiment == 's':
        throttle_amplitudes = [0.2, 0.3, 0.4]
    for t_amp in throttle_amplitudes:
        periods = [2, 3, 5]
        for period in periods:
            sim_wrapper.start_sim(map_name, True)
            steer, throttle = 0, 0
            while True:
                sim_out, done = sim_wrapper.step(steer, throttle, True, False, True, True)

                data_storage.record_data(sim_out)

                if sim_out['time_passed_s'] > 20:
                    break
                if experiment == 'a':
                    steer = 0
                    throttle = np.cos(sim_out['time_passed_s'] / period * 2 * np.pi)
                elif experiment == 's':
                    throttle = t_amp/2 + t_amp/2*np.cos(sim_out['time_passed_s'] / (2*period) * 2 * np.pi)
                    steer = np.cos(sim_out['time_passed_s'] / period * 2 * np.pi)

                time.sleep(0.025)

            sim_wrapper.stop_sim()
            data_storage.save_dataset(period)

