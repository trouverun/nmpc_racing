import time
import sys
import os
import config
import pickle
import numpy as np
from datetime import datetime
from utils import make_lap_plot
from driver.sim_wrapper.sim import SimWrapper
from driver.vision_pipeline.vision import VisionPipeline
from driver.mapping.mapping import MappingPipeline
from driver.nonlinear_mpc.controller import Controller
from driver.nonlinear_mpc.dynamics_identification.data_storage import DataStorage


def driver_process(sim_executable_path, map_list, render_queue, graph_queue, exit_event, dynamics_type,
                   mapping_from_scratch, disable_camera, output_data_dir):

    time_now = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    record_lap_data = output_data_dir is not None
    session_output_dir = ''
    if record_lap_data:
        session_output_dir = output_data_dir + time_now + '_%s/' % dynamics_type
        os.makedirs(session_output_dir, exist_ok=True)

    sys.path.append('driver')
    sim_wrapper = SimWrapper(sim_executable_path)
    vision_pipeline = VisionPipeline()
    mapping_pipeline = MappingPipeline()

    controller = Controller(dynamics_type)
    data_storage = DataStorage(config.dynamics_data_folder, time_now + '_%s/' % dynamics_type)

    known_tracks = {}
    if not mapping_from_scratch:
        try:
            with open('tracks.pickle', 'rb') as handle:
                known_tracks = pickle.load(handle)
        except:
            pass
    else:
        try:
            os.remove('tracks.pickle')
        except:
            pass

    while True:
        for map_name in map_list:
            track_output_dir = session_output_dir + map_name
            if record_lap_data:
                os.makedirs(track_output_dir, exist_ok=True)
            controller.reset()

            blue_cones = None
            yellow_cones = None
            if map_name in known_tracks.keys():
                blue_cones, yellow_cones = known_tracks[map_name]
                known_track = True
            else:
                known_track = False

            mapping_pipeline.reset(blue_cones, yellow_cones)
            sim_wrapper.start_sim(map_name, known_track)

            vision_out, mapping_out = None, None
            if known_track:
                vision_out = {
                    'blue_cones': None,
                    'yellow_cones': None,
                    'blue_pixels': None,
                    'yellow_pixels': None,
                }

            steer, throttle, solver_success = 0, 0, True
            mapping_pos, mapping_hdg, mapping_frame = None, None, None
            using_mapping_camera = False

            # Drive until a collision with a cone, while increasing the maximum allowed speed after each lap
            while True:
                should_disable_camera = disable_camera and known_track
                using_mapping_camera = not using_mapping_camera

                t_start_ns = time.time_ns()
                if exit_event.is_set():
                    sim_wrapper.stop_sim(wait=False)
                    return

                sim_out, done = sim_wrapper.step(steer, throttle, solver_success, using_mapping_camera, should_disable_camera)

                if done:
                    break

                if sim_out['lap_switch'] and record_lap_data:
                    filename = '%s/lap_%d_(max_speed_%.2f).png' % (track_output_dir, sim_out['laps_done'], sim_out['max_speed'])
                    track_data = {'blue_cones': mapping_out['blue_cones'], 'yellow_cones': mapping_out['yellow_cones']}
                    make_lap_plot(filename=filename, track_data=track_data, lap_data=sim_out['lap_data_dict'])

                if known_track:
                    data_storage.record_data(sim_out)

                # Save the car state and camera frame, to be used next iteration for updating the map
                if using_mapping_camera:
                    mapping_pos = sim_out['car_pos']
                    mapping_hdg = sim_out['car_hdg']
                    mapping_frame = sim_out['camera_frame']

                # After first lap vision system and map updates are disabled, and we can run higher control Hz:
                if not (sim_out['laps_done'] > 0 or known_track):
                    mpc_dt = config.mpc_first_lap_dt
                else:
                    mpc_dt = config.mpc_fast_lap_dt
                    if map_name not in known_tracks.keys():
                        known_tracks[map_name] = (mapping_out['blue_cones'], mapping_out['yellow_cones'])
                        with open('tracks.pickle', 'wb') as handle:
                            pickle.dump(known_tracks, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    known_track = True

                # If we are not reading the camera this iteration, we use the previous iteration frame for mapping
                run_vision_system = (not using_mapping_camera or mapping_out is None) and not known_track
                vision_start = time.time_ns()
                if run_vision_system:
                    vision_out = vision_pipeline.process_frame(mapping_pos, mapping_hdg, mapping_frame)
                vision_ms = (time.time_ns() - vision_start) / 1e6

                # Find the midpoints of the track, and update the map every other iteration
                mapping_out = mapping_pipeline.update_map(
                    sim_out['car_pos'], sim_out['car_hdg'], vision_out['blue_cones'], vision_out['yellow_cones'],
                    skip=not run_vision_system,
                    mapping_pos=mapping_pos, mapping_hdg=mapping_hdg
                )

                solver_start = time.time_ns()
                steer, throttle, predicted_states, solver_success = controller.get_control(
                    sim_out, mapping_out['midpoints'], mpc_dt
                )
                solve_ms = (time.time_ns() - solver_start) / 1e6
                predicted_extracted = controller.extract_common_info(predicted_states)

                # Data visualization:
                render_data = {
                    'disable_camera': should_disable_camera,
                    'using_mapping_camera': sim_out['using_mapping_camera'],
                    'camera_frame': sim_out['camera_frame'],
                    'car_pos': predicted_extracted['car_pos'][0],
                    'car_hdg': predicted_extracted['car_hdg'][0],
                    'blue_pixels': vision_out['blue_pixels'],
                    'yellow_pixels': vision_out['yellow_pixels'],
                    'blue_cones': mapping_out['blue_cones'],
                    'yellow_cones': mapping_out['yellow_cones'],
                    'midpoints': mapping_out['midpoints'],
                    'path': predicted_extracted['car_pos'],
                    'steer': steer,
                }
                render_queue.put(render_data)

                iter_time_ms = (time.time_ns() - t_start_ns) / 1e6
                graph_data = {
                    'linv': [sim_out['car_linear_vel'][0], sim_out['car_linear_vel'][1]],
                    'lina': [sim_out['car_linear_acc'][0], sim_out['car_linear_acc'][1]],
                    'angular': [sim_out['car_angular_vel'] * 180/np.pi, sim_out['gt_angular_vel'] * 180/np.pi],
                    'control': [steer, throttle],
                    'sim': [sim_out['frame_time_ms']],
                    'vision': [vision_ms],
                    'solver': [solve_ms],
                    'driver': [iter_time_ms, 1e3 * mpc_dt]
                }
                graph_queue.put(graph_data)

            sim_wrapper.stop_sim()
            data_storage.save_dataset(1.5)