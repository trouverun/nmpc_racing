import pygame
import config
import numpy as np
from multiprocessing import Queue, Event
from scipy import interpolate
import casadi
import cv2

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
DEBUG = (0, 255, 255)


def render_process(render_queue: Queue, exit_event: Event):
    pygame.init()
    screen = pygame.display.set_mode([2*config.human_cam_resolution, 2*config.human_cam_resolution])
    prev_positions = np.zeros([config.mpc_horizon, 2])
    while True:
        for _ in pygame.event.get():
            pass

        inputs = render_queue.get()

        if exit_event.is_set():
            return

        if not inputs['disable_camera']:
            if inputs['using_mapping_camera']:
                frame = cv2.resize(
                    inputs['camera_frame'], dsize=(config.human_cam_resolution, config.human_cam_resolution),
                    interpolation=cv2.INTER_LANCZOS4
                )
            else:
                frame = inputs['camera_frame']
            camera_view_surf = draw_detections(
                frame, inputs['blue_pixels'], inputs['yellow_pixels'],
                skip_detections=not inputs['using_mapping_camera']
            )
            offset = config.human_cam_resolution
        else:
            camera_view_surf = pygame.Surface((config.human_cam_resolution, config.human_cam_resolution))
            camera_view_surf.fill(BLACK)
            offset = 0
        prev_positions[:-1, :] = prev_positions[1:, :]
        prev_positions[-1, :] = inputs['car_pos'].flatten()
        global_map_surf = draw_global_map(
            inputs['car_pos'], inputs['car_hdg'], inputs['blue_cones'], inputs['yellow_cones'],
            inputs['midpoints'], inputs['path'],  inputs['steer'], prev_positions.copy(),
            inputs['disable_camera']
        )

        screen.blits([
            (camera_view_surf, (0, 0)),
            (global_map_surf, (offset, 0)),
        ])
        pygame.display.flip()


def draw_detections(img, blue_detections, yellow_detections, skip_detections):
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    if not skip_detections:
        for detections, color in [(blue_detections, BLUE), (yellow_detections, YELLOW)]:
            for det in detections:
                det /= config.mapping_cam_resolution / config.human_cam_resolution
                det = det.astype(np.int32)
                pygame.draw.rect(surf, color, [det[0], det[1], det[2] - det[0], det[3] - det[1]], 2)
                for point in det[4:].reshape(7, 2):
                    pygame.draw.circle(surf, RED, [point[0], point[1]], 2)
    return surf


def draw_global_map(car_pos, car_hdg, blue_array, yellow_array, midpoints, path, steer, prev_positions, big=False):
    if big:
        res = config.human_cam_resolution*2
    else:
        res = config.human_cam_resolution
    surf = pygame.Surface((res, res))
    surf.fill(BLACK)

    car_pos = car_pos.flatten()
    if len(blue_array) and len(yellow_array):
        blue_array = blue_array[np.sqrt(np.square(blue_array[:, :2] - car_pos).sum(axis=1)) < 25]
        yellow_array = yellow_array[np.sqrt(np.square(yellow_array[:, :2] - car_pos).sum(axis=1)) < 25]
        if len(midpoints):
            midpoints = midpoints[np.sqrt(np.square(midpoints[:, 1:3] - car_pos).sum(axis=1)) < 25]

        if len(blue_array) and len(yellow_array):
            max_x = max(blue_array[:, 0].max(), yellow_array[:, 0].max(), car_pos[0])
            min_x = min(blue_array[:, 0].min(), yellow_array[:, 0].min(), car_pos[0])
            center_x = (max_x + min_x) / 2
            width = max_x - min_x + 20
            scaler_x = res/2 / (width/2)

            max_y = max(blue_array[:, 1].max(), yellow_array[:, 1].max(), car_pos[1])
            min_y = min(blue_array[:, 1].min(), yellow_array[:, 1].min(), car_pos[1])
            center_y = (max_y + min_y) / 2
            height = max_y - min_y + 20
            scaler_y = res/2 / (height/2)

            maxs = max(scaler_x, scaler_y)
            scaler_x = maxs
            scaler_y = maxs

            blue_array[:, :2] -= [center_x, center_y]
            yellow_array[:, :2] -= [center_x, center_y]
            if len(midpoints):
                midpoints[:, 1:3] -= [center_x, center_y]
            if len(path):
                path[:, :2] -= [center_x, center_y]
            prev_positions[:, :2] -= [center_x, center_y]

            p1x = int(res / 2)
            p1y = int(res / 2)

            for cone_array, color in [(yellow_array, YELLOW), (blue_array, BLUE)]:
                for cone in cone_array:
                    pygame.draw.circle(surf, color, [p1x + int(cone[0] * scaler_x), p1y - int(cone[1] * scaler_y)], 5)

            if len(midpoints):
                for pt in midpoints:
                    pygame.draw.circle(surf, GREEN, [p1x + int(pt[1] * scaler_x), p1y - int(pt[2] * scaler_y)], maxs*config.track_radius)

            # Draw the same splines MPC will use internally:
            if len(midpoints) > 1:
                k = 1
                if len(midpoints) > 2:
                    k = config.spline_deg
                distances = config.b_spline_points
                cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=k)
                cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=k)
                # Fitting casadi bsplines on all zeros returns NaN, fix:
                c_x = interpolate.splev(distances, cx_spline) + 1e-10
                c_y = interpolate.splev(distances, cy_spline) + 1e-10
                c_dx = interpolate.splev(distances, cx_spline, der=1) + 1e-10
                c_dy = interpolate.splev(distances, cy_spline, der=1) + 1e-10

                theta_cx = casadi.interpolant("theta_cx", "linear", [distances], c_x)
                theta_cy = casadi.interpolant("theta_cy", "linear", [distances], c_y)
                theta_cdx = casadi.interpolant("theta_cdx", "linear", [distances], c_dx)
                theta_cdy = casadi.interpolant("theta_cdy", "linear", [distances], c_dy)
                cx = np.asarray(theta_cx(distances))
                cy = np.asarray(theta_cy(distances))
                cdx = np.asarray(theta_cdx(distances))
                cdy = np.asarray(theta_cdy(distances))

                for (x, y, dx, dy) in zip(cx, cy, cdx, cdy):
                    pygame.draw.circle(surf, DEBUG, [p1x + int(x * scaler_x), p1y - int(y * scaler_y)], 2)
                    angle = np.arctan2(dy, dx)
                    mp_x_b = x + 1*np.cos(angle)
                    mp_y_b = y + 1*np.sin(angle)
                    pygame.draw.line(surf, DEBUG,
                                     [p1x + int(x * scaler_x), p1y - int(y * scaler_y)],
                                     [p1x + int(mp_x_b * scaler_x), p1y - int(mp_y_b * scaler_y)],
                                     2)

            if len(path):
                for pt in path:
                    pygame.draw.circle(surf, RED, [p1x + int(pt[0] * scaler_x), p1y - int(pt[1] * scaler_y)], 5)

            for pos in prev_positions:
                pygame.draw.circle(surf, RED, [p1x + int(pos[0] * scaler_x), p1y - int(pos[1] * scaler_y)], 5)

            x_car = p1x + int((car_pos[0]-center_x) * scaler_x)
            y_car = p1y - int((car_pos[1]-center_y) * scaler_y)
            pygame.draw.circle(surf, WHITE, [x_car, y_car], 5)
            p2x = int(x_car + np.cos(car_hdg + steer*config.car_max_steer + np.deg2rad(config.fov/2)) * scaler_x * config.max_valid_cone_distance)
            p2y = int(y_car - np.sin(car_hdg + steer*config.car_max_steer + np.deg2rad(config.fov/2)) * scaler_y * config.max_valid_cone_distance)
            p3x = int(x_car + np.cos(car_hdg + steer*config.car_max_steer - np.deg2rad(config.fov/2)) * scaler_x * config.max_valid_cone_distance)
            p3y = int(y_car - np.sin(car_hdg + steer*config.car_max_steer - np.deg2rad(config.fov/2)) * scaler_y * config.max_valid_cone_distance)
            pygame.draw.line(surf, WHITE, [x_car, y_car], [p2x, p2y], 5)
            pygame.draw.line(surf, WHITE, [x_car, y_car], [p3x, p3y], 5)

    return surf

