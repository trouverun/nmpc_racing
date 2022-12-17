import numpy as np
import time
import config
from scipy.spatial.transform import Rotation
from utils import is_point_in_triangle


class MappingPipeline:
    def __init__(self):
        self.blue_cones = []
        self.yellow_cones = []

    def reset(self, blue_cones=None, yellow_cones=None):
        if blue_cones is not None and yellow_cones is not None:
            self.blue_cones = blue_cones
            self.yellow_cones = yellow_cones
        else:
            self.blue_cones = []
            self.yellow_cones = []

    def _update_cones(self, cones, measured_cones, R_car, car_pos, v1, v2, v3, alpha=0.05):
        n_initial_cones = len(cones)
        cone_array = np.asarray(cones)
        observed_cones = []

        # Adjust cone variance based on distance
        cone_distances = np.sqrt(np.square(car_pos[:2] - measured_cones[:, :2]).sum(axis=1))
        dist_variance = np.where(cone_distances > 10, (cone_distances - config.variance_increase_distance) / (config.variance_increase_distance/config.additional_cone_pos_variance), np.zeros_like(cone_distances))
        dist_variance = np.clip(dist_variance, 0, config.additional_cone_pos_variance)
        measurement_variance = dist_variance + config.cone_position_variance

        # For each measured cone, calculated the likelihood that it corresponds to an existing cone in the map.
        # If likelihood is below a threshold, a new cone is added to the map, otherwise the position of the most likely match is adjusted
        for i, measured_cone_pos in enumerate(measured_cones):
            pz = np.zeros(n_initial_cones + 1)
            # Threshold probability for creating a new cone:
            pz[n_initial_cones] = alpha
            if n_initial_cones:
                cone_pos_mean = cone_array[:, :3]
                cone_pos_variance = cone_array[:, 3]
                dist = np.sqrt(np.square(measured_cone_pos - cone_pos_mean).sum(axis=1))
                adjusted_variance = measurement_variance[i] + cone_pos_variance
                # Correspondence likelihood for all existing cones in the map (adapted from FastSLAM):
                pz[:n_initial_cones] = np.power(np.sqrt(np.abs(2*np.pi*adjusted_variance)), -0.5) * np.exp(-0.5*dist*(1/adjusted_variance)*dist)
            pz = pz / pz.sum()
            j = np.argmax(pz)
            if j != n_initial_cones:
                # Matched an existing cone, adjust the position estimate with the measurement and measurement variance:
                K = cone_pos_variance[j] * 1 / measurement_variance[i]
                cones[j][:3] += K * (measured_cone_pos - cone_array[j, :3])
                cones[j][3] *= 1 - K
                observed_cones.append(j)
            else:
                # Create a new cone:
                j = len(cones)
                cones.append(np.zeros(6))
                cones[j][:3] = measured_cone_pos
                cones[j][3] = measurement_variance[i]
            cones[j][4] = cones[j][4] + 1

        # For all existing cones which are inside the field of vision triangle, but weren't matched to any measured cones,
        # decrease the likelihood of a correct estimate, and remove if below a threshold:
        all_previous_cone_i = [*range(0, n_initial_cones)]
        not_observed_previous_cone_i = sorted(np.setdiff1d(all_previous_cone_i, observed_cones).tolist())
        not_observed_previous_cone_i.reverse()
        if len(not_observed_previous_cone_i):
            rotated_cones = (R_car @ (cone_array[:, :3] - car_pos[:3]).T).T
            for cone_i in not_observed_previous_cone_i:
                if np.sqrt(np.square(rotated_cones[cone_i, :2]).sum()) > 2:
                    if is_point_in_triangle(rotated_cones[cone_i, :2], v1, v2, v3):
                        cones[cone_i][5] += 1
                        if cones[cone_i][4] / cones[cone_i][5] < config.delete_threshold:
                            cones.pop(cone_i)

    def _find_centerline(self, blue_array, yellow_array, car_pos, car_hdg):
        matches = []
        path = []
        # Iterate through the shorter cone array, and match each cone with the nearest cone of opposing color.
        # From each match calculate a track midpoint.
        if len(blue_array) <= len(yellow_array):
            # Start matching from the nearest blue cone
            dist = np.sqrt(np.square(blue_array[:, :2] - car_pos[:2]).sum(axis=1))
            blue_array = blue_array[np.argsort(dist)]
            for point in blue_array:
                available_yellow = [*range(0, len(yellow_array))]
                dist = np.sqrt(np.square((yellow_array[:, :2] - point[:2])).sum(axis=1))
                matched_yellow_i = np.argmin(dist)
                # If the distance between cones is large, then they are not even close to being correct correspondences, and we ignore them
                if dist[matched_yellow_i] < config.max_cone_separation:
                    matches.append(
                        np.concatenate([
                            (point + yellow_array[matched_yellow_i])[:2] / 2,
                            point[:2], yellow_array[matched_yellow_i, :2]
                        ], axis=0))
                    # remove matched cone, to avoid matching same cone multiple times
                    available_yellow.pop(matched_yellow_i)
                    yellow_array = yellow_array[available_yellow]
        else:
            # Start matching from the nearest yellow cone
            dist = np.sqrt(np.square(yellow_array[:, :2] - car_pos[:2]).sum(axis=1))
            yellow_array = yellow_array[np.argsort(dist)]
            for point in yellow_array:
                available_blue = [*range(0, len(blue_array))]
                dist = np.sqrt(np.square((blue_array[:, :2] - point[:2])).sum(axis=1))
                matched_blue_i = np.argmin(dist)
                # If the distance between cones is large, then they are not even close to being correct correspondences, and we ignore them
                if dist[matched_blue_i] < config.max_cone_separation:
                    matches.append(
                        np.concatenate([
                            (point + blue_array[matched_blue_i])[:2] / 2,
                            blue_array[matched_blue_i, :2], point[:2]
                        ], axis=0))
                    # remove matched cone, to avoid matching same cone multiple times
                    available_blue.pop(matched_blue_i)
                    blue_array = blue_array[available_blue]

        # Iterate through all midpoints, and find the track path by looking for the nearest midpoint, starting from the car position.
        # Path is constrained so that blue cones are to the left, and yellow cones to the right
        if len(matches):
            matches = np.asarray(matches)
            current = car_pos[:2].copy()
            path_len = 0
            first = True
            while len(matches) and path_len < config.max_path_length:
                # Calculate the angle at which we travel through the midpoint, and based on that create a constraint/weight vector:
                displacement = matches[:, :2] - current
                angles = np.arctan2(displacement[:, 1], displacement[:, 0])
                weights = np.array([-np.sin(angles), np.cos(angles), np.sin(angles), -np.cos(angles)]).T

                if first:
                    # We want to have one midpoint behind the car, it makes the midpoint splines better behaved
                    values = (matches[:, 2:] * -weights).sum(axis=1)
                else:
                    # Find the valid midpoints in front of the car
                    values = (matches[:, 2:] * weights).sum(axis=1)

                dist = np.sqrt(np.square(matches[:, :2] - current).sum(axis=1))
                # Filter out points far away as they would likely form an incorrect path
                valid_indices = np.where(np.all([values > 0, dist < config.max_midpoint_distance], axis=0))[0]

                # If no valid midpoints found, time to exit
                if not len(valid_indices):
                    if first:
                        # If we were searching for midpoints behind the car, just project a point directly behind the car instead
                        angle = car_hdg
                        path.append(np.r_[-1, np.array([car_pos[0] - 1*np.cos(angle), car_pos[1] - 1*np.sin(angle)])])
                        first = False
                        continue
                    break

                order = np.argsort(dist)
                valid_order = np.intersect1d(order, valid_indices)
                best = valid_order[0]

                if first:
                    path.append(np.r_[-np.sqrt(np.square(matches[best, :2] - car_pos[:2]).sum()), matches[best, :2]])
                    first = False
                    continue

                path_len += dist[best]
                current = matches[best, :2]
                path.append(np.r_[path_len, current])
                available_indices = [*range(0, len(matches))]
                available_indices.pop(best)
                matches = matches[available_indices]

        return np.asarray(path)

    def update_map(self, car_pos, car_hdg, located_blue_cones, located_yellow_cones, skip, mapping_pos=None, mapping_hdg=None):
        R_car = Rotation.from_euler("xyz", [0, 0, -(mapping_hdg-np.deg2rad(90))]).as_matrix()
        # Define a field of vision triangle in front of the car:
        v1 = np.array([0, 0])
        v2 = v1 + [
            -np.sin(np.deg2rad(config.mapping_vision_adjuster * config.fov / 2)) * config.max_valid_cone_distance,
            np.cos(np.deg2rad(config.mapping_vision_adjuster * config.fov / 2)) * config.mapping_vision_adjuster * config.max_valid_cone_distance
        ]
        v3 = v1 + [
            -np.sin(-np.deg2rad(config.mapping_vision_adjuster * config.fov / 2)) * config.max_valid_cone_distance,
            np.cos(-np.deg2rad(config.mapping_vision_adjuster * config.fov / 2)) * config.mapping_vision_adjuster * config.max_valid_cone_distance
        ]

        t1 = time.time_ns()
        # Only update the map during the first lap:
        if not skip:
            if len(located_blue_cones):
                self._update_cones(self.blue_cones, located_blue_cones, R_car, mapping_pos, v1, v2, v3)
            if len(located_yellow_cones):
                self._update_cones(self.yellow_cones, located_yellow_cones, R_car, mapping_pos, v1, v2, v3)
        blue_array = np.array([])
        yellow_array = np.array([])
        if len(self.blue_cones):
            blue_array = np.asarray(self.blue_cones)[:, :3]
        if len(self.yellow_cones):
            yellow_array = np.asarray(self.yellow_cones)[:, :3]
        t2 = time.time_ns()
        cone_update_ms = (t2-t1) / 1e6

        t1 = time.time_ns()
        midpoints = np.array([])
        if len(blue_array) and len(yellow_array):
            midpoints = self._find_centerline(blue_array, yellow_array, car_pos, car_hdg)
        t2 = time.time_ns()
        find_midpoints_ms = (t2-t1) / 1e6

        outputs = {
            "blue_cones": blue_array,
            "yellow_cones": yellow_array,
            "midpoints": midpoints,
            "cones_ms": cone_update_ms,
            "midpoints_ms": find_midpoints_ms,
        }

        return outputs