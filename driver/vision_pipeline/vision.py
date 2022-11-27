import sys
import numpy as np
import torch
import time
import cv2
import torch.backends.cudnn as cudnn
import config
from driver.vision_pipeline.keypoints.kp_model import KeypointModel
from driver.vision_pipeline.yolov7.models.yolo import Model
from driver.vision_pipeline.yolov7.utils.general import non_max_suppression
from torchvision.transforms.functional import resize
from scipy.spatial.transform import Rotation
from utils import TransInv, RpToTrans


class VisionPipeline:
    def __init__(self):
        sys.path.append('driver/vision_pipeline/yolov7')
        self.yolo_model = Model(cfg='driver/vision_pipeline/yolov7/yolov7-tiny.yaml', nc=10).to('cuda:0')
        self.yolo_model.load_state_dict(torch.load('driver/vision_pipeline/yolov7/yolo_model.pt'))
        self.yolo_model.eval()
        self.yolo_model.half()
        sys.path.append('driver/vision_pipeline/keypoints/')
        self.kp_model = KeypointModel().to('cuda:0')
        self.kp_model.load_state_dict(torch.load('driver/vision_pipeline/keypoints/kp_weights.pt'))
        self.kp_model.eval()
        self.kp_model.half()
        cudnn.benchmark = False

    def process_frame(self, car_pos, car_hdg, img):
        # Find bounding boxes for all cones in frame
        with torch.no_grad():
            torch_image = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).half().to('cuda:0') / 255.0
            t1 = time.time_ns()
            bbox_pred = self.yolo_model(torch_image)[0]
            bbox_pred = non_max_suppression(bbox_pred, conf_thres=config.min_bbox_conf)[0].cpu()
            t2 = time.time_ns()
            yolo_time_ms = (t2-t1) / 1e6
            torch_image = torch_image.squeeze(0)
            ratios = []
            crops = torch.zeros([len(bbox_pred), 3, config.kp_resize_h, config.kp_resize_w]).half().to('cuda:0')
            new_i = 0
            indices = []
            for i, det in enumerate(bbox_pred):
                det = det.unsqueeze(0)
                if len(det):
                    for *xyxy, conf, cone_class in reversed(det):
                        if conf > config.min_bbox_conf and cone_class in [2, 7]:
                            top = xyxy[1].to(torch.int32)
                            bot = xyxy[3].to(torch.int32)
                            left = torch.max(torch.zeros_like(xyxy[0]), xyxy[0].to(torch.int32)).to(torch.int32)
                            right = xyxy[2].to(torch.int32)
                            if bot - top > config.kp_min_h and right - left > config.kp_min_w and (bot - top) / (right - left) < 4 / 3:
                                crops[new_i, :, :, :] = resize(torch_image[:, top:bot, left:right], [config.kp_resize_h, config.kp_resize_w])
                                ratios.append(torch.tensor([config.kp_resize_w / (right - left), config.kp_resize_h / (bot - top)]))
                                indices.append(i)
                                new_i += 1

            t1 = time.time_ns()
            # Regress keypoints on to the found cones
            points = self.kp_model(crops[:new_i]).cpu()
            t2 = time.time_ns()
            kp_time_ms = (t2-t1) / 1e6

        t1 = time.time_ns()
        R_car = Rotation.from_euler("xyz", [0, 0, car_hdg-np.deg2rad(90)]).as_matrix()
        P_car = car_pos
        cones = []
        for i in range(len(points)):
            cones.append(self._process_cones(points[i], ratios[i], bbox_pred[indices][i], R_car, P_car))
        cones = np.asarray(cones)
        t2 = time.time_ns()
        pnp_time_ms = (t2-t1) / 1e6

        try:
            blue_cones = cones[cones[:, 0] == 7][:, 1:4]
            blue_pixels = cones[cones[:, 0] == 7][:, 4:]
        except Exception:
            blue_cones = np.array([])
            blue_pixels = np.array([])
        try:
            yellow_cones = cones[cones[:, 0] == 2][:, 1:4]
            yellow_pixels = cones[cones[:, 0] == 2][:, 4:]
        except Exception:
            yellow_cones = np.array([])
            yellow_pixels = np.array([])

        outputs = {
            "blue_cones": blue_cones,
            "yellow_cones": yellow_cones,
            "blue_pixels": blue_pixels,
            "yellow_pixels": yellow_pixels,
            "yolo_ms": yolo_time_ms,
            "kp_ms": kp_time_ms,
            "pnp_ms": pnp_time_ms,
        }

        return outputs

    def _process_cones(self, img_points, ratios, det, R_car, P_car):
        left = det[0].to(torch.int32).numpy()
        top = det[1].to(torch.int32).numpy()
        right = det[2].to(torch.int32).numpy()
        bot = det[3].to(torch.int32).numpy()
        cone_class = det[5].to(torch.int32).numpy()

        img_points = (img_points.reshape(7, 2) / ratios).numpy() + [left, top]
        retval, rvec, tvec = cv2.solvePnP(config.cone_3d_points, img_points, config.K, np.zeros(4))
        if retval:
            T = RpToTrans(np.eye(3), tvec)
            T = config.camera_T @ TransInv(T)
            cone_pos = T @ np.array([0, 0, 0, 1]) * np.array([-1, -1, 0, 0])
            if config.min_valid_cone_distance < cone_pos[1] < config.max_valid_cone_distance:
                cone_class = np.array([cone_class])
                return np.concatenate([cone_class, (R_car @ cone_pos[:3] + P_car[:3]), np.array([left, top, right, bot]), img_points.flatten()])

        return np.zeros(22)