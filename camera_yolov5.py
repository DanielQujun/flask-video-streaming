import os
import time
import cv2

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import (check_img_size, non_max_suppression, plot_one_box,
                           scale_coords, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.utils_rotate import RotateClockWise180
from base_camera import BaseCamera
from distutils.util import strtobool

# Initialize
set_logging()


class Optclass(object):

    def __init__(self, source="0", conf_thres=0.25):
        self.weights = "yolov5s.pt"
        self.source = source
        self.view_img = True
        self.img_size = 416
        self.augment = ""
        self.conf_thres = conf_thres
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.webcam = self.source.isnumeric() or self.source.startswith(('rtsp://', 'rtmp://', 'http://')) or self.source.endswith('.txt')


class PredictModel(object):
    def __init__(self):
        self.opt = Optclass(source=os.environ.get("VIDEO_SOURCE", 0))
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self.init_model()

    def init_model(self):
        
        # Load model
        model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.opt.img_size, s=model.stride.max())  # check img_size
        if self.half:
            model.half()  # to FP16

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        # Run inference

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        return model

    def detect(self, path, img, im0s):
        t0 = time.time()

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.opt.webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.opt.view_img:  # Add bbox to image
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

        print('Done. (%.3fs)' % (time.time() - t0))
        return im0

p_model = PredictModel()

NEED_REVERSE = strtobool(str(os.environ.get('REVERSE', "False")))


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(p_model.opt.source, img_size=p_model.imgsz)
        print(f'load dataset finished!')

        while True:
            # read current frame
            for path, img, im0s, _ in dataset:
                predict_img = p_model.detect(path, img, im0s)
                if NEED_REVERSE:
                    print("revesed img!")
                    reverse_img = RotateClockWise180(predict_img)
                else:
                    reverse_img = predict_img
                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', reverse_img)[1].tobytes()

if __name__ == "__main__":
    pass