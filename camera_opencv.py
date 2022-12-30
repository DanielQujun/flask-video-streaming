import os
import cv2
from base_camera import BaseCamera
from utils.utils_rotate import RotateClockWise180
from distutils.util import strtobool

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
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()
            if NEED_REVERSE:
                print("reverse img!")
                img = RotateClockWise180(img)
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
