docker run -it --net=host --privileged=true -e CAMERA=yolov5 -e REVERSE=False -e VIDEO_SOURCE="http://192.168.1.210:17227/video_feed" \
-v /dev:/dev \
-v /home/danielqu/data/mypi/projects/flask-video-streaming:/home/app \
miniforge3_camera_flask_x86:latest
