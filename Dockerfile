FROM harbor.tiduyun.com/qujun/miniforge3_camera_x86:latest
WORKDIR /home/app
ADD . /home/app
ENV CAMERA=opencv
ENV REVERSE=True
ENV VIDEO_SOURCE=0
CMD ["python", "app.py"]