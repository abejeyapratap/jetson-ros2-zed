#!/bin/bash
### Source: Yahboom Rosmaster R2 docs + Stereolabs Zed docs
xhost +

docker run -it \
--net=host \
--runtime nvidia \
--privileged \
-e DISPLAY \
--env="QT_X11_NO_MITSHM=1" \
-v /tmp/.X11-unix:/tmp/.X11-unix \
abejeyapratap/zed_foxy_yahboom:dev /bin/bash
