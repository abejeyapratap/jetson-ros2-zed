# jetson-ros2-zed

This repository documents my experiments integrating a Zed 2 AI Stereo camera, ROS2 Foxy, and YOLOv5 object detection with a Rosmaster R2 rover that has a Jetson NX and 4ROS Lidar.

## System Requirements
### Hardware:
- Yahboom Rosmaster R2
- Nvidia Jetson Xavier NX
- Stereolabs Zed 2 camera

### Software:
- Ubuntu 20.04 (L4T 35.3.1)
- Cuda 11.4
- Docker

## System Compatibility & Usage
1. Verify system compatibility (later versions may work too but not tested):
   1. `cat /etc/nv_tegra_release`
   2. `nvcc --version`
2. Create udev rules to map Lidar and Serial USB ports to hard-link names:
   1. Place `usb.rules` in `/etc/udev/rules.d` & `udev-usb-tweak.sh` in `/root/bin`
   2. Reboot Jetson & test using `ls /dev/myserial && ls /dev/rplidar`
3. Verify Zed 2 connectivity using `ll /dev/video*`
4. Run Docker container in privileged mode using `run_docker.sh` (pulls the latest working docker image)

### Examples
`scripts/` contains examples written in ROS2 Python for object detection using Zed + YOLO

### Full setup
Please see TODO.md for how these images were created

### Docker Image Repos:
- Images for [ZED, ROS2, and YOLO](https://hub.docker.com/repository/docker/abejeyapratap/zed_foxy_focal/general)
- Images with support for [Yahboom packages](https://hub.docker.com/repository/docker/abejeyapratap/zed_foxy_yahboom/general)
