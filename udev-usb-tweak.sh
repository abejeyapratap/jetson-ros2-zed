#!/bin/bash
# Source: https://hackaday.io/page/13294-solved-docker-udev-usb-naming

p=""
if [ "$1" == "myserial" ]
then
  p=`realpath /dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`
elif [ "$1" == "rplidar" ]
then
  p=`realpath /dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0`
fi

if  [ "x$p" != "x" ]
then
    rm -f /dev/ttyUSB-$1
    ln $p /dev/ttyUSB-$1
fi
