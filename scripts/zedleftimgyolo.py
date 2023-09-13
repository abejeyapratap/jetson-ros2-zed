import rclpy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms
from PIL import Image as PILImage
from PIL import ImageDraw
import time
import sys
#sys.path.insert(0, "/home/jetson/zed2_recording_for_yolo/yolov5")

class ZEDImageSubscriber:
    def __init__(self):
        self.node = rclpy.create_node('zed_image_viewer')
        self.image_sub = self.node.create_subscription(
            Image, '/zed2/zed_node/left/image_rect_color', self.image_callback, 10)
        self.latest_image = None
        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            print(e)

class ObjectDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        self.yolov5_model = self.load_yolov5_model()

    def load_yolov5_model(self):
        # Load pre-trained YOLOv5s model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
        model.to(self.device).eval()
        return model

    def detect_objects(self, image):
        # Perform object detection using YOLOv5
        start_time = time.time()
        results = self.yolov5_model(image)
        end_time = time.time()
        print(f"Time taken for detection: {end_time - start_time:.4f} seconds")

        im_with_boxes = results.render()[0]
        return im_with_boxes

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ZEDImageSubscriber()
    object_detector = ObjectDetector()

    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber.node)
            if image_subscriber.latest_image is not None:
                image = image_subscriber.latest_image
                im_with_boxes = object_detector.detect_objects(image)

                cv2.imshow("Left Camera Image with Detection", im_with_boxes)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.destroyAllWindows()

    image_subscriber.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

