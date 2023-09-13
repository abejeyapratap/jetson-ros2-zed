import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
import torch
import onnxruntime
from PIL import Image as PILImage
from PIL import ImageDraw
import time
import sys
sys.path.insert(0,"/root/yolo/yolov5")
from utils.general import non_max_suppression

class ZEDImageSubscriber(Node):
    def __init__(self):
        super().__init__('zed_image_subscriber')
        self.image_sub = self.create_subscription(Image, '/zed2/zed_node/left/image_rect_color', self.image_callback, 10)
        self.latest_image = None

    def image_callback(self, msg):
        try:
            # Convert ROS image data to NumPy array
            im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.latest_image = im[:, :, [2, 1, 0]]
        except Exception as e:
            self.get_logger().error(str(e))

class ObjectDetector:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        # Define the stop publisher
        #self.stop_publisher = Node.create_publisher(Bool, '/stop_detected', 1)

    def detect_objects(self, image):
        # Convert NumPy image array to PIL format
        im_pil = PILImage.fromarray(image)

        # Preprocess image
        im_resized = im_pil.resize((640, 640))
        input_data = np.array(im_resized).astype(np.float32) / 255.0  # Convert to float and normalize
        input_data = np.transpose(input_data, (2, 0, 1))  # Channel-first format
        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        start_time = time.time()
        # Run inference
        results = self.session.run(None, {'images': input_data})
        end_time = time.time()
        print(f"Time taken for detection: {end_time - start_time:.4f} seconds")

        # Post-process detections
        detections = self.post_process(results)

        # Visualize detections
        im_with_boxes = self.visualize_detections(im_resized, detections)
        return np.array(im_with_boxes)[:, :, [2, 1, 0]]

    def post_process(self, results):
        # Post-process YOLOv5 output
        # ... Perform necessary reshaping and non-maximum suppression here ...
        # Return list of detections, each detection is a list [x1, y1, x2, y2, conf, cls]
        detections = results[0]
        detections = non_max_suppression(torch.Tensor(detections), conf_thres=0.6, iou_thres=0.4)[0]
        return detections.tolist()

    def visualize_detections(self, image, detections):
        # Visualize detections on the image
        im_draw = image.copy()
        draw = ImageDraw.Draw(im_draw)
        stop_sign_detected = False
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            class_name = f"Class {cls}"
            label = f"{class_name}: {conf:.2f}"
            color = (0, 255, 0)
            if cls == 1:
                color = (255, 0, 0)
                s = (x2-x1)*(y2-y1)
                #print("S: ", s)
                if s >= 5500:
                    stop_sign_detected = True
            elif cls == 2:
                color = (0, 0, 255)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1), label, fill=color)
        stop_msg = Bool(data=stop_sign_detected)
        #self.stop_publisher.publish(stop_msg)
        return im_draw

def main():
    rclpy.init()
    model_path = './modelyolos.onnx'  # Path to your ONNX model

    image_subscriber = ZEDImageSubscriber()
    object_detector = ObjectDetector(model_path)

    try:
        while rclpy.ok():
            rclpy.spin_once(image_subscriber)
            if image_subscriber.latest_image is not None:
                image = image_subscriber.latest_image
                im_with_boxes = object_detector.detect_objects(image)

                cv2.imshow("Left Camera Image with Detection", im_with_boxes)
                cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.destroyAllWindows()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

