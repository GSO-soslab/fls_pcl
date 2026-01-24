#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageThresholdRecorder(Node):

    def __init__(self):
        super().__init__('image_threshold_recorder')

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/alpha_rise/oculus/raw_image',
            self.image_callback,
            10
        )

        self.video_writer = None
        self.output_path = (
            '/home/tony/auv_ws/src/fls_pcl/utils/images/output.mp4'
        )
        self.get_logger().info("Initialized")
    def apply_threshold(self, img, upper):
        """Lower threshold fixed at 22, upper threshold variable"""
        out = img.copy()
        out[out < 22] = 0
        out[out > upper] = 255
        return out

    def image_callback(self, msg):
        try:
            # Convert to grayscale
            gray = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='mono8'
            )

            # Apply thresholds
            thresh_50 = self.apply_threshold(gray, 50)
            # thresh_60 = self.apply_threshold(gray, 60)

            # Column stack: original | >50 | >60
            stacked = np.hstack((gray, thresh_50))

            # Convert to BGR for video
            stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)

            # Initialize writer once
            if self.video_writer is None:
                h, w, _ = stacked_bgr.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    30.0,
                    (w, h)
                )
                self.get_logger().info(
                    f"Recording video to {self.output_path}"
                )

            self.video_writer.write(stacked_bgr)

        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageThresholdRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
