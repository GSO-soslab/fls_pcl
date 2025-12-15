#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Int16MultiArray
from rcl_interfaces.msg import SetParametersResult
from oculus_interfaces.msg import Ping

class Plot_FLS_Beam(Node):
    def __init__(self):
        super().__init__('plot_fls_beam')

        self.bridge = CvBridge()

        # Subscribe to image topic
        self.create_subscription(
            Image,
            '/alpha_rise/oculus/raw_image',   # <-- change topic name if needed
            self.image_callback,
            10
        )

        self.image_publisher = self.create_publisher(Int16MultiArray, '/alpha_rise/oculus/raw_image/intensity', 10)
        # self.image_publisher = self.create_publisher(Image, '/alpha_rise/oculus/raw_image/intensity', 10)

        self.create_subscription(Ping, '/alpha_rise/oculus/ping' ,self.ping_CB, 10)
        self.declare_parameter('sonar_beam_number', 2)
        self.sonar_beam_number = self.get_parameter('sonar_beam_number').value

        # Register callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)
    
    def ping_CB(self, ping):
        self.range = ping.range

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'sonar_beam_number' and param.type_ == param.Type.INTEGER:
                self.sonar_beam_number = param.value
                self.get_logger().info(f"Updated sonar_beam_number={self.sonar_beam_number}")
        return SetParametersResult(successful=True)
    
    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        # print(cv_img.shape, flush=True)
        
        #Sonar TF
        rotated = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
        rotated = cv2.flip(rotated, 0)   # flip vertically
        self.sonar_beam_number = rotated.shape[0]//2 + 12

        # self.image_publisher.publish(self.bridge.cv2_to_imgmsg(rotated))

        if rotated.shape[0]>self.sonar_beam_number > 0:
            beam_intenity = rotated[self.sonar_beam_number, :]
            
            msg = Int16MultiArray()
            msg.data = beam_intenity
            self.image_publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = Plot_FLS_Beam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
