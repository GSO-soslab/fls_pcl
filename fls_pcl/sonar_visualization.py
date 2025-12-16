#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class LineMarkerNode(Node):

    def __init__(self):
        super().__init__('line_marker_node')

        self.marker_pub = self.create_publisher(
            Marker,
            'fls/geometry',
            10
        )

        self.timer = self.create_timer(0.1, self.publish_marker)

    def publish_marker(self):
        marker = Marker()

        # Frame the marker is attached to
        marker.header.frame_id = 'alpha_rise/fls_link'
        # marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'line'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line width (meters)
        marker.scale.x = 0.05

        # Color (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # ---- FOV parameters ----
        length = 40.0

        v_half = math.radians(12.0 / 2.0)  # vertical
        h_half = math.radians(70.0 / 2.0)  # horizontal

        v_extent = length * math.tan(v_half)
        h_extent = length * math.tan(h_half)

        origin = Point(x=0.0, y=0.0, z=0.0)
        marker.points = []

        # 4 corner boundary rays
        corners = [
            ( length,  h_extent,  v_extent),  # top-right
            ( length, -h_extent,  v_extent),  # top-left
            ( length,  h_extent, -v_extent),  # bottom-right
            ( length, -h_extent, -v_extent),  # bottom-left
        ]

        for x, y, z in corners:
            end = Point(x=x, y=y, z=z)
            marker.points.append(origin)
            marker.points.append(end)

        self.marker_pub.publish(marker)
def main():
    rclpy.init()
    node = LineMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
