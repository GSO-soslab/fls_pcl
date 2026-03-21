#!/usr/bin/env python3

# 512 * 517 * 12 = 3176448 points.
# Arc length along azimuth = tan(35)*40*2 = 56m
# Arc length along elevation = tan(6)*40*2 = 8.4m
import math
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from rclpy.parameter import Parameter


class FLS_Voxels(Node):
    '''
    ROS2 node that publishes voxels defining SONAR FOV
    '''
    def __init__(self):
        super().__init__('fls_voxel_node')

        # ---- Parameters ----
        self.declare_parameter('range_max', Parameter.Type.DOUBLE)
        self.declare_parameter('horizontal_fov_deg', Parameter.Type.DOUBLE)
        self.declare_parameter('vertical_fov_deg', Parameter.Type.DOUBLE)
        self.declare_parameter('resolution', Parameter.Type.DOUBLE)  # meters
        self.declare_parameter('sensor_frame_id', Parameter.Type.STRING)
        self.declare_parameter('marker_topic', Parameter.Type.STRING)

        self.max_range = self.get_parameter('range_max').value
        self.h_fov = math.radians(self.get_parameter('horizontal_fov_deg').value )
        self.v_fov = math.radians(self.get_parameter('vertical_fov_deg').value )
        self.resolution = self.get_parameter('resolution').value
        self.frame_id = self.get_parameter('sensor_frame_id').value
        marker_topic = self.get_parameter('marker_topic').value

        # ---- Publishers ----
        self.marker_pub = self.create_publisher(Marker, marker_topic, 10)
        self.create_timer(0.2, self.publish_all)
    
    def publish_all(self):
        # self.publish_marker()
        self.publish_voxel_fov_marker()

    def make_point(self, x, y, z):
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p
    
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

    def publish_voxel_fov_marker(self):

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = 'fov_voxels'
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = self.resolution

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3

        marker.points = []

        h_half = self.h_fov / 2.0
        v_half = self.v_fov / 2.0

        x_steps = int(math.ceil(self.max_range / self.resolution))

        for ix in range(x_steps):
            # voxel center in x
            x = (ix + 0.5) * self.resolution
            if x > self.max_range:
                continue

            y_limit = x * math.tan(h_half)
            z_limit = x * math.tan(v_half)

            y_steps = int(math.ceil((2.0 * y_limit) / self.resolution))
            z_steps = int(math.ceil((2.0 * z_limit) / self.resolution))
            # print(f"y_steps:{y_steps}, z_steps:{z_steps}")
            for iy in range(-y_steps // 2, y_steps // 2 + 1):
                y = iy * self.resolution
                if abs(y) > y_limit + self.resolution * 0.5:
                    continue

                for iz in range(-z_steps // 2, z_steps // 2 + 1):
                    z = iz * self.resolution
                    if abs(z) > z_limit + self.resolution * 0.5:
                        continue

                    marker.points.append(self.make_point(x, y, z))

        self.marker_pub.publish(marker)
        # print(len(marker.points), flush=True)

def main():
    rclpy.init()
    node = FLS_Voxels()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
