#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Float32
from std_srvs.srv import SetBool
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs
from collections import deque
from rclpy.time import Time
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d

class PointCloudBuffer(Node):
    def __init__(self):
        super().__init__('depth_planner')
        
        # Parameters
        self.pointcloud_topic = '/alpha_rise/fls/pointcloud'
        self.target_frame = 'alpha_rise/odom'
        self.queue_size = 100
        self.merged_msg = None
        self.save_map = False
        self.plan_depth = False
        self.within_distance = False

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pointcloud_queue = deque()
        self.pointcloud_queue_local = deque(maxlen=self.queue_size)

        # Subscriber
        self.create_subscription(PointCloud2,self.pointcloud_topic,self.pointcloud_callback,10)
        self.create_subscription(Float32,'/alpha_rise/path/distance_to_obstacle',self.distance_callback, 10)

        self.create_service(SetBool, '/alpha_rise/iceberg/save_map', self.save_map_service_cb)
        self.create_service(SetBool, '/alpha_rise/iceberg/plan_depth', self.plan_depth_service_cb)


        # Publisher
        self.publisher = self.create_publisher(PointCloud2,'/alpha_rise/fls/pointcloud/submap',10)
        self.depth_publisher = self.create_publisher(Int16,'/alpha_rise/path/depth',10)
        self.surge_publisher = self.create_publisher(Int16,'/alpha_rise/path/surge',10)


    def plan_depth_service_cb(self, request, response):
        if request.data:
            self.plan_depth = True
            response.success = True
            response.message = "Planning Depth"
        
            return response
        else:
            self.plan_depth = False
            response.success = False
            response.message = "Depth Planning Stoppped"
            return response
    def save_map_service_cb(self, request, response):
        if request.data:
            response.success = True
            response.message = "Revisit triggered."
            self.save_map = True
            return response

    def process_submap(self):
        if self.save_map:
            if isinstance(self.merged_msg, PointCloud2):
                points = np.fromiter(
                    pc2.read_points(self.merged_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True),
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
                )

                if len(points) == 0:
                    self.get_logger().warn("No valid points in the latest pointcloud.")
                    return

                # Convert structured array to regular Nx4 array
                np_points = np.stack([points['x'], points['y'], points['z'], points['intensity']], axis=-1)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np_points[:, :3])  # only xyz
                # Optional coloring
                intensity = np_points[:, 3]
                colors = np.tile(intensity[:, None], (1, 3))
                colors = colors / np.max(colors)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                o3d.io.write_point_cloud("submap.pcd", pcd)
                print("SAVED", flush=True)
                self.save_map = False


            else:
                self.get_logger().error("merged_msg is not a valid PointCloud2 message")
    
    def pointcloud_callback(self, msg: PointCloud2):
        try:
            # Transform point cloud to 'odom' frame
            trans_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            transformed_cloud = tf2_sensor_msgs.do_transform_cloud(msg, trans_msg)
            self.pointcloud_queue.append(transformed_cloud)

            if self.plan_depth:
                self.pointcloud_queue_local.append(transformed_cloud)

            # self.get_logger().info(f"Stored pointcloud #{len(self.pointcloud_queue)} in odom frame")
            self.publish_stored_pointclouds()

            self.depth_planner()


        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f"TF transform failed: {str(e)}")

    def depth_planner(self):
        if self.plan_depth:
            if self.within_distance:
                print("PLANNING DEPTH")

                #Slow speed
                msg = Int16()
                msg.data = 1
                self.surge_publisher.publish(msg)

                #Store pointclouds(?)

                #Capture keypoints and compare

                #Publish increment depth command
                # msg = Int16()
                # msg.data = 1

                # self.depth_publisher.publish(msg)
            else:
                msg = Int16()
                msg.data = 0
                self.surge_publisher.publish(msg)  
        else:
            msg = Int16()
            msg.data = 0
            self.surge_publisher.publish(msg)            

    def publish_stored_pointclouds(self):
        if not self.pointcloud_queue:
            self.get_logger().info("No pointclouds stored yet.")
            return

        # self.get_logger().info(f"Publishing merged pointcloud with {len(self.pointcloud_queue)} messages")

        merged_points = []

        # Convert all PointCloud2 messages to lists of (x, y, z, ...) tuples
        for pc in self.pointcloud_queue:
            points = list(pc2.read_points(pc, skip_nans=True))
            merged_points.extend(points)

        if not merged_points:
            self.get_logger().warn("No valid points to publish")
            return

        # Assume all point clouds have same fields as the first one
        fields = self.pointcloud_queue[0].fields
        header = self.pointcloud_queue[0].header
        header.frame_id = self.target_frame  # e.g., 'odom'

        # Optionally update stamp to last msg or now
        header.stamp = self.pointcloud_queue[-1].header.stamp

        # Create merged PointCloud2 message
        self.merged_msg = pc2.create_cloud(header, fields, merged_points)
        self.process_submap()
        # Publish merged pointcloud
        self.publisher.publish(self.merged_msg)

    def distance_callback(self, msg):
        print(msg.data)
        if msg.data < 1.1*20:
            self.within_distance = True
        else:
            self.within_distance = False




def main(args=None):
    rclpy.init(args=args)
    node = PointCloudBuffer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
