#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray
import tf_transformations


class SparseVoxelLogOddsVisualizer(Node):
    def __init__(self):
        super().__init__('sparse_voxel_logodds_visualizer')

        # Parameters
        self.declare_parameter('voxel_resolution', 1.0)
        self.declare_parameter('grid_size', 200.0)
        self.declare_parameter('logodds_min', -20.0)
        self.declare_parameter('logodds_max', 20.0)
        self.declare_parameter('frame_id', 'alpha_rise/odom')

        self.voxel_res = self.get_parameter('voxel_resolution').get_parameter_value().double_value
        self.grid_size = self.get_parameter('grid_size').get_parameter_value().double_value
        self.logodds_min = self.get_parameter('logodds_min').get_parameter_value().double_value
        self.logodds_max = self.get_parameter('logodds_max').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.prob_threshold = 0.5

        # Voxel grid size (number of voxels along each axis)
        self.n_voxels = int(np.ceil(self.grid_size / self.voxel_res))

        # Sparse structures: only store voxels that have been observed
        self.logodds_grid = {}  # key: (i,j,k), value: log-odds
        self.markers = {}       # key: (i,j,k), value: Marker

        # ROS subscriber + publisher
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/alpha_rise/fls/pointcloud/post',
            self.pc_callback,
            10
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/alpha_rise/voxel_map', 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Sparse 3D voxel grid initialized: resolution={self.voxel_res}m, grid_size={self.grid_size}m")

    def pc_callback(self, msg: PointCloud2):
        # Transform from sensor frame → odom
        try:
            trans = self.tf_buffer.lookup_transform(
                self.frame_id,
                msg.header.frame_id,
                msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Extract points robustly
        gen = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "intensity"),
            skip_nans=True
        )
        pointclouds = np.fromiter(
            gen,
            dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
        )

        if pointclouds.size == 0:
            return

        # Structured → (N,4)
        points = np.vstack([
            pointclouds['x'],
            pointclouds['y'],
            pointclouds['z'],
            pointclouds['intensity']
        ]).T
        xyz = points[:, :3]
        probs = points[:, 3]

        # Transform points to odom frame
        xyz_hom = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
        T = self.transform_to_matrix(trans)
        xyz_odom = (T @ xyz_hom.T).T[:, :3]

        # Remove NaN/Inf points
        mask_valid = np.all(np.isfinite(xyz_odom), axis=1)
        xyz_odom = xyz_odom[mask_valid]
        probs = probs[mask_valid]

        if xyz_odom.shape[0] == 0:
            return

        # Compute voxel indices (3D)
        half_grid = self.grid_size / 2.0
        voxel_idx = np.floor((xyz_odom + half_grid) / self.voxel_res).astype(int)

        # Keep only indices inside the cube
        mask_in_grid = np.all((voxel_idx >= 0) & (voxel_idx < self.n_voxels), axis=1)
        voxel_idx = voxel_idx[mask_in_grid]
        probs = probs[mask_in_grid]

        if len(probs) == 0:
            return

        # Sparse log-odds update
        logodds_update = np.log(probs / (1 - probs))
        for idx, key in enumerate(map(tuple, voxel_idx)):
            if key in self.logodds_grid:
                self.logodds_grid[key] += logodds_update[idx]
            else:
                self.logodds_grid[key] = logodds_update[idx]

            # Clip log-odds
            self.logodds_grid[key] = np.clip(self.logodds_grid[key], self.logodds_min, self.logodds_max)

        # Update markers
        self.update_markers()

    def update_markers(self):
        """Update persistent markers for all observed voxels with probability > threshold."""
        half_grid = self.grid_size / 2.0
        marker_array = MarkerArray()

        for key, logodds in self.logodds_grid.items():
            # Convert log-odds → probability
            prob = 1 - 1 / (1 + np.exp(logodds))
            prob = np.clip(prob, 0.0, 1.0)

            # Skip voxels below threshold
            if prob < self.prob_threshold:
                continue

            # Create marker if not already
            if key not in self.markers:
                i, j, k = key
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.ns = "voxels"
                marker.id = len(self.markers)
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = i * self.voxel_res - half_grid + self.voxel_res/2
                marker.pose.position.y = j * self.voxel_res - half_grid + self.voxel_res/2
                marker.pose.position.z = k * self.voxel_res - half_grid + self.voxel_res/2
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = self.voxel_res
                self.markers[key] = marker

            marker = self.markers[key]
            # Set color & alpha
            marker.color.r = 1.0 - prob
            marker.color.g = prob
            marker.color.b = 0.0
            marker.color.a = np.clip(prob, 0.1, 1.0)

            marker_array.markers.append(marker)

        # Update timestamps and publish
        now = self.get_clock().now().to_msg()
        for m in marker_array.markers:
            m.header.stamp = now
        self.marker_pub.publish(marker_array)


    def transform_to_matrix(self, trans):
        """Convert TransformStamped → 4x4 homogeneous matrix"""
        t = trans.transform.translation
        q = trans.transform.rotation
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T


def main(args=None):
    rclpy.init(args=args)
    node = SparseVoxelLogOddsVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
