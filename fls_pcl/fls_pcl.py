#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
import cv2
import open3d as o3d
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_sensor_msgs.tf2_sensor_msgs
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformException
import struct
from math import nan
from oculus_interfaces.msg import Ping

class FLS_PCL(Node):
    def __init__(self):
        super().__init__('fls_pcl_node')
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Declare and read parameters
        self.declare_parameter('max_depth',Parameter.Type.DOUBLE)
        self.declare_parameter('min_depth',Parameter.Type.DOUBLE)

        # self.declare_parameter('threshold_intensity',Parameter.Type.INTEGER)
        self.declare_parameter('threshold_min_range',Parameter.Type.DOUBLE)
        self.declare_parameter('threshold_max_range',Parameter.Type.DOUBLE)

        self.declare_parameter('beam_skip_count',Parameter.Type.INTEGER)
        self.declare_parameter('frame_id',Parameter.Type.STRING)
        self.declare_parameter('ping_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('image_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('pointcloud_pub_topic', Parameter.Type.STRING)
        self.declare_parameter('save_as_pcd', Parameter.Type.BOOL)
        self.declare_parameter('pcd_filename', Parameter.Type.STRING)
        self.declare_parameter('filter_mode', Parameter.Type.STRING)

        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        # self.intensity_threshold = self.get_parameter('threshold_intensity').value
        self.threshold_min_range = self.get_parameter('threshold_min_range').value
        self.threshold_max_range = self.get_parameter('threshold_max_range').value
        self.beam_skip_count = self.get_parameter('beam_skip_count').value
        self.frame_id = self.get_parameter('frame_id').value
        ping_sub_topic = self.get_parameter('ping_sub_topic').value
        sub_topic = self.get_parameter('image_sub_topic').value
        pub_topic = self.get_parameter('pointcloud_pub_topic').value
        self.save_as_pcd_bool = self.get_parameter('save_as_pcd').value
        self.pcd_filename = self.get_parameter('pcd_filename').value
        self.filter_mode = self.get_parameter('filter_mode').value

        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.pub_pcl = self.create_publisher(PointCloud2, pub_topic, 10)
        # self.pub_fls_frost_image = self.create_publisher(Image, pub_topic+'/image/frost', 10)
        self.pub_fls_edge_image = self.create_publisher(Image, pub_topic+'/image/edge', 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/median', 10)

        # Subscriber
        self.create_subscription(Image,sub_topic,self.image_CB,10)
        self.create_subscription(PointCloud2, pub_topic,self.save_as_pcd, 10)
        self.create_subscription(Ping, ping_sub_topic ,self.ping_CB, 10)

        # Populate PointCloud2 message
        self.pointcloud_msg = PointCloud2()
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pointcloud_msg.fields = self.fields
        self.pointcloud_msg.point_step = 4 * (len(self.fields))  # Each point occupies 16 bytes
        self.pointcloud_msg.is_dense = True  # All points are valid

        # Persistent accumulated cloud
        self.accumulated_cloud = o3d.geometry.PointCloud()
        self.receive_ping = False

        # Register shutdown handler
        rclpy.get_default_context().on_shutdown(self.on_shutdown)

    def ping_CB(self,msg):
        self.bearings = np.array([np.radians(bearings * 0.01) for bearings in msg.bearings]).squeeze()
        if self.receive_ping == False:
            # np.savetxt("bearings.txt", self.bearings, fmt="%.8f")
            self.max_range = msg.range

            # High / Low frequency mode
            if msg.master_mode == 2:
                self.horizontal_beamwidth = 70 
            elif msg.master_mode == 1:
                self.horizontal_beamwidth = 130
        
            self.receive_ping = True

    def image_CB(self, msg):
        if self.receive_ping:
            current = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            rows, columns = current.shape
            self.n_bins, self.n_beams = rows, columns

            # Median Filtering, Higher ksize, stronger smoothening, higher comp
            current = cv2.medianBlur(current, ksize=5)

            # === Convert all valid pixels to sensor frame coordinates ===
            edge_list, sensor_frame = self.extract_all_points_in_sensor_frame(current, mode=self.filter_mode)
            
            edge_image = np.zeros((rows, columns), dtype=np.float32)

            if len(edge_list) == 0:
                self.get_logger().warn("No measurements", throttle_duration_sec=3)
                self.points = np.empty((0, len(self.fields)), dtype=np.float32)
                return

            # === Configure PointCloud2 metadata ===
            h = Header()
            h.frame_id = self.frame_id
            self.pointcloud_msg.header = h
            self.pointcloud_msg.height = 1

            # === Vectorized filtering ===
            sensor_x = sensor_frame[:, 0]
            sensor_y = sensor_frame[:, 1]
            intensity = sensor_frame[:, 2]

            # Filter valid points
            valid_x = sensor_x
            valid_y = sensor_y
            valid_i = intensity

            # === Allocate and fill the point array ===
            num_points = len(valid_x)
            self.pointcloud_msg.width = num_points
            self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

            self.points = np.full((num_points, len(self.fields)), np.nan, dtype=np.float32)
            if num_points > 0:
                self.points[:, 0] = valid_x
                self.points[:, 1] = valid_y
                self.points[:, 2] = 0.0  # Z = 0 for 2D sensor
                self.points[:, 3] = valid_i

            # === Build visualization image ===
            rows_i = edge_list[:, 0].astype(int)
            cols_i = edge_list[:, 1].astype(int)
            intensities_i = edge_list[:, 2]
            edge_image[rows_i, cols_i] = intensities_i

            # Normalize to 0–255 for display
            edge_image = cv2.normalize(edge_image, None, 0, 255, cv2.NORM_MINMAX)
            edge_image = edge_image.astype(np.uint8)

            median_current = cv2.normalize(current, None, 0, 255, cv2.NORM_MINMAX)
            median_current = median_current.astype(np.uint8)

            # === Publish results ===
            self.pub_fls_edge_image.publish(self.bridge.cv2_to_imgmsg(edge_image, encoding="mono8"))
            self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(median_current, encoding="mono8"))

            self.pointcloud_msg.data = self.points.tobytes()
            if self.filter_mode == "all":
                self.pub_pcl.publish(self.pointcloud_msg)
            
            elif self.filter_mode == "max_intensity":
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'alpha_rise/world',
                        self.frame_id,
                        rclpy.time.Time()
                    )

                    # Apply transform
                    self.pointcloud_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(self.pointcloud_msg, transform)

                    # Read points including intensity
                    points = list(pc2.read_points(
                        self.pointcloud_msg,
                        field_names=('x', 'y', 'z', 'intensity'),
                        skip_nans=False
                    ))

                    # Modify points: set points with z > -1.0 to zero
                    new_points = []
                    for x, y, z, intensity in points:
                        if z > self.max_depth:
                            x, y, z = nan, nan, nan
                        elif z< self.min_depth:
                            x, y, z = nan, nan, nan
                        new_points.append([x, y, z, intensity])

                    # Create new PointCloud2 preserving intensity
                    self.pointcloud_msg = pc2.create_cloud(self.pointcloud_msg.header, self.fields, new_points)
                    try:
                        transform = self.tf_buffer.lookup_transform(
                        self.frame_id,
                        'alpha_rise/world',
                        rclpy.time.Time()
                        )

                        # Apply transform
                        self.pointcloud_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(self.pointcloud_msg, transform)

                        self.pointcloud_msg.is_dense = True
                        self.pub_pcl.publish(self.pointcloud_msg)
                        # self.pub_pcl_depth_filtered.publish(pc_modified)
                    except TransformException as e:
                        self.get_logger().warn(f'Transform not available: {e}')
                except TransformException as e:
                    self.get_logger().warn(f'Transform not available: {e}')
                

        
    def extract_all_points_in_sensor_frame(self, image, mode='all'):
        """
        Convert pixels in the image into sensor-frame coordinates (x, y, intensity),
        optionally filtering by beam skipping and range threshold first, then applying
        intensity selection mode.

        Parameters
        ----------
        image : np.ndarray
            2D array (rows × columns) representing intensity values.
        mode : str, optional
            'all' - keep all points above threshold
            'max_intensity' - keep only the highest-intensity point per column

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - image_coordinates: array of (row_index, column_index, intensity_value)
            - sensor_frame_coordinates: array of (x, y, intensity_value)
        """
        if image is None:
            raise ValueError("Image not found or unable to load.")

        range_factor = self.max_range / self.n_bins

        # Create coordinate grids
        rows, cols = np.indices(image.shape)
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        intensities = image.flatten()

        theta_values = self.bearings[cols_flat]
        r_values = range_factor * (self.n_bins - rows_flat)

        # Convert to sensor frame (x, y)
        sensor_x = r_values * np.cos(theta_values)
        sensor_y = r_values * np.sin(theta_values)

        # Apply beam skipping
        indices = np.arange(len(sensor_x))
        mask_beam_skip = (indices % self.beam_skip_count) == 0

        # Apply range threshold
        distances = np.sqrt(sensor_x**2 + sensor_y**2)
        mask_min_range = distances > self.threshold_min_range
        mask_max_range = distances < self.threshold_max_range

        # Combine masks and filter first
        final_mask = mask_beam_skip & mask_min_range & mask_max_range
        sensor_x = sensor_x[final_mask]
        sensor_y = sensor_y[final_mask]
        intensities = intensities[final_mask]
        rows_flat = rows_flat[final_mask]
        cols_flat = cols_flat[final_mask]

        # Now apply intensity selection mode
        if mode == 'all':
            valid_mask = intensities > 0#self.intensity_threshold
            rows_flat = rows_flat[valid_mask]
            cols_flat = cols_flat[valid_mask]
            sensor_x = sensor_x[valid_mask]
            sensor_y = sensor_y[valid_mask]
            intensities = intensities[valid_mask]


        elif mode == 'max_intensity':
            # Step 0: apply intensity threshold
            valid_mask = intensities > 0#self.intensity_threshold
            rows_flat = rows_flat[valid_mask]
            cols_flat = cols_flat[valid_mask]
            sensor_x = sensor_x[valid_mask]
            sensor_y = sensor_y[valid_mask]
            intensities = intensities[valid_mask]
            if len(intensities) == 0:
                # no valid points
                return np.empty((0, 3)), np.empty((0, 3))

            # Step 1: sort by columns
            sort_idx = np.argsort(cols_flat)
            sorted_cols = cols_flat[sort_idx]
            sorted_intensities = intensities[sort_idx]

            # Step 2: find boundaries for each column
            col_change = np.diff(sorted_cols, prepend=sorted_cols[0]-1)
            col_starts = np.flatnonzero(col_change)

            # Step 3: find max intensity per column using reduceat
            max_vals = np.maximum.reduceat(sorted_intensities, col_starts)
            # map back to original indices
            max_indices_in_sorted = []
            for start, val in zip(col_starts, max_vals):
                # search for the first occurrence of max in this column
                end = col_starts[col_starts > start][0] if np.any(col_starts > start) else len(sorted_intensities)
                local_idx = np.argmax(sorted_intensities[start:end])
                max_indices_in_sorted.append(start + local_idx)
            max_indices_in_sorted = np.array(max_indices_in_sorted)

            # Step 4: mask
            final_mask = np.zeros_like(intensities, dtype=bool)
            final_mask[sort_idx[max_indices_in_sorted]] = True

            # Apply mask
            rows_flat = rows_flat[final_mask]
            cols_flat = cols_flat[final_mask]
            sensor_x = sensor_x[final_mask]
            sensor_y = sensor_y[final_mask]
            intensities = intensities[final_mask]

        else:
            raise ValueError("Invalid mode. Choose 'all' or 'max_intensity'.")

        # Combine into arrays
        image_coordinates = np.column_stack((rows_flat, cols_flat, intensities))
        sensor_frame_coordinates = np.column_stack((sensor_x, sensor_y, intensities))

        return image_coordinates, sensor_frame_coordinates
    
    def save_as_pcd(self, msg):
        if not self.save_as_pcd_bool:
            return

        pc_data = msg.data
        point_step = msg.point_step
        fields = msg.fields

        field_names = [f.name for f in fields]
        if not all(f in field_names for f in ['x', 'y', 'z', 'intensity']):
            self.get_logger().warn("PointCloud2 missing fields x,y,z,intensity")
            return

        points = []
        for i in range(0, len(pc_data), point_step):
            point_bytes = pc_data[i:i + point_step]
            x, y, z, intensity = struct.unpack('ffff', point_bytes)
            points.append([x, y, z, intensity])

        np_points = np.array(points)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np_points[:, :3])

        intens = np_points[:, 3]
        colors = intens.reshape(-1, 1)
        colors = np.repeat(colors, 3, axis=1)
        colors = colors / np.max(colors)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        # Add to accumulated cloud
        self.accumulated_cloud.points.extend(cloud.points)
        self.accumulated_cloud.colors.extend(cloud.colors)

        # self.get_logger().info(f"Added {len(points)} points")

    def on_shutdown(self):
        # Called automatically on Ctrl+C
        # self.get_logger().info("Ctrl+C detected → saving PCD...")
        o3d.io.write_point_cloud(self.pcd_filename, self.accumulated_cloud)
        # self.get_logger().info(f"Saved PCD: {self.pcd_filename}")

def main():
    rclpy.init()
    node = FLS_PCL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()