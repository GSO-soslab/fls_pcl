#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from math import nan, sqrt

class FLS_PCL(Node):
    def __init__(self):
        super().__init__('fls_pcl_node')

        # Declare and read parameters
        self.declare_parameter('horizontal_beamwidth', Parameter.Type.INTEGER)
        self.declare_parameter('max_range',Parameter.Type.INTEGER)
        self.declare_parameter('intensity_threshold',Parameter.Type.INTEGER)
        self.declare_parameter('range_threshold',Parameter.Type.DOUBLE)
        self.declare_parameter('beam_skip_count',Parameter.Type.INTEGER)
        self.declare_parameter('frame_id',Parameter.Type.STRING)
        self.declare_parameter('sub_topic',Parameter.Type.STRING)


        self.horizontal_beamwidth = self.get_parameter('horizontal_beamwidth').value
        self.max_range = self.get_parameter('max_range').value
        self.intensity_threshold = self.get_parameter('intensity_threshold').value
        self.range_threshold = self.get_parameter('range_threshold').value
        self.beam_skip_count = self.get_parameter('beam_skip_count').value
        self.frame_id = self.get_parameter('frame_id').value
        sub_topic = self.get_parameter('sub_topic').value


        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.pub_pcl = self.create_publisher(PointCloud2, '/alpha_rise/fls/pointcloud/post', 10)
        self.pub_fls_edge_image = self.create_publisher(Image, '/alpha_rise/fls/data/image/edge/post', 10)

        # Subscriber
        self.create_subscription(Image,sub_topic,self.image_CB,10)


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
        
    def image_CB(self, msg):
        current = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Set the 12th and 11th columns from the end to 0
        current[:, -12] = 0
        current[:, -11] = 0
        
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns

        # === Convert all valid pixels to sensor frame coordinates ===
        edge_list, sensor_frame = self.extract_all_points_in_sensor_frame(current)
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

        # Apply beam skipping (every Nth pixel)
        indices = np.arange(len(sensor_frame))
        mask_beam_skip = (indices % self.beam_skip_count) == 0

        # Compute Euclidean range and apply range threshold
        distances = np.sqrt(sensor_x**2 + sensor_y**2)
        mask_range = distances > self.range_threshold

        # Combine all masks
        valid_mask = mask_beam_skip & mask_range

        # Filter valid points
        valid_x = sensor_x[valid_mask]
        valid_y = sensor_y[valid_mask]
        valid_i = intensity[valid_mask]

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

        # === Publish results ===
        self.pub_fls_edge_image.publish(self.bridge.cv2_to_imgmsg(edge_image, encoding="mono8"))
        self.pointcloud_msg.data = self.points.tobytes()
        self.pub_pcl.publish(self.pointcloud_msg)
    
    def extract_all_points_in_sensor_frame(self, image):
        """
        Convert every pixel in the image into sensor-frame coordinates (x, y, intensity).

        Parameters
        ----------
        image : np.ndarray
            2D array (rows × columns) representing intensity values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - image_coordinates: array of (row_index, column_index, intensity_value)
            - sensor_frame_coordinates: array of (x, y, intensity_value)
        """
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Precompute constants
        delta_theta = np.radians(self.horizontal_beamwidth) / self.n_beams
        base_theta = np.radians(-self.horizontal_beamwidth / 2)
        range_factor = self.max_range / self.n_bins

        # Create coordinate grids
        rows, cols = np.indices(image.shape)  # rows[i,j], cols[i,j]

        # Flatten for vectorized computation
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        intensities = image.flatten()

        # apply threshold
        valid_mask = intensities > self.intensity_threshold

        rows_flat = rows_flat[valid_mask]
        cols_flat = cols_flat[valid_mask]
        intensities = intensities[valid_mask]

        # Compute angles and ranges
        theta_values = base_theta + cols_flat * delta_theta
        r_values = range_factor * (self.n_bins - rows_flat)

        # Convert to sensor frame (x, y)
        sensor_x = r_values * np.cos(theta_values)
        sensor_y = r_values * np.sin(theta_values)

        # Combine into arrays
        image_coordinates = np.column_stack((rows_flat, cols_flat, intensities))
        sensor_frame_coordinates = np.column_stack((sensor_x, sensor_y, intensities))

        return image_coordinates, sensor_frame_coordinates
    
def main():
    rclpy.init()
    node = FLS_PCL()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()