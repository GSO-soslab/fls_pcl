#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
import math
import cv2
import open3d as o3d
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_sensor_msgs.tf2_sensor_msgs
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformException
import struct
from scipy.ndimage import uniform_filter
from math import nan
from oculus_interfaces.msg import Ping
from visualization_msgs.msg import Marker
from scipy.spatial import cKDTree
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class FLS_PCL(Node):
    """
    ROS2 Node that converts Polar Image from FLS to Probability Clouds
    
    :var Publishers: PointCloud2 that indicates probabilities
    :var Subscribers: Image: Polar Image from FLS <br> Marker: Sonar FOV Voxels <br> Ping: Oculus Ping
    """
    def __init__(self):
        super().__init__('fls_pcl_node')
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Declare and read parameters
        self.declare_parameter('max_depth',Parameter.Type.DOUBLE)
        self.declare_parameter('min_depth',Parameter.Type.DOUBLE)

        self.declare_parameter('threshold_intensity',Parameter.Type.INTEGER)
        self.declare_parameter('threshold_min_range',Parameter.Type.DOUBLE)

        self.declare_parameter('beam_skip_count',Parameter.Type.INTEGER)
        self.declare_parameter('vertical_beamwidth', Parameter.Type.DOUBLE)
        self.declare_parameter('frame_id',Parameter.Type.STRING)
        self.declare_parameter('ping_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('marker_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('image_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('pointcloud_pub_topic', Parameter.Type.STRING)
        self.declare_parameter('save_as_pcd', Parameter.Type.BOOL)
        self.declare_parameter('pcd_filename', Parameter.Type.STRING)
        self.declare_parameter('filter_mode', Parameter.Type.STRING)

        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.intensity_threshold = self.get_parameter('threshold_intensity').value
        self.threshold_min_range = self.get_parameter('threshold_min_range').value
        self.beam_skip_count = self.get_parameter('beam_skip_count').value
        self.vertical_beamwidth = self.get_parameter('vertical_beamwidth').value

        self.frame_id = self.get_parameter('frame_id').value
        pub_topic = self.get_parameter('pointcloud_pub_topic').value
        self.save_as_pcd_bool = self.get_parameter('save_as_pcd').value
        self.pcd_filename = self.get_parameter('pcd_filename').value
        self.filter_mode = self.get_parameter('filter_mode').value

        # CV bridge
        self.bridge = CvBridge()

        # Persistent accumulated cloud
        self.accumulated_cloud = o3d.geometry.PointCloud()

        # Triggers
        self.receive_ping = False
        self.receive_marker = False
        self.bool_create_sonar_geometry = False

        # Publishers
        self.pub_pcl = self.create_publisher(PointCloud2, pub_topic, 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/median', 10)

        # Subscribers
        self.sub_image = self.create_subscription(Image, self.get_parameter('image_sub_topic').value, self.image_CB,10)
        self.sub_ping = self.create_subscription(Ping, self.get_parameter('ping_sub_topic').value, self.ping_CB, 10)
        self.sub_marker = self.create_subscription(Marker, self.get_parameter('marker_sub_topic').value, self.marker_CB, 10)

        # Populate PointCloud2 message
        self.pointcloud_msg = PointCloud2()
        h = Header()
        h.frame_id = self.frame_id
        self.pointcloud_msg.header = h
        self.pointcloud_msg.height = 1  # unorganized cloud
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pointcloud_msg.fields = self.fields
        self.pointcloud_msg.point_step = 4 * (len(self.fields))  # Each point occupies 16 bytes
        self.pointcloud_msg.is_dense = True  # All points are valid

        # === Elevation angles ===
        elevation_angles = np.arange(
            -self.vertical_beamwidth / 2,
            self.vertical_beamwidth / 2 + 1,
            1,
            dtype=np.float32
        )  # (B,)

        angles_rad = np.deg2rad(elevation_angles)

        # === Gaussian beam probabilities ===
        min_prob = 0.5
        max_prob = 0.9
        sigma = 3.0

        self.beam_probs = min_prob + (max_prob - min_prob) * np.exp(
            -0.5 * (elevation_angles / sigma) ** 2
        ).astype(np.float32)  # (B,)

        # === Precompute trig ===
        self.cos_a = np.cos(angles_rad)[:, None]  # (B, 1)
        self.sin_a = np.sin(angles_rad)[:, None]  # (B, 1)
        
        # Register shutdown handler
        rclpy.get_default_context().on_shutdown(self.on_shutdown)
    
    def marker_CB(self, msg:Marker):
        '''
        Callback for Marker msg. 
        Stores the voxel centroids in sensor frame.

        Returns: 
            self.voxel_centroids as numpy array
        '''
        if not self.receive_marker:
            self.voxel_centroids = np.array([(p.x, p.y, p.z) for p in msg.points],dtype=np.float64)
            self.marker_resolution = msg.scale.x
            # np.savetxt("voxel_centroids.txt", self.voxel_centroids, fmt="%.6f", delimiter=" ", header="x y z", comments="")
            self.receive_marker = True
        self.destroy_subscription(self.sub_marker)
        self.sub_marker = None
        
    def ping_CB(self,msg:Ping):
        '''
        Callback for Ping msg.
        Stores information about the SONAR. All range & azimuth are sensor frame.

        Returns:
            self.max_range<br>
            self.bearings (azimuth of each beam in radians) as np.array<br>
        '''
        if not self.receive_ping:
            self.bearings = np.array([np.radians(bearings * 0.01) for bearings in msg.bearings]).squeeze()
            # np.savetxt("bearings.txt", self.bearings, fmt="%.8f")
            self.max_range = msg.range

            # High / Low frequency mode
            if msg.master_mode == 2:
                self.horizontal_beamwidth = 70 
            elif msg.master_mode == 1:
                self.horizontal_beamwidth = 130
        
            self.receive_ping = True
        self.destroy_subscription(self.sub_ping)
        self.sub_ping = None

    def image_CB(self, msg:Image):
        '''
        Callback for Image msg.<br>
        Uses the polar image from the SONAR to extract intensities and project it as probability clouds
        '''
        # Only start when both Ping & Marker msgs are in memory.
        if self.receive_ping and self.receive_marker:
            current = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            rows, columns = current.shape
            self.n_bins, self.n_beams = rows, columns

            # Lee filter. Better for multiplicative noise.
            # current = self.lee_filter(current, kernel_size=15)

            # Median Filtering, Higher ksize, stronger smoothening, higher comp
            # current = cv2.medianBlur(current, ksize=11)
            # self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(current, encoding="mono8"))

            # === Convert all valid pixels to sensor frame coordinates ===
            edge_list, sensor_frame = self.extract_points_in_sensor_frame(current, mode=self.filter_mode)
            
            sensor_x = sensor_frame[:, 0].astype(np.float32)
            sensor_y = sensor_frame[:, 1].astype(np.float32)
            point_prob = sensor_frame[:, 2].astype(np.float32)

            # Direct broadcast-ready arrays
            x = sensor_x[None, :]   # (1, N)
            y = sensor_y[None, :]
            z = np.zeros_like(sensor_x)[None, :]

            # === Rotate around Y (broadcasted) ===
            x_r = x * self.cos_a + z * self.sin_a
            y_r = y * np.ones_like(x_r)  # (B, N)
            z_r = -x * self.sin_a + z * self.cos_a

            # === Stack rotated points ===
            rotated_points = np.stack((x_r, y_r, z_r), axis=-1)
            # shape: (B, N, 3)

            # === Union probability ===
            # P = 1 - (1 - P_point)(1 - P_beam)
            # final_probs = 1.0 - (1.0 - point_prob[None, :]) * (1.0 - self.beam_probs[:, None])
            # === Joint probability ===
            # P = P_point * P_beam
            final_probs = point_prob[None, :] * self.beam_probs[:, None]
            # shape: (B, N)

            # === Flatten to match your original output ===
            all_points = rotated_points.reshape(-1, 3)   # (B*N, 3)
            all_probs  = final_probs.reshape(-1)          # (B*N,)

            # Range Filtering
            ranges_xy = np.hypot(all_points[:, 0], all_points[:, 1])
            mask = ranges_xy >= self.threshold_min_range
            filtered_points = all_points[mask]
            filtered_probs = all_probs[mask]

            # Ensure N(points) = N(voxels)
            indices, _ = self.create_voxel_corresponding_points(self.voxel_centroids, filtered_points)
            # print(f"indices:{indices.shape}, Voxels:{self.voxel_centroids.shape}, Filtered_points:{filtered_points.shape}", flush=True)
            num_points = filtered_points[indices].shape[0]

            self.pointcloud_msg.width = num_points
            self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

            self.points = np.full((num_points, len(self.fields)), np.nan, dtype=np.float32)
            # Position
            self.points[:, 0:3] = filtered_points[indices]
            # Probability
            self.points[:, 3] = filtered_probs[indices]

            self.pointcloud_msg.data = self.points.tobytes()

            # Depth filtering
            try:
                transform = self.tf_buffer.lookup_transform(
                    'alpha_rise/world',
                    self.frame_id,
                    rclpy.time.Time()
                )

                # Apply transform
                self.pointcloud_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(self.pointcloud_msg, transform)

                # Read points including intensity
                points_struct = pc2.read_points(
                    self.pointcloud_msg,
                    field_names=('x', 'y', 'z', 'intensity'),
                    skip_nans=False
                )

                # Convert structured → normal ndarray (N,4)
                points = np.column_stack((
                    points_struct['x'],
                    points_struct['y'],
                    points_struct['z'],
                    points_struct['intensity']
                )).astype(np.float32)

                z = points[:, 2]

                # Mask for points OUTSIDE depth range
                invalid_mask = (z > self.max_depth) | (z < self.min_depth)

                # Set xyz to NaN
                points[invalid_mask, 0:3] = np.nan

                # Set intensity to nan
                points[invalid_mask, 3] = np.nan

                points = [tuple(p) for p in points]

                # Create new PointCloud2 preserving intensity
                self.pointcloud_msg = pc2.create_cloud(self.pointcloud_msg.header, self.fields, points)
                try:
                    transform = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    'alpha_rise/world',
                    rclpy.time.Time()
                    )

                    # Transform back to sensor frame
                    self.pointcloud_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(self.pointcloud_msg, transform)

                    self.pointcloud_msg.is_dense = True
                    self.pub_pcl.publish(self.pointcloud_msg)

                except TransformException as e:
                    self.get_logger().warn(f'Transform not available: {e}')
            except TransformException as e:
                self.get_logger().warn(f'Transform not available: {e}')     

    def create_voxel_corresponding_points(self, voxel_points:np.ndarray, geometry_points:np.ndarray):
        '''
        Finding the closest geometry_points corresponding to the voxel_points.
        
        :param geometry_points: List of geometry centroids (x,y,z)
        :param voxel_points: List of voxel centroids (x,y,z)

        :return list of geometry_points (x,y,z) that is closest correspondence with voxel_points. Same size as of voxel_points 
        '''
        voxel_points = np.asarray(voxel_points, dtype=np.float32)
        geometry_points = np.asarray(geometry_points, dtype=np.float32)

        # Build KD-tree on geometry_points.
        tree = cKDTree(geometry_points)

        # Query nearest neighbor for each voxel
        distance, indices = tree.query(voxel_points, k=1)

        return indices, distance
        

    def create_valid_SONAR_points(self, min_range:float, max_range:float, n_bins:int):
        """
        Generate valid SONAR return points in 3D (x,y,z=0).

        :param min_range: minimum range of the SONAR
        :param max_range: maximum range of the SONAR
        :param n_bins: number of range bins

        :return: (n_bins * n_beams, 3) array of (x, y, 3) points. Ex. (517*512,3)
        """
        # Range bins
        ranges = np.linspace(min_range, max_range, n_bins)

        azimuth_angles = self.bearings

        # Meshgrid (range × azimuth)
        R, AZ = np.meshgrid(ranges, azimuth_angles, indexing="ij")

        # Convert to Cartesian
        X = R * np.cos(AZ)
        Y = R * np.sin(AZ)
        # Elevation is 0
        Z = np.zeros_like(X)

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        return points
    
    def extract_points_in_sensor_frame(self, image:np.ndarray, mode='threshold_intensity'):
        """
        Convert pixels in the image into sensor-frame coordinates (x, y, intensity),
        optionally filtering by beam skipping and range threshold first, then applying
        intensity selection mode.

        Parameters
        ----------
        image : np.ndarray
            2D array (rows × columns) representing intensity values.
        mode : str, optional
            'threshold_intensity' - keep all points above threshold
            'max_intensity' - keep only the highest-intensity point per column
            'probabilistic_intensity' - convert pixel intensities into probablities.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - image_coordinates: array of (row_index, column_index, intensity_value)
            - sensor_frame_coordinates: array of (x, y, intensity_value)
        """
        if image is None:
            raise ValueError("Image not found or unable to load.")
        
        """
            |<----->| number of beams = 512
            ------>x   -
            |          ^
            |          | 517 beams
            |          v 
         y  v          -      <-------- Sensor Origin
        """
        #40m / 517 beams = 0.07m/beams
        meters_per_beam = self.max_range / self.n_bins 

        # Create coordinate grids
        rows, cols = np.indices(image.shape)
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        intensities = image.flatten()

        theta_values = self.bearings[cols_flat]
        r_values = meters_per_beam * (self.n_bins - rows_flat)

        # Convert to sensor frame (x, y)
        sensor_x = r_values * np.cos(theta_values) #len(sensor_x) = len(sensor_y) = 512*517 = 264704
        sensor_y = r_values * np.sin(theta_values)
        # print(len(sensor_x), flush=True)

        # Apply beam skipping
        indices = np.arange(len(sensor_x))
        mask_beam_skip = (indices % self.beam_skip_count) == 0
    
        if mode == 'probabilistic_intensity':
            
            if not self.bool_create_sonar_geometry:
                # Build KD-tree from voxel points
                z = self.voxel_centroids[:, 2]

                # Mask for voxels on sensor plane
                z0_mask = np.isclose(z, 0.0)   # safer than z == 0 for floats

                # Keep only z = 0 voxels
                voxel_centroids_z0 = self.voxel_centroids[z0_mask]   # (7646, 3)

                # Use only XY for correspondence
                voxel_points_xy = voxel_centroids_z0[:, :2]          # (1160, 2)
                
                # Stack sensor points
                sensor_xy = np.column_stack((sensor_x, sensor_y))  # sensor_xy.shape: (264706, 2)
                
                # Extract occupancy points from SONAR frame which form correspondence with voxel centroids.
                self.sensor_indices, distance = self.create_voxel_corresponding_points(voxel_points_xy, sensor_xy) #self.sensor_indices: (1160,)
                
                self.bool_create_sonar_geometry = True

            # Get intensities corresponding to the matched points
            intensities = intensities[self.sensor_indices]  # (N,)
            # Also get original sensor info corresponding to each voxel
            rows_flat   = rows_flat[self.sensor_indices]   # (N,)
            cols_flat   = cols_flat[self.sensor_indices]   # (N,)
            sensor_x    = sensor_x[self.sensor_indices]    # (N,)
            sensor_y    = sensor_y[self.sensor_indices]    # (N,)

            # --- intensity -> probability mapping ---
            min_intensity = self.intensity_threshold
            max_intensity = 255.0
            min_prob = 0.5
            max_prob = 0.9

            # Create array of all 0.1 size of intensities.
            probabilities = np.full_like(intensities, 0.1, dtype=float)

            # Mask that determines if intensity > threshold
            mask = intensities > self.intensity_threshold
            
            # Convert all entries which return true to a scaled probability from min_prob to max_prob. 
            # If intensity > threshold, map it to (0.5,0.9), else 0.1
            probabilities[mask] = min_prob + (
                (intensities[mask] - min_intensity) /
                (max_intensity - min_intensity)
            ) * (max_prob - min_prob)

            probabilities = np.clip(probabilities, min_prob, max_prob)
            intensities = probabilities

        elif mode == 'max_intensity':
            # Step 0: apply intensity threshold
            valid_mask = intensities > self.intensity_threshold
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
        
        elif mode == 'threshold_intensity':
            # Only keep intensities greater than threshold
            valid_mask = intensities > self.intensity_threshold
            rows_flat = rows_flat[valid_mask]
            cols_flat = cols_flat[valid_mask]
            sensor_x = sensor_x[valid_mask]
            sensor_y = sensor_y[valid_mask]
            intensities = intensities[valid_mask]
        
        else:
            raise ValueError("Invalid mode")

        # Combine into arrays
        image_coordinates = np.column_stack((rows_flat, cols_flat, intensities))
        sensor_frame_coordinates = np.column_stack((sensor_x, sensor_y, intensities))

        return image_coordinates, sensor_frame_coordinates
    
    def lee_filter(self, image, kernel_size=5):
        """
        Lee filter for speckle noise reduction.
        
        Parameters:
            image (np.ndarray): Grayscale image (uint8 or float)
            kernel_size (int): Odd window size
        
        Returns:
            np.ndarray: Filtered image
        """
        image = image.astype(np.float32)

        # Local mean
        local_mean = uniform_filter(image, kernel_size)

        # Local variance
        local_mean_sq = uniform_filter(image**2, kernel_size)
        local_var = local_mean_sq - local_mean**2

        # Estimate noise variance (global)
        noise_var = np.mean(local_var)

        # Lee filter
        weight = local_var / (local_var + noise_var)
        filtered = local_mean + weight * (image - local_mean)

        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def pointcloud_CB(self, msg):
        '''
        Callback for PointCloud2 msg.<br>
        Publish the resultant probabilty cloud.
        Stores the msg as a .pcl file
        '''
        marker = Marker()
        marker.header = msg.header
        marker.ns = "cloud"
        marker.id = 0
        marker.type = Marker.CUBE_LIST      # 🔹 changed
        marker.action = Marker.ADD

        # Cube size
        marker.scale.x = self.marker_resolution
        marker.scale.y = self.marker_resolution
        marker.scale.z = self.marker_resolution

        for x, y, z, intensity in point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True):

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if not math.isfinite(intensity):
                continue
            x = (x + 0.5) * self.marker_resolution
            y = (y + 0.5) * self.marker_resolution
            z = (z + 0.5) * self.marker_resolution

            # Assign explicitly to Point
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            marker.points.append(p)


            # Intensity → color (per cube)
            i = max(0.0, min(1.0, intensity))

            c = ColorRGBA()
            c.r = float(i)
            c.g = float(1.0 - abs(i - 0.5) * 2.0)
            c.b = float(1.0 - i)
            c.a = float(1.0)

            marker.colors.append(c)   # 🔹 required for CUBE_LIST
        # print(len(marker.points), flush=True)
        self.probability_marker_pub.publish(marker)
        
        if self.save_as_pcd_bool:     
            pc_data = msg.data
            point_step = msg.point_step
            fields = msg.fields

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