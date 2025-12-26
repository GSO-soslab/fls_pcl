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
from scipy.ndimage import uniform_filter
from math import nan
from oculus_interfaces.msg import Ping
from visualization_msgs.msg import Marker
from scipy.spatial import cKDTree

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
        self.pub_fls_edge_image = self.create_publisher(Image, pub_topic+'/image/edge', 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/median', 10)

        # Subscribers
        self.create_subscription(Image, self.get_parameter('image_sub_topic').value, self.image_CB,10)
        self.create_subscription(PointCloud2, pub_topic,self.pointcloud_CB, 10)
        self.create_subscription(Ping, self.get_parameter('ping_sub_topic').value, self.ping_CB, 10)
        self.create_subscription(Marker, self.get_parameter('marker_sub_topic').value, self.marker_CB, 10)

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
            # np.savetxt("voxel_centroids.txt", self.voxel_centroids, fmt="%.6f", delimiter=" ", header="x y z", comments="")
            self.receive_marker = True

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

            if not self.bool_create_sonar_geometry:
                # Create 3D points in SONAR frame where each pixel can occupy. Here elevation is assumed to be 0.
                self.valid_SONAR_points = self.create_valid_SONAR_points(min_range=0.1, max_range=self.max_range, n_bins=self.n_bins)

                # Extract occupancy points from SONAR frame which form correspondence with voxel centroids. 
                # self.voxel_corresponding_points = self.create_voxel_corresponding_points(self.voxel_centroids,self.valid_SONAR_points) 
                self.bool_create_sonar_geometry = True
            
            # Lee filter. Better for multiplicative noise.
            # current = self.lee_filter(current, kernel_size=15)

            # Median Filtering, Higher ksize, stronger smoothening, higher comp
            current = cv2.medianBlur(current, ksize=11)
            self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(current, encoding="mono8"))

            # === Convert all valid pixels to sensor frame coordinates ===
            edge_list, sensor_frame = self.extract_points_in_sensor_frame(current, mode=self.filter_mode)

            pointcloud_image = np.zeros((rows, columns), dtype=np.float32)

            if len(edge_list) == 0:
                self.get_logger().warn("No measurements", throttle_duration_sec=3)
                self.points = np.empty((0, len(self.fields)), dtype=np.float32)
                return
            
            # === Extract sensor frame data ===
            sensor_x = sensor_frame[:, 0]
            sensor_y = sensor_frame[:, 1]
            point_prob = sensor_frame[:, 2]   # already a probability in [0,1]

            # === Base points (center beam geometry) ===
            points = np.zeros((len(sensor_frame), 3), dtype=np.float32)
            points[:, 0] = sensor_x
            points[:, 1] = sensor_y
            points[:, 2] = 0.0  # Z = 0 for 2D sensor

            # === Elevation angles ===
            elevation_angles = np.arange(-self.vertical_beamwidth/2, self.vertical_beamwidth/2 + 1, 1)   # -6 … 0 … +6

            # === Gaussian beam probabilities (swath membership) ===
            ### POWER LAW SWITCH####
            min_prob = 0.5
            max_prob = 0.9
            sigma = 3.0

            beam_probs = min_prob + (max_prob - min_prob) * np.exp(
                -0.5 * (elevation_angles / sigma) ** 2
            )

            # === Generate beams with UNION probability ===
            all_points = []
            all_probs = []

            for angle, beam_p in zip(elevation_angles, beam_probs):
                rotated_pts = self.rotate_points_y(points, angle)

                # Union of independent probabilities:
                # P = 1 - (1 - P_point)(1 - P_beam)
                final_prob = 1.0 - (1.0 - point_prob) * (1.0 - beam_p)

                all_points.append(rotated_pts)
                all_probs.append(final_prob)

            # === Stack results ===
            all_points = np.vstack(all_points)      # (N * num_beams, 3)
            all_probs = np.hstack(all_probs)        # (N * num_beams,)

            # Range Filtering
            ranges_xy = np.hypot(all_points[:, 0], all_points[:, 1])
            mask = ranges_xy >= self.threshold_min_range
            filtered_points = all_points[mask]
            filtered_probs = all_probs[mask]

            # Ensure N(points) = N(voxels)
            indices, _ = self.create_voxel_corresponding_points(self.voxel_centroids, filtered_points)
            num_points = filtered_points[indices].shape[0]

            self.pointcloud_msg.width = num_points
            self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

            self.points = np.full((num_points, len(self.fields)), np.nan, dtype=np.float32)
            # Position
            self.points[:, 0:3] = filtered_points[indices]
            # Probability
            self.points[:, 3] = filtered_probs[indices]

            # === Visualization image (unchanged) ===
            rows_i = edge_list[:, 0].astype(int)
            cols_i = edge_list[:, 1].astype(int)
            intensities_i = edge_list[:, 2]

            pointcloud_image[rows_i, cols_i] = intensities_i
            pointcloud_image = np.clip(pointcloud_image, 0, 255).astype(np.uint8)

            # === Publish ===
            self.pub_fls_edge_image.publish(
                self.bridge.cv2_to_imgmsg(pointcloud_image, encoding="mono8")
            )
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
                points = list(pc2.read_points(
                    self.pointcloud_msg,
                    field_names=('x', 'y', 'z', 'intensity'),
                    skip_nans=False
                ))

                # Modify points: set points with z > -1.0 to zero
                new_points = []
                for x, y, z, intensity in points:
                    if z > self.max_depth:
                        x, y, z, intensity = nan, nan, nan, 0.5
                    elif z< self.min_depth:
                        x, y, z, intensity = nan, nan, nan, 0.5
                    new_points.append([x, y, z, intensity])

                # Create new PointCloud2 preserving intensity
                self.pointcloud_msg = pc2.create_cloud(self.pointcloud_msg.header, self.fields, new_points)
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

        # === Rotation about Y axis (elevation) ===
    def rotate_points_y(self, points, angle_deg):
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(
            [[ c, 0,  s],
            [ 0, 1,  0],
            [-s, 0,  c]],
            dtype=np.float32
        )
        return points @ R.T      

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
        sensor_x = r_values * np.cos(theta_values)
        sensor_y = r_values * np.sin(theta_values)

        # Apply beam skipping
        indices = np.arange(len(sensor_x))
        mask_beam_skip = (indices % self.beam_skip_count) == 0
    
        if mode == 'probabilistic_intensity':

            # Build KD-tree from voxel points
            voxel_points_xy = self.voxel_centroids[:, :2]  # (N, 2)

            # Stack sensor points
            sensor_xy = np.column_stack((sensor_x, sensor_y))  # (M, 2)
            
            # Extract occupancy points from SONAR frame which form correspondence with voxel centroids.
            sensor_indices, distance = self.create_voxel_corresponding_points(voxel_points_xy, sensor_xy)

            # Get intensities corresponding to the matched points
            intensities = intensities[sensor_indices]  # (N,)
            # Also get original sensor info corresponding to each voxel
            rows_flat   = rows_flat[sensor_indices]   # (N,)
            cols_flat   = cols_flat[sensor_indices]   # (N,)
            sensor_x    = sensor_x[sensor_indices]    # (N,)
            sensor_y    = sensor_y[sensor_indices]    # (N,)

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
        Stores the msg as a .pcl file
        '''
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