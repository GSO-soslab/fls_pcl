#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
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
        
        # === TF2 ===
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === Declare and read parameters ===
        self.declare_parameter('sim',Parameter.Type.BOOL)
        self.sim = self.get_parameter('sim').value

        self.declare_parameter('max_depth',Parameter.Type.DOUBLE)
        self.max_depth = self.get_parameter('max_depth').value

        self.declare_parameter('min_depth',Parameter.Type.DOUBLE)
        self.min_depth = self.get_parameter('min_depth').value

        self.declare_parameter('threshold_intensity',Parameter.Type.INTEGER)
        self.intensity_threshold = self.get_parameter('threshold_intensity').value

        self.declare_parameter('threshold_min_range',Parameter.Type.DOUBLE)
        self.threshold_min_range = self.get_parameter('threshold_min_range').value

        self.declare_parameter('beam_skip_count',Parameter.Type.INTEGER)
        self.beam_skip_count = self.get_parameter('beam_skip_count').value

        self.declare_parameter('vertical_beamwidth', Parameter.Type.DOUBLE)
        self.vertical_beamwidth = self.get_parameter('vertical_beamwidth').value

        self.declare_parameter('frame_id',Parameter.Type.STRING)
        self.frame_id = self.get_parameter('frame_id').value

        self.declare_parameter('ping_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('marker_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('image_sub_topic',Parameter.Type.STRING)
        
        self.declare_parameter('pointcloud_pub_topic', Parameter.Type.STRING)
        pub_topic = self.get_parameter('pointcloud_pub_topic').value

        self.declare_parameter('save_as_pcd', Parameter.Type.BOOL)
        self.save_as_pcd_bool = self.get_parameter('save_as_pcd').value

        self.declare_parameter('pcd_filename', Parameter.Type.STRING)
        self.pcd_filename = self.get_parameter('pcd_filename').value

        self.declare_parameter('filter_mode', Parameter.Type.STRING)
        self.filter_mode = self.get_parameter('filter_mode').value

        if self.filter_mode == "probabilistic_intensity":
            self.declare_parameter('probabilistic_intensity.intensity.lower_bound_intensity', Parameter.Type.INTEGER)
            self.lower_bound_intensity = self.get_parameter('probabilistic_intensity.intensity.lower_bound_intensity').value

            self.declare_parameter('probabilistic_intensity.intensity.upper_bound_intensity', Parameter.Type.INTEGER)
            self.upper_bound_intensity = self.get_parameter('probabilistic_intensity.intensity.upper_bound_intensity').value

            self.declare_parameter('probabilistic_intensity.intensity.min_probability', Parameter.Type.DOUBLE)
            self.min_prob = self.get_parameter('probabilistic_intensity.intensity.min_probability').value

            self.declare_parameter('probabilistic_intensity.intensity.max_probability', Parameter.Type.DOUBLE)
            self.max_prob = self.get_parameter('probabilistic_intensity.intensity.max_probability').value

            self.declare_parameter('probabilistic_intensity.elevation.center_beam_probability', Parameter.Type.DOUBLE)
            self.elevation_beam_center_probability = self.get_parameter('probabilistic_intensity.elevation.center_beam_probability').value

            self.declare_parameter('probabilistic_intensity.elevation.edge_beam_probability', Parameter.Type.DOUBLE)
            self.elevation_beam_edge_probability = self.get_parameter('probabilistic_intensity.elevation.edge_beam_probability').value

            self.declare_parameter('probabilistic_intensity.sonar_params.frequency', Parameter.Type.DOUBLE)
            self.frequency = self.get_parameter('probabilistic_intensity.sonar_params.frequency').value

            self.declare_parameter('probabilistic_intensity.sonar_params.sound_speed', Parameter.Type.INTEGER)
            self.sound_speed = self.get_parameter('probabilistic_intensity.sonar_params.sound_speed').value

            self.declare_parameter('probabilistic_intensity.sonar_params.aperture_size', Parameter.Type.DOUBLE)
            self.aperture_size = self.get_parameter('probabilistic_intensity.sonar_params.aperture_size').value

            # === Sonar Physical Parameters ===
            wavelength = self.sound_speed / self.frequency
            k = 2 * np.pi / wavelength

            # === Elevation angles ===
            elevation_angles = np.arange(
                -self.vertical_beamwidth / 2,
                self.vertical_beamwidth / 2 + 1,
                1,
                dtype=np.float32
            )

            angles_rad = np.deg2rad(elevation_angles)

            # === Precompute trig ===
            self.cos_a = np.cos(angles_rad)[:, None]
            self.sin_a = np.sin(angles_rad)[:, None]

            # === Physical beam pattern (sinc) using SONAR beam directivity pattern ===
            temp = (k * self.aperture_size / 2) * np.sin(angles_rad)
            DI = np.ones_like(temp, dtype=np.float32)
            non_zero_mask = np.abs(temp) > 1e-10
            
            # === DI = sinc(kh/2*sin(elevation_angle)) ===
            DI[non_zero_mask] = np.sin(temp[non_zero_mask]) / temp[non_zero_mask]
            self.beam_probs = np.clip(DI, self.elevation_beam_edge_probability, self.elevation_beam_center_probability)

        # === CV bridge ===
        self.bridge = CvBridge()

        # === Persistent accumulated cloud ===
        self.accumulated_cloud = o3d.geometry.PointCloud()

        # === Triggers ===
        self.receive_ping = False
        self.receive_marker = False
        self.bool_create_sonar_geometry = False

        # === Publishers ===
        self.pub_pcl = self.create_publisher(PointCloud2, pub_topic, 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/median', 10)

        # === Subscribers === 
        self.sub_image = self.create_subscription(Image, self.get_parameter('image_sub_topic').value, self.image_CB,10)
        self.sub_pcl = self.create_subscription(PointCloud2, pub_topic, self.pointcloud_CB, 10)
        
        # === Sub to Ping topic else, get sim parameters ===
        if not self.sim:
            self.sub_ping = self.create_subscription(Ping, self.get_parameter('ping_sub_topic').value, self.ping_CB, 10)
        else:
            self.max_range = 40.0
            self.horizontal_beamwidth = 70
            bearings = np.loadtxt('bearings.txt')
            self.bearings = np.array([np.radians(bearing * 0.01) for bearing in bearings]).squeeze()

        # === Sub to Marker topic for Voxels ===
        self.sub_marker = self.create_subscription(Marker, self.get_parameter('marker_sub_topic').value, self.marker_CB, 10)

        # === Initialize PointCloud2 message ===
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

        # === Register shutdown handler ===
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

            current = self.image_preprocess(current)

            # === Convert all valid pixels to sensor frame coordinates ===
            edge_list, sensor_frame = self.extract_points_in_sensor_frame(current, mode=self.filter_mode)
            
            sensor_x = sensor_frame[:, 0].astype(np.float32)
            sensor_y = sensor_frame[:, 1].astype(np.float32)
            raw_intensity = sensor_frame[:, 2].astype(np.float32)
            
            if self.filter_mode == 'probabilistic_intensity':
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

                # === Joint probability ===
                # P = P_point * P_beam
                final_probs = raw_intensity[None, :] * self.beam_probs[:, None]
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
                # Filter points by probability > 0.5
                prob_mask = filtered_probs[indices] > 0.0
                indices = indices[prob_mask]
                
                num_points = indices.shape[0]

                self.pointcloud_msg.width = num_points
                self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

                self.points = np.full((num_points, len(self.fields)), np.nan, dtype=np.float32)
                # Position
                self.points[:, 0:3] = filtered_points[indices]
                # Probability
                self.points[:, 3] = filtered_probs[indices]
                
            else:
                # Use points as-is from sensor_frame
                z = np.zeros_like(sensor_x)
                all_points = np.stack((sensor_x, sensor_y, z), axis=-1)

                filtered_points = all_points
                filtered_intensity = raw_intensity

                num_points = filtered_points.shape[0]

                self.pointcloud_msg.width = num_points
                self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

                self.points = np.full((num_points, len(self.fields)), np.nan, dtype=np.float32)

                # Position
                self.points[:, 0:3] = filtered_points

                # Raw intensity
                self.points[:, 3] = filtered_intensity

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
                invalid_mask = (z > self.max_depth) #| (z < self.min_depth)

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
    
    def image_preprocess(self, current):
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns

        # Lee filter. Better for multiplicative noise.
        # current = self.lee_filter(current, kernel_size=5)

        # Zero out the middle 10 columns
        h, w = current.shape
        mid = w // 2
        # current[:, mid - 15 : mid] = 0

        # Parameters
        top_width = 20
        bottom_width = 15
        bottom_offset = 20  # pixels left of center at bottom

        # Row coordinates [0 .. h-1]
        y = np.arange(h, dtype=np.float32)
        frac = y / (h - 1)

        # Width tapers from 20 → 1
        widths = np.round(top_width + frac * (bottom_width - top_width)).astype(int)

        # Left boundary moves so the wedge ends at (mid - bottom_offset)
        top_left = mid - top_width
        bottom_left = mid - bottom_offset

        left_bounds = np.round(
            top_left + frac * (bottom_left - top_left)
        ).astype(int)

        # Column coordinates
        x = np.arange(w)

        # Build mask: True where pixels should be zeroed
        mask = (x[None, :] >= left_bounds[:, None]) & \
            (x[None, :] < (left_bounds + widths)[:, None])

        # Apply mask
        current[mask] = 0

        # Median Filtering, Higher ksize, stronger smoothening, higher comp
        # current = cv2.medianBlur(current, ksize=11)
        
        self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(current, encoding="mono8"))
        
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

            # Start with all values at min_prob
            probabilities = np.full_like(intensities, 0.1, dtype=float)

            # Mask for linear interpolation range
            mid_mask = (intensities > self.lower_bound_intensity) & (intensities < self.upper_bound_intensity)

            # Linear interpolation between 0.1 and 0.9
            probabilities[mid_mask] = self.min_prob + (
                (intensities[mid_mask] - self.lower_bound_intensity) /
                (self.upper_bound_intensity - self.lower_bound_intensity)
            ) * (self.max_prob - self.min_prob)

            # Anything above upper threshold → max_prob
            probabilities[intensities >= self.upper_bound_intensity] = 0.9

            # Optional: round to nearest 0.1
            probabilities = np.round(probabilities * 10) / 10
            intensities = probabilities

        elif mode == 'max_intensity':
            # Step 0: apply minimum range threshold
            ranges = np.sqrt(sensor_x**2 + sensor_y**2)
            valid_mask = ranges > self.threshold_min_range
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

            # Step 5: remove points with low intensity
            intensity_mask = intensities >= self.intensity_threshold

            rows_flat = rows_flat[intensity_mask]
            cols_flat = cols_flat[intensity_mask]
            sensor_x = sensor_x[intensity_mask]
            sensor_y = sensor_y[intensity_mask]
            intensities = intensities[intensity_mask]

            if len(intensities) == 0:
                return np.empty((0, 3)), np.empty((0, 3))
            
        elif mode == 'threshold_intensity':
            ranges = np.sqrt(sensor_x**2 + sensor_y**2)
            valid_mask = ranges > self.threshold_min_range
            rows_flat = rows_flat[valid_mask]
            cols_flat = cols_flat[valid_mask]
            sensor_x = sensor_x[valid_mask]
            sensor_y = sensor_y[valid_mask]
            intensities = intensities[valid_mask]

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
        if self.save_as_pcd_bool:  
            transform = self.tf_buffer.lookup_transform(
                'alpha_rise/world',
                self.frame_id,
                rclpy.time.Time()
                )

            # Transform to world frame
            msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(msg, transform)   
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

    def on_shutdown(self):
        self.get_logger().info(f"Shutting down → saving PCD: {self.pcd_filename}")
        
        if not self.accumulated_cloud.has_points():
            print("⚠ Warning: Point cloud is empty")
            return
        
        # Clean the point cloud before saving
        points = np.asarray(self.accumulated_cloud.points)
        
        # Remove NaN and Inf values
        valid_mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
        valid_points = points[valid_mask]
        
        if len(valid_points) == 0:
            print("⚠ Warning: No valid points after filtering")
            return
        
        # Create clean point cloud
        clean_pcd = o3d.geometry.PointCloud()
        clean_pcd.points = o3d.utility.Vector3dVector(valid_points)
        
        # Copy colors/normals if they exist
        if self.accumulated_cloud.has_colors():
            colors = np.asarray(self.accumulated_cloud.colors)[valid_mask]
            clean_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        try:
            o3d.io.write_point_cloud(self.pcd_filename, clean_pcd)
            print(f"✓ Saved {len(valid_points)} valid points to {self.pcd_filename}")
        except Exception as e:
            print(f"✗ Failed to save: {e}")

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