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
from oculus_interfaces.msg import Ping
from visualization_msgs.msg import Marker
from scipy.spatial import cKDTree
import cv2

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

        self.declare_parameter('vertical_beamwidth', Parameter.Type.DOUBLE)
        self.vertical_beamwidth = self.get_parameter('vertical_beamwidth').value

        self.declare_parameter('sensor_frame_id', Parameter.Type.STRING)
        self.frame_id = self.get_parameter('sensor_frame_id').value

        self.declare_parameter('world_frame_id', Parameter.Type.STRING)
        self.world_frame_id = self.get_parameter('world_frame_id').value

        self.declare_parameter('ping_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('marker_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('image_sub_topic',Parameter.Type.STRING)
        
        self.declare_parameter('pointcloud_pub_topic', Parameter.Type.STRING)
        pub_topic = self.get_parameter('pointcloud_pub_topic').value

        self.declare_parameter('save_as_pcd', Parameter.Type.BOOL)
        self.save_as_pcd_bool = self.get_parameter('save_as_pcd').value

        self.declare_parameter('pcd_filename', Parameter.Type.STRING)
        self.pcd_filename = self.get_parameter('pcd_filename').value

        self.declare_parameter('probabilistic_intensity.intensity.lower_bound_intensity', Parameter.Type.INTEGER)
        self.lower_bound_intensity = self.get_parameter('probabilistic_intensity.intensity.lower_bound_intensity').value

        self.declare_parameter('probabilistic_intensity.intensity.upper_bound_intensity', Parameter.Type.INTEGER)
        self.upper_bound_intensity = self.get_parameter('probabilistic_intensity.intensity.upper_bound_intensity').value

        self.declare_parameter('probabilistic_intensity.intensity.min_probability', Parameter.Type.DOUBLE)
        self.min_prob = self.get_parameter('probabilistic_intensity.intensity.min_probability').value

        self.declare_parameter('probabilistic_intensity.intensity.max_probability', Parameter.Type.DOUBLE)
        self.max_prob = self.get_parameter('probabilistic_intensity.intensity.max_probability').value

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
        # === round to 2 precision ===
        DI[non_zero_mask] = np.sin(temp[non_zero_mask]) / temp[non_zero_mask]
        self.beam_probs = np.array([round(theta_prob, 2) for theta_prob in DI], dtype=np.float32)

        # === CV bridge ===
        self.bridge = CvBridge()

        # === Persistent accumulated cloud ===
        self.accumulated_cloud = o3d.geometry.PointCloud()

        # === Triggers ===
        self.receive_ping = False
        self.receive_marker = False
        self.bool_create_sonar_geometry = False

        # === Publishers ===
        self.pub_pcl      = self.create_publisher(PointCloud2, pub_topic, 10)
        self.pub_pcl_prob = self.create_publisher(PointCloud2, pub_topic + '/probability', 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/filtered', 10)

        # === Subscribers === 
        self.sub_image = self.create_subscription(Image, self.get_parameter('image_sub_topic').value, self.image_CB,10)
        self.sub_pcl = self.create_subscription(PointCloud2, pub_topic, self.pointcloud_CB, 10)
        
        # === Sub to Ping topic else, get sim parameters ===
        if not self.sim:
            self.sub_ping = self.create_subscription(Ping, self.get_parameter('ping_sub_topic').value, self.ping_CB, 10)
        else:
            self.max_range = 40.0
            self.horizontal_beamwidth = 70
            self.bearings = None  # will be set on first image using actual image width
            self.receive_ping = True

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
        Uses the polar image from the SONAR to extract intensities and project it as probability clouds.
        Publishes two clouds simultaneously: max_intensity (fan) and probabilistic_intensity.
        '''
        # Only start when both Ping & Marker msgs are in memory.
        if self.receive_ping and self.receive_marker:
            current = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            current = self.image_preprocess(current)
            self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(current, encoding="mono8"))

            # === Convert all valid pixels to sensor frame coordinates (both modes) ===
            mx_edge, mx_frame, pr_edge, pr_frame = self.extract_points_in_sensor_frame(current)

            # --- Build max_intensity (fan) cloud ---
            mx_sensor_x = mx_frame[:, 0].astype(np.float32)
            mx_sensor_y = mx_frame[:, 1].astype(np.float32)
            mx_intensity = mx_frame[:, 2].astype(np.float32)

            thresh_mask = mx_intensity >= self.intensity_threshold
            mx_sensor_x = mx_sensor_x[thresh_mask]
            mx_sensor_y = mx_sensor_y[thresh_mask]
            mx_intensity = mx_intensity[thresh_mask]

            mx_num = mx_sensor_x.shape[0]
            mx_points = np.zeros((mx_num, 4), dtype=np.float32)
            mx_points[:, 0] = mx_sensor_x
            mx_points[:, 1] = mx_sensor_y
            mx_points[:, 3] = mx_intensity

            mx_pcl_msg = self._build_pcl_msg(mx_num, mx_points.tobytes())
            self._depth_filter_and_publish(mx_pcl_msg, self.pub_pcl)

            # --- Build probabilistic_intensity cloud ---
            pr_sensor_x = pr_frame[:, 0].astype(np.float32)
            pr_sensor_y = pr_frame[:, 1].astype(np.float32)
            pr_intensity = pr_frame[:, 2].astype(np.float32)

            if not self.bool_create_sonar_geometry:
                # === Fixed geometry — compute and cache once ===
                x = pr_sensor_x[None, :]   # (1, N)
                y = pr_sensor_y[None, :]
                z = np.zeros_like(pr_sensor_x)[None, :]

                # === Rotate around Y (broadcasted) ===
                x_r = x * self.cos_a + z * self.sin_a
                y_r = y * np.ones_like(x_r)  # (B, N)
                z_r = -x * self.sin_a + z * self.cos_a

                all_points = np.stack((x_r, y_r, z_r), axis=-1).reshape(-1, 3)  # (B*N, 3)

                # Range filter mask
                ranges_xy = np.hypot(all_points[:, 0], all_points[:, 1])
                self._geom_mask = ranges_xy >= self.threshold_min_range
                filtered_points = all_points[self._geom_mask]

                # Voxel correspondence
                self.indices, _ = self.create_voxel_corresponding_points(self.voxel_centroids, filtered_points, method="closest_to_centroid", intensities=None)
                self.cached_positions = filtered_points[self.indices]

                self.bool_create_sonar_geometry = True

            # === Per-frame: probabilities only ===
            # shape: (B*N,) → filter → index
            all_probs = (pr_intensity[None, :] * self.beam_probs[:, None]).reshape(-1)
            filtered_probs = all_probs[self._geom_mask]

            pr_num = self.indices.shape[0]
            pr_points = np.empty((pr_num, 4), dtype=np.float32)
            pr_points[:, 0:3] = self.cached_positions
            pr_points[:, 3] = filtered_probs[self.indices]

            pr_pcl_msg = self._build_pcl_msg(pr_num, pr_points.tobytes())
            self._depth_filter_and_publish(pr_pcl_msg, self.pub_pcl_prob)

    def _build_pcl_msg(self, num_points: int, data: bytes) -> PointCloud2:
        '''Create a PointCloud2 from the fixed template, with the given width and data.'''
        pcl_msg = PointCloud2()
        pcl_msg.header.frame_id = self.frame_id
        pcl_msg.height = 1
        pcl_msg.fields = self.fields
        pcl_msg.point_step = self.pointcloud_msg.point_step
        pcl_msg.is_dense = True
        pcl_msg.width = num_points
        pcl_msg.row_step = pcl_msg.point_step * num_points
        pcl_msg.data = data
        return pcl_msg

    def _depth_filter_and_publish(self, pcl_msg: PointCloud2, publisher):
        '''
        Apply world-frame depth filter to a PointCloud2 message then publish it.
        Transforms to world frame, filters by max_depth, transforms back, publishes.
        '''
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                self.frame_id,
                rclpy.time.Time()
            )

            pcl_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(pcl_msg, transform)

            points_struct = pc2.read_points(
                pcl_msg,
                field_names=('x', 'y', 'z', 'intensity'),
                skip_nans=False
            )

            points = np.column_stack((
                points_struct['x'],
                points_struct['y'],
                points_struct['z'],
                points_struct['intensity']
            )).astype(np.float32)

            z = points[:, 2]
            points = points[z <= self.max_depth]
            points = [tuple(p) for p in points]

            pcl_msg = pc2.create_cloud(pcl_msg.header, self.fields, points)

            try:
                transform = self.tf_buffer.lookup_transform(
                    self.frame_id,
                    self.world_frame_id,
                    rclpy.time.Time()
                )
                pcl_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(pcl_msg, transform)
                publisher.publish(pcl_msg)
            except TransformException as e:
                self.get_logger().warn(f'Transform not available: {e}')
        except TransformException as e:
            self.get_logger().warn(f'Transform not available: {e}')
    
    def anisotropic_diffusion(self, image, niter=10, kappa=30, gamma=0.1):
        """
        Perona-Malik anisotropic diffusion.

        Smooths homogeneous regions while preserving edges.

        :param image: uint8 or float input image
        :param niter:  number of diffusion iterations
        :param kappa:  edge-sensitivity threshold — lower = more edge preservation
        :param gamma:  diffusion rate per iteration (≤ 0.25 for stability)
        :return: diffused image, same dtype as input
        """
        src_dtype = image.dtype
        img = image.astype(np.float32)

        for _ in range(niter):
            # Gradients in 4 directions
            dN = np.roll(img,  1, axis=0) - img
            dS = np.roll(img, -1, axis=0) - img
            dE = np.roll(img, -1, axis=1) - img
            dW = np.roll(img,  1, axis=1) - img

            # Perona-Malik conductance (exponential)
            cN = np.exp(-(dN / kappa) ** 2)
            cS = np.exp(-(dS / kappa) ** 2)
            cE = np.exp(-(dE / kappa) ** 2)
            cW = np.exp(-(dW / kappa) ** 2)

            img += gamma * (cN * dN + cS * dS + cE * dE + cW * dW)

        return np.clip(img, 0, 255).astype(src_dtype)

    def image_preprocess(self, current):    
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns
        if self.sim:
            # current = cv2.rotate(current, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return current
        else:
            # Lee filter. Better for multiplicative noise.
            # current = self.lee_filter(current, kernel_size=5)
            rows, columns = current.shape
            self.n_bins, self.n_beams = rows, columns
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

            # Anisotropic diffusion — smooths homogeneous regions, preserves edges
            current = self.anisotropic_diffusion(current, niter=5, kappa=30, gamma=0.1)
            return current
        
    def create_voxel_corresponding_points(self, voxel_points:np.ndarray, geometry_points:np.ndarray, method, intensities:np.ndarray=None):
        '''
        Finding the closest geometry_points corresponding to the voxel_points.

        :param geometry_points: List of geometry centroids (x,y,z)
        :param voxel_points: List of voxel centroids (x,y,z)
        :param intensities: Required for method="max_pool". Per-point intensity values aligned with geometry_points.

        :return (indices, distances_or_peak_intensities)
        '''
        if method == "closest_to_centroid":
            voxel_points = np.asarray(voxel_points, dtype=np.float32)
            geometry_points = np.asarray(geometry_points, dtype=np.float32)

            # Build KD-tree on geometry_points.
            tree = cKDTree(geometry_points)

            # Query nearest neighbor for each voxel
            distance, indices = tree.query(voxel_points, k=1)

            return indices, distance

        elif method == "max_pool":
            voxel_points    = np.asarray(voxel_points,    dtype=np.float32)
            geometry_points = np.asarray(geometry_points, dtype=np.float32)
            intensities     = np.asarray(intensities,     dtype=np.float32)

            # === Geometry (fixed) — cache once ===
            if not hasattr(self, '_max_pool_cache'):
                tree_v = cKDTree(voxel_points)

                # Estimate voxel cell half-size from nearest-neighbour spacing
                nn_dists, _ = tree_v.query(voxel_points, k=2)
                half_size = float(np.median(nn_dists[:, 1])) / 2.0

                # Assign every sensor point to its nearest voxel (264K → 1160 tree)
                dists, voxel_assignment = tree_v.query(geometry_points, k=1)

                # Keep only sensor points that fall inside the voxel cell
                in_cell = dists <= half_size
                self._max_pool_cache = {
                    'valid_sensor_idx': np.where(in_cell)[0],
                    'valid_voxel_idx':  voxel_assignment[in_cell],
                    'in_cell':          in_cell,
                    'n_voxels':         len(voxel_points),
                }

            valid_sensor_idx = self._max_pool_cache['valid_sensor_idx']
            valid_voxel_idx  = self._max_pool_cache['valid_voxel_idx']
            in_cell          = self._max_pool_cache['in_cell']
            n_voxels         = self._max_pool_cache['n_voxels']

            # === Per-frame: intensities only ===
            valid_intensities = intensities[in_cell]

            peak_indices     = np.full(n_voxels, -1, dtype=np.int64)
            peak_intensities = np.zeros(n_voxels, dtype=np.float32)

            if valid_sensor_idx.size > 0:
                # Sort by (voxel_id asc, intensity desc) → first entry per voxel = peak
                order = np.lexsort((-valid_intensities, valid_voxel_idx))
                sv = valid_voxel_idx[order]
                ss = valid_sensor_idx[order]
                si = valid_intensities[order]

                _, first = np.unique(sv, return_index=True)
                peak_indices[sv[first]]     = ss[first]
                peak_intensities[sv[first]] = si[first]

            return peak_indices, peak_intensities

        elif method == "median_pool":
            voxel_points    = np.asarray(voxel_points,    dtype=np.float32)
            geometry_points = np.asarray(geometry_points, dtype=np.float32)
            intensities     = np.asarray(intensities,     dtype=np.float32)

            # === Geometry (fixed) — cache once ===
            if not hasattr(self, '_median_pool_cache'):
                tree_v = cKDTree(voxel_points)

                nn_dists, _ = tree_v.query(voxel_points, k=2)
                half_size = float(np.median(nn_dists[:, 1])) / 2.0

                dists, voxel_assignment = tree_v.query(geometry_points, k=1)

                in_cell = dists <= half_size
                self._median_pool_cache = {
                    'valid_sensor_idx': np.where(in_cell)[0],
                    'valid_voxel_idx':  voxel_assignment[in_cell],
                    'in_cell':          in_cell,
                    'n_voxels':         len(voxel_points),
                }

            valid_sensor_idx = self._median_pool_cache['valid_sensor_idx']
            valid_voxel_idx  = self._median_pool_cache['valid_voxel_idx']
            in_cell          = self._median_pool_cache['in_cell']
            n_voxels         = self._median_pool_cache['n_voxels']

            # === Per-frame: intensities only ===
            valid_intensities = intensities[in_cell]

            median_indices     = np.full(n_voxels, -1, dtype=np.int64)
            median_intensities = np.zeros(n_voxels, dtype=np.float32)

            if valid_sensor_idx.size > 0:
                # Sort by (voxel_id asc, intensity asc) → middle entry per voxel = median
                order = np.lexsort((valid_intensities, valid_voxel_idx))
                sv = valid_voxel_idx[order]
                ss = valid_sensor_idx[order]
                si = valid_intensities[order]

                _, first, counts = np.unique(sv, return_index=True, return_counts=True)
                mid = first + counts // 2
                median_indices[sv[first]]     = ss[mid]
                median_intensities[sv[first]] = si[mid]

            return median_indices, median_intensities

    def extract_points_in_sensor_frame(self, image:np.ndarray):
        """
        Convert pixels in the image into sensor-frame coordinates (x, y, intensity),
        running both max_intensity and probabilistic_intensity branches simultaneously.

        Parameters
        ----------
        image : np.ndarray
            2D array (rows × columns) representing intensity values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - mx_image_coords:   (row, col, intensity) for max_intensity points
            - mx_sensor_coords:  (x, y, intensity) for max_intensity points
            - pr_image_coords:   (row, col, intensity) for probabilistic_intensity points
            - pr_sensor_coords:  (x, y, intensity) for probabilistic_intensity points
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
        # Create coordinate grids
        rows, cols = np.indices(image.shape)
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        intensities = image.flatten()

        # Convert to sensor frame (x, y) — cache cos/sin since geometry is fixed
        if not hasattr(self, '_cached_sensor_xy'):
            if self.bearings is None or len(self.bearings) != image.shape[1]:
                raw = np.linspace(-3500, 3500, image.shape[1])
                self.bearings = np.array([np.radians(b * 0.01) for b in raw]).squeeze()
            theta_values = self.bearings[cols_flat]
            # 40m / 517 beams = 0.07m/beams
            meters_per_beam = self.max_range / self.n_bins
            r_values = meters_per_beam * (self.n_bins - rows_flat)
            self._cached_sensor_xy = (
                r_values * np.cos(theta_values),  # sensor_x
                r_values * np.sin(theta_values),  # sensor_y
            )
        sensor_x, sensor_y = self._cached_sensor_xy

        # ------------------------------------------------------------------ #
        # --- max_intensity branch ---                                        #
        # ------------------------------------------------------------------ #
        mx_rows = rows_flat.copy()
        mx_cols = cols_flat.copy()
        mx_sx   = sensor_x.copy()
        mx_sy   = sensor_y.copy()
        mx_int  = intensities.copy()

        # Step 0: apply minimum range threshold
        ranges = np.sqrt(mx_sx**2 + mx_sy**2)
        valid_mask = ranges > self.threshold_min_range
        mx_rows = mx_rows[valid_mask]
        mx_cols = mx_cols[valid_mask]
        mx_sx   = mx_sx[valid_mask]
        mx_sy   = mx_sy[valid_mask]
        mx_int  = mx_int[valid_mask]

        if len(mx_int) == 0:
            mx_image_coords  = np.empty((0, 3))
            mx_sensor_coords = np.empty((0, 3))
        else:
            # Step 1: sort by columns
            sort_idx = np.argsort(mx_cols)
            sorted_cols        = mx_cols[sort_idx]
            sorted_intensities = mx_int[sort_idx]

            # Step 2: find boundaries for each column
            col_change = np.diff(sorted_cols, prepend=sorted_cols[0] - 1)
            col_starts = np.flatnonzero(col_change)

            # Step 3: find max intensity per column using reduceat
            max_vals = np.maximum.reduceat(sorted_intensities, col_starts)
            max_indices_in_sorted = []
            for start, val in zip(col_starts, max_vals):
                end = col_starts[col_starts > start][0] if np.any(col_starts > start) else len(sorted_intensities)
                local_idx = np.argmax(sorted_intensities[start:end])
                max_indices_in_sorted.append(start + local_idx)
            max_indices_in_sorted = np.array(max_indices_in_sorted)

            # Step 4: apply column-max mask
            final_mask = np.zeros_like(mx_int, dtype=bool)
            final_mask[sort_idx[max_indices_in_sorted]] = True
            mx_rows = mx_rows[final_mask]
            mx_cols = mx_cols[final_mask]
            mx_sx   = mx_sx[final_mask]
            mx_sy   = mx_sy[final_mask]
            mx_int  = mx_int[final_mask]

            # Step 5: remove points with low intensity
            intensity_mask = mx_int >= self.intensity_threshold
            mx_rows = mx_rows[intensity_mask]
            mx_cols = mx_cols[intensity_mask]
            mx_sx   = mx_sx[intensity_mask]
            mx_sy   = mx_sy[intensity_mask]
            mx_int  = mx_int[intensity_mask]

            if len(mx_int) == 0:
                mx_image_coords  = np.empty((0, 3))
                mx_sensor_coords = np.empty((0, 3))
            else:
                mx_image_coords  = np.column_stack((mx_rows, mx_cols, mx_int))
                mx_sensor_coords = np.column_stack((mx_sx,   mx_sy,   mx_int))

        # ------------------------------------------------------------------ #
        # --- probabilistic_intensity branch ---                              #
        # ------------------------------------------------------------------ #
        pr_rows = rows_flat.copy()
        pr_cols = cols_flat.copy()
        pr_sx   = sensor_x.copy()
        pr_sy   = sensor_y.copy()
        pr_int  = intensities.copy()

        # Build KD-tree from voxel points (z = 0 plane only)
        z_vox    = self.voxel_centroids[:, 2]
        z0_mask  = np.isclose(z_vox, 0.0)
        voxel_centroids_z0 = self.voxel_centroids[z0_mask]
        voxel_points_xy    = voxel_centroids_z0[:, :2]

        sensor_xy = np.column_stack((pr_sx, pr_sy))

        # Voxel correspondence via max-pool
        self.sensor_indices, pr_int = self.create_voxel_corresponding_points(
            voxel_points_xy, sensor_xy, method="median_pool", intensities=pr_int
        )

        # Spatial coordinates are voxel centroids; image coords from winning sensor point
        pr_rows = pr_rows[self.sensor_indices]
        pr_cols = pr_cols[self.sensor_indices]
        pr_sx   = voxel_points_xy[:, 0]
        pr_sy   = voxel_points_xy[:, 1]

        # Linear-interpolation probability mapping
        probabilities = np.full_like(pr_int, self.min_prob, dtype=float)
        mid_mask = (pr_int > self.lower_bound_intensity) & (pr_int < self.upper_bound_intensity)
        probabilities[mid_mask] = self.min_prob + (
            (pr_int[mid_mask] - self.lower_bound_intensity) /
            (self.upper_bound_intensity - self.lower_bound_intensity)
        ) * (self.max_prob - self.min_prob)
        probabilities[pr_int >= self.upper_bound_intensity] = 0.9
        pr_int = np.round(probabilities * 10) / 10

        pr_image_coords  = np.column_stack((pr_rows, pr_cols, pr_int))
        pr_sensor_coords = np.column_stack((pr_sx,   pr_sy,   pr_int))

        return mx_image_coords, mx_sensor_coords, pr_image_coords, pr_sensor_coords
    
    def pointcloud_CB(self, msg):
        '''
        Callback for PointCloud2 msg.<br>
        Publish the resultant probabilty cloud.
        Stores the msg as a .pcl file
        '''
        if self.save_as_pcd_bool:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame_id,
                self.frame_id,
                rclpy.time.Time()
                )

            # Transform to world frame
            msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(msg, transform)   
            pc_data = msg.data
            point_step = msg.point_step
            fields = msg.fields

            np_points = np.frombuffer(bytes(pc_data), dtype=np.float32).reshape(-1, point_step // 4)[:, :4]

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(np_points[:, :3])

            intens = np_points[:, 3]
            colors = intens.reshape(-1, 1)
            colors = np.repeat(colors, 3, axis=1)
            max_val = np.max(colors) if colors.size > 0 else 1.0
            colors = colors / max_val if max_val > 0 else colors
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