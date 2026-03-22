#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
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

        self.declare_parameter('vertical_fov_deg', Parameter.Type.DOUBLE)
        self.vertical_beamwidth = self.get_parameter('vertical_fov_deg').value

        self.declare_parameter('sensor_frame_id', Parameter.Type.STRING)
        self.frame_id = self.get_parameter('sensor_frame_id').value

        self.declare_parameter('world_frame_id', Parameter.Type.STRING)
        self.world_frame_id = self.get_parameter('world_frame_id').value

        self.declare_parameter('ping_sub_topic',Parameter.Type.STRING)
        self.declare_parameter('marker_topic',Parameter.Type.STRING)
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
        self.pub_pcl_max      = self.create_publisher(PointCloud2, pub_topic + '/max_intensity', 10)
        self.pub_pcl_prob = self.create_publisher(PointCloud2, pub_topic + '/probability', 10)
        self.pub_pcl_raw  = self.create_publisher(PointCloud2, pub_topic + '/raw', 10)
        self.pub_fls_median_image = self.create_publisher(Image, pub_topic+'/image/filtered', 10)

        # === Subscribers === 
        self.sub_image = self.create_subscription(Image, self.get_parameter('image_sub_topic').value, self.image_cb, 10)
        self.sub_pcl = self.create_subscription(PointCloud2, pub_topic, self.pointcloud_cb, 10)
        
        # === Sub to Ping topic else, get sim parameters ===
        if not self.sim:
            self.sub_ping = self.create_subscription(Ping, self.get_parameter('ping_sub_topic').value, self.ping_cb, 10)
        else:
            self.max_range = 40.0
            self.horizontal_beamwidth = 70
            self.bearings = None  # will be set on first image using actual image width
            self.receive_ping = True

        # === Sub to Marker topic for Voxels ===
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_marker = self.create_subscription(Marker, self.get_parameter('marker_topic').value, self.marker_cb, latched_qos)

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

    def marker_cb(self, msg:Marker):
        '''
        Callback for Marker msg. 
        Stores the voxel centroids in sensor frame.

        Returns: 
            self.voxel_centroids as numpy array
        '''
        if not self.receive_marker:
            self.voxel_centroids = np.array([(p.x, p.y, p.z) for p in msg.points],dtype=np.float64)
            self.marker_resolution = msg.scale.x
            self.receive_marker = True
            self.destroy_subscription(self.sub_marker)
            self.sub_marker = None
        
    def ping_cb(self, msg:Ping):
        '''
        Callback for Ping msg.
        Stores information about the SONAR. All range & azimuth are sensor frame.

        Returns:
            self.max_range<br>
            self.bearings (azimuth of each beam in radians) as np.array<br>
        '''
        if not self.receive_ping:
            self.bearings = np.array([np.radians(bearings * 0.01) for bearings in msg.bearings]).squeeze()
            self.max_range = msg.range

            # High / Low frequency mode
            if msg.master_mode == 2:
                self.horizontal_beamwidth = 70 
            elif msg.master_mode == 1:
                self.horizontal_beamwidth = 130
        
            self.receive_ping = True
            self.destroy_subscription(self.sub_ping)
            self.sub_ping = None

    def image_cb(self, msg:Image):
        '''
        Callback for Image msg.<br>
        Uses the polar image from the SONAR to extract intensities and project it as probability clouds.
        Publishes two clouds simultaneously: max_intensity (fan) and probabilistic_intensity.
        '''
        # Only start when both Ping & Marker msgs are in memory.
        if self.receive_ping and self.receive_marker:
            raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            K = self.image_preprocess(raw_image.copy())
            self.pub_fls_median_image.publish(self.bridge.cv2_to_imgmsg(K, encoding="mono8"))

            # === Build sensor-frame coordinate cache, then extract both clouds ===
            self.ensure_sensor_xy_cache(K)
            mx_frame = self.extract_max_intensity(K)
            pr_frame = self.extract_probabilistic(K)

            # --- Build raw intensity cloud (unfiltered image, no range/depth filter) ---
            raw_intensities = raw_image.flatten().astype(np.float32)
            raw_sx, raw_sy = self._cached_sensor_xy
            raw_num = raw_sx.shape[0]
            raw_points = np.zeros((raw_num, 4), dtype=np.float32)
            raw_points[:, 0] = raw_sx
            raw_points[:, 1] = raw_sy
            raw_points[:, 3] = raw_intensities
            raw_pcl_msg = self.build_pcl_msg(raw_num, raw_points.tobytes())
            raw_pcl_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_pcl_raw.publish(raw_pcl_msg)

            # --- Build max_intensity (fan) cloud ---
            mx_sensor_x = mx_frame[:, 0].astype(np.float32)
            mx_sensor_y = mx_frame[:, 1].astype(np.float32)
            mx_intensity = mx_frame[:, 2].astype(np.float32)

            mx_num = mx_sensor_x.shape[0]
            mx_points = np.zeros((mx_num, 4), dtype=np.float32)
            mx_points[:, 0] = mx_sensor_x
            mx_points[:, 1] = mx_sensor_y
            mx_points[:, 3] = mx_intensity

            mx_pcl_msg = self.build_pcl_msg(mx_num, mx_points.tobytes())
            self.depth_filter_and_publish(mx_pcl_msg, self.pub_pcl_max)

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
                self.indices, _ = self.closest_to_centroid(self.voxel_centroids, filtered_points)
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

            pr_pcl_msg = self.build_pcl_msg(pr_num, pr_points.tobytes())
            self.depth_filter_and_publish(pr_pcl_msg, self.pub_pcl_prob)

    def build_pcl_msg(self, num_points: int, data: bytes) -> PointCloud2:
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

    def depth_filter_and_publish(self, pcl_msg: PointCloud2, publisher):
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

            pcl_msg.data = points.tobytes()
            pcl_msg.width = len(points)
            pcl_msg.row_step = pcl_msg.point_step * len(points)

            inv_transform = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.world_frame_id,
                rclpy.time.Time()
            )
            pcl_msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(pcl_msg, inv_transform)
            publisher.publish(pcl_msg)
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

    def image_preprocess(self, K):    
        rows, columns = K.shape
        self.n_bins, self.n_beams = rows, columns
        if self.sim:
            return K
        else:
            h, w = K.shape
            mid = w // 2

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
            K[mask] = 0

            # Anisotropic diffusion — smooths homogeneous regions, preserves edges
            K = self.anisotropic_diffusion(K, niter=5, kappa=30, gamma=0.1)
            return K
        
    def closest_to_centroid(self, voxel_points: np.ndarray, geometry_points: np.ndarray):
        '''Nearest-neighbor lookup: for each voxel, find the closest geometry point.'''
        voxel_points    = np.asarray(voxel_points,    dtype=np.float32)
        geometry_points = np.asarray(geometry_points, dtype=np.float32)
        tree = cKDTree(geometry_points)
        distance, indices = tree.query(voxel_points, k=1)
        return indices, distance

    def max_pool(self, voxel_points: np.ndarray, geometry_points: np.ndarray, intensities: np.ndarray):
        '''Assign each sensor point to its nearest voxel; return peak intensity per voxel.'''
        voxel_points    = np.asarray(voxel_points,    dtype=np.float32)
        geometry_points = np.asarray(geometry_points, dtype=np.float32)
        intensities     = np.asarray(intensities,     dtype=np.float32)

        if not hasattr(self, '_max_pool_cache'):
            tree_v = cKDTree(voxel_points)
            nn_dists, _ = tree_v.query(voxel_points, k=2)
            half_size = float(np.median(nn_dists[:, 1])) / 2.0
            dists, voxel_assignment = tree_v.query(geometry_points, k=1)
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

        valid_intensities = intensities[in_cell]
        peak_indices      = np.full(n_voxels, -1, dtype=np.int64)
        peak_intensities  = np.zeros(n_voxels, dtype=np.float32)

        if valid_sensor_idx.size > 0:
            order = np.lexsort((-valid_intensities, valid_voxel_idx))
            sv = valid_voxel_idx[order]
            ss = valid_sensor_idx[order]
            si = valid_intensities[order]
            _, first = np.unique(sv, return_index=True)
            peak_indices[sv[first]]     = ss[first]
            peak_intensities[sv[first]] = si[first]

        return peak_indices, peak_intensities

    def ensure_sensor_xy_cache(self, image: np.ndarray):
        '''Build and cache sensor-frame XY coordinates for all pixels (fixed geometry).'''
        if not hasattr(self, '_cached_sensor_xy'):
            rows_flat, cols_flat = (a.flatten() for a in np.indices(image.shape))
            if self.bearings is None or len(self.bearings) != image.shape[1]:
                raw = np.linspace(-3500, 3500, image.shape[1])
                self.bearings = np.array([np.radians(b * 0.01) for b in raw]).squeeze()
            theta = self.bearings[cols_flat]
            meters_per_beam = self.max_range / self.n_bins
            r = meters_per_beam * (self.n_bins - rows_flat)
            self._cached_sensor_xy = (r * np.cos(theta), r * np.sin(theta))

    def extract_max_intensity(self, image: np.ndarray) -> np.ndarray:
        '''Peak-intensity pixel per sonar beam column. Returns (N, 3): sensor_x, sensor_y, intensity.'''
        sensor_x, sensor_y = self._cached_sensor_xy
        cols_flat = np.indices(image.shape)[1].flatten()
        ins = image.flatten().astype(np.float32)

        valid = np.hypot(sensor_x, sensor_y) > self.threshold_min_range
        sx, sy, col, ins = sensor_x[valid], sensor_y[valid], cols_flat[valid], ins[valid]

        if len(ins) == 0:
            return np.empty((0, 3), dtype=np.float32)

        order = np.lexsort((-ins, col))
        _, first_occ = np.unique(col[order], return_index=True)
        keep = order[first_occ]
        mask = ins[keep] >= self.intensity_threshold
        return np.column_stack((sx[keep][mask], sy[keep][mask], ins[keep][mask]))

    def extract_probabilistic(self, image: np.ndarray) -> np.ndarray:
        '''Max-pool intensities onto voxel centroids, map to probabilities. Returns (N, 3): voxel_x, voxel_y, probability.'''
        sensor_x, sensor_y = self._cached_sensor_xy
        ins = image.flatten().astype(np.float32)

        voxel_xy = self.voxel_centroids[np.isclose(self.voxel_centroids[:, 2], 0.0), :2]
        sensor_xy = np.column_stack((sensor_x, sensor_y))

        _, pr_int = self.max_pool(voxel_xy, sensor_xy, ins)

        # Start with 0.2
        probabilities = np.full_like(pr_int, self.min_prob, dtype=np.float32)
        # Find linear region
        mid_mask = (pr_int > self.lower_bound_intensity) & (pr_int < self.upper_bound_intensity)
        # Map linear region
        probabilities[mid_mask] = self.min_prob + (
            (pr_int[mid_mask] - self.lower_bound_intensity) /
            (self.upper_bound_intensity - self.lower_bound_intensity)
        ) * (self.max_prob - self.min_prob)
        # Map higher regions
        probabilities[pr_int >= self.upper_bound_intensity] = self.max_prob
        pr_int = np.round(probabilities * 10) / 10

        return np.column_stack((voxel_xy[:, 0], voxel_xy[:, 1], pr_int))
    
    def pointcloud_cb(self, msg):
        '''
        Callback for PointCloud2 msg.<br>
        Publish the resultant probabilty cloud.
        Stores the msg as a .pcl file
        '''
        if self.save_as_pcd_bool:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame_id,
                    self.frame_id,
                    rclpy.time.Time()
                )
            except TransformException as e:
                self.get_logger().warn(f'Transform not available: {e}')
                return

            # Transform to world frame
            msg = tf2_sensor_msgs.tf2_sensor_msgs.do_transform_cloud(msg, transform)   
            pc_data = msg.data
            point_step = msg.point_step

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