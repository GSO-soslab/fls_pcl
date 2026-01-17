import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.spatial import cKDTree
import cv2

# Arc length along azimuth = tan(35)*40*2 = 56m
# Arc length along elevation = tan(6)*40*2 = 8.4m
def create_valid_SONAR_points_2d(min_range, max_range, n_bins, azimuth_degrees, n_beams):
    """
    Generate valid SONAR return points in 2D (range + azimuth only).

    :param min_range: minimum range of the SONAR
    :param max_range: maximum range of the SONAR
    :param n_bins: number of range bins
    :param azimuth_degrees: horizontal field of view (degrees)
    :param n_beams: number of azimuth beams

    :return: (n_bins * n_beams, 2) array of (x, y) points
    """

    # Range bins
    ranges = np.linspace(min_range, max_range, n_bins)

    # Azimuth angles
    azimuth_half_rad = np.deg2rad(azimuth_degrees / 2)
    azimuth_angles = np.linspace(
        -azimuth_half_rad,
         azimuth_half_rad,
         n_beams
    )

    # Meshgrid (range × azimuth)
    R, AZ = np.meshgrid(ranges, azimuth_angles, indexing="ij")

    # Convert to Cartesian
    X = R * np.cos(AZ)
    Y = R * np.sin(AZ)
    Z = np.zeros_like(X)

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points

def create_valid_SONAR_points(min_range, max_range, n_bins, azimuth_degrees, n_beams, elevation_degrees, elevation_step_degree):
    """
    Using geometric information of the SONAR to determine points in cartesian space,
    wherein valid returns can lie in the sensor frame.
    
    :param min_range: minimum range of the SONAR
    :param max_range: maximum range of the SONAR
    :param n_bins: Number of bins per beam
    :param azimuth_degrees: Horizontal FOV of the SONAR
    :param n_beams: Number of beams.
    :param elevation_degrees: Vertical FOV of the SONAR
    :param elevation_step_degree: Increments of degree to consider for elevation.

    :return list of tuples which has (x,y,z) cordinates. len(list) should be n_bins x n_beams x (elevation_degrees/ elevation_step_degrees)
    """

    # Range along x
    x_vals = np.linspace(min_range, max_range, n_bins)

    # Angles
    azimuth_half_rad = np.deg2rad(azimuth_degrees / 2)
    elevation_half_rad = np.deg2rad(elevation_degrees / 2)

    azimuth_angles = np.linspace(
        -azimuth_half_rad,
         azimuth_half_rad,
         n_beams
    )

    n_elev_steps = int(elevation_degrees / elevation_step_degree)
    elevation_angles = np.linspace(
        -elevation_half_rad,
         elevation_half_rad,
         n_elev_steps
    )

    # Meshgrid
    X, AZ, EL = np.meshgrid(
        x_vals,
        azimuth_angles,
        elevation_angles,
        indexing="ij"
    )

    # Planar projection
    Y = X * np.tan(AZ)
    Z = X * np.tan(EL)

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    return points

def create_voxel_corresponding_points(voxel_points, geometry_points):
    '''
    Finding the closest geometry_points corresponding to the voxel_points.
    
    :param voxel_points: List of voxel centroids (x,y,z)
    :param geometry_points: List of geometry centroids (x,y,z)

    :return list of geometry_points (x,y,z) that is closest correspondence with voxel_points. Same size as of voxel_points 
    '''
    voxel_points = np.asarray(voxel_points, dtype=np.float32)
    geometry_points = np.asarray(geometry_points, dtype=np.float32)

    # Build KD-tree on geometry points
    tree = cKDTree(geometry_points)

    # Query nearest neighbor for each voxel
    _, indices = tree.query(voxel_points, k=1)

    return geometry_points[indices]
    

# geometr_points = create_valid_SONAR_points(0.1,40.0,517,70,512,12,1)
# geometr_points = create_valid_SONAR_points_2d(0.1, 40, 517,70,512)

# Load voxel centroids
voxel_centroids = np.loadtxt("/home/tony/auv_ws/src/fls_pcl/utils/voxel_centroids.txt", skiprows=1)  # skip header
post_process_points = np.loadtxt("/home/tony/auv_ws/src/fls_pcl/utils/post_process_points.txt", skiprows=1)  # skip header

print(voxel_centroids.shape)
print(post_process_points.shape)

plt.figure(0)
plt.hist(post_process_points[:,3])
plt.show()
# plt.figure(0)
# plt.scatter(post_process_points[:,0], post_process_points[:,1])
# plt.figure(1)

# plt.scatter(voxel_centroids[:,0], voxel_centroids[:,1])

# plt.show()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Voxel centroids (just xyz)
# ax.scatter(
#     voxel_centroids[:, 0],
#     voxel_centroids[:, 1],
#     voxel_centroids[:, 2],
#     c='blue',
#     s=5,
#     label='Voxel Centroids',
#     alpha=0.6
# )

# # Post-processed points (xyz)
# ax.scatter(
#     post_process_points[:, 0],
#     post_process_points[:, 1],
#     post_process_points[:, 2],
#     c='red',
#     s=5,
#     label='Post-Processed Points',
#     alpha=0.6
# )

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Voxel Centroids vs Post-Processed Points')
# ax.legend()
# plt.show()
# print(geometr_points[0])
# print(voxel_centroids[0])

# print(geometr_points.shape)  # should be (M, 3)
# print(voxel_centroids.shape)  # should be (N, 3)

# correspodence = create_voxel_corresponding_points(voxel_centroids, geometr_points)
# print(correspodence.shape)  # should be (N, 3)

# # image = cv2.imread('alpha_rise_oculus_raw_image-1764953518-592108253.png')

# points_xy_voxel = np.loadtxt("/home/tony/auv_ws/src/fls_pcl/utils/points_xy_voxel_points.txt") #7656
# sensor_xy_image = np.loadtxt("/home/tony/auv_ws/src/fls_pcl/utils/sensor_xy_image.txt") #264704
# corresponding_sensor_xy = create_voxel_corresponding_points(points_xy_voxel, sensor_xy_image)
# # print(len(sensor_xy_image))
# # print(len(points_xy_voxel))
# # print(len(corresponding_sensor_xy))

# voxel_to_image = np.loadtxt("/home/tony/auv_ws/src/fls_pcl/utils/voxel_to_image.txt")

# plt.figure(figsize=(8, 8))
# plt.scatter(points_xy_voxel[:, 0], points_xy_voxel[:, 1],
#             c='blue', s=1, label='Voxel Points', alpha=1.0)

# plt.scatter(voxel_to_image[:, 0], voxel_to_image[:, 1],
#             c='red', s=1, label='Sensor Points', alpha=0.1)

# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.title("2D Scatter: Voxel Points vs Sensor Points")
# plt.axis('equal')  # keep aspect ratio 1:1
# plt.legend()
# plt.grid(True)
# plt.show()
# tree = cKDTree(points_xy_voxel)

# distances, indices = tree.query(sensor_xy_image, k=1)

# print("min distance:", distances.min())
# print("mean distance:", distances.mean())
# print("max distance:", distances.max())

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot voxel centroids
# ax.scatter(
#     voxel_centroids[::100, 0],
#     voxel_centroids[::100, 1],
#     voxel_centroids[::100, 2],
#     c='blue',
#     s=5,
#     label='Voxel Centroids',
#     alpha=0.6
# )

# # Plot corresponding geometry points
# ax.scatter(
#     correspodence[::100, 0],
#     correspodence[::100, 1],
#     correspodence[::100, 2],
#     c='red',
#     s=5,
#     label='Corresponding Geometry',
#     alpha=0.6
# )

# Optional: draw lines connecting each voxel to its correspondence
# for i in range(0, len(voxel_centroids), max(1, len(voxel_centroids)//5000)):  # limit number of lines for performance
#     ax.plot(
#         [voxel_centroids[i,0], correspodence[i,0]],
#         [voxel_centroids[i,1], correspodence[i,1]],
#         [voxel_centroids[i,2], correspodence[i,2]],
#         c='gray',
#         linewidth=0.3,
#         alpha=0.3
#     )
# points = correspodence
# x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
# y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
# z_min, z_max = np.min(points[:,2]), np.max(points[:,2])

# length_x = x_max - x_min
# length_y = y_max - y_min
# length_z = z_max - z_min

# print(f"Length along X: {length_x:.3f} m")
# print(f"Length along Y (azimuth span): {length_y:.3f} m")
# print(f"Length along Z (elevation span): {length_z:.3f} m")

# points = voxel_centroids
# x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
# y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
# z_min, z_max = np.min(points[:,2]), np.max(points[:,2])

# length_x = x_max - x_min
# length_y = y_max - y_min
# length_z = z_max - z_min

# print(f"Length along X: {length_x:.3f} m")
# print(f"Length along Y (azimuth span): {length_y:.3f} m")
# print(f"Length along Z (elevation span): {length_z:.3f} m")

# diff = voxel_centroids - correspodence  # shape (N,3)

# # Compute squared distances
# squared_dists = np.sum(diff**2, axis=1)  # shape (N,)

# # Compute RMSE
# rmse = np.sqrt(np.mean(squared_dists))

# print(f"RMSE between voxel centroids and corresponding geometry points: {rmse:.6f} m")

# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('Voxel Centroids and Corresponding Geometry Points')
# ax.legend()
# ax.view_init(elev=30, azim=-60)  # optional: rotate view for better perspective

# plt.show()