import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("/home/tony/auv_ws/submap.pcd")
points = np.asarray(pcd.points)

# Define normal and a point on the plane
normal = np.array([1, 0, 0])        # e.g., z-plane
normal = normal / np.linalg.norm(normal)
p0 = np.array([0, 0, 0])            # origin as plane point

# Project all points onto the plane
projections = points - np.dot((points - p0), normal[:, None]) * normal

# Create a new point cloud with projected points
proj_pcd = o3d.geometry.PointCloud()
proj_pcd.points = o3d.utility.Vector3dVector(projections)

# Visualize original and projected clouds together
pcd.paint_uniform_color([1, 0, 0])      # red: original
proj_pcd.paint_uniform_color([0, 1, 0]) # green: projected
o3d.visualization.draw_geometries([pcd, proj_pcd])


# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt

# # Load point cloud
# pcd = o3d.io.read_point_cloud("/home/tony/auv_ws/submap.pcd")
# points = np.asarray(pcd.points)

# # --- STEP 1: Define pole center (adjust this to match your setup) ---
# # For example, if your scanner is centered, then (0, 0)
# center = np.array([0.0, 0.0])  # (x0, y0)

# # Shift points to center the pole
# x, y, z = points[:, 0] - center[0], points[:, 1] - center[1], points[:, 2]

# # --- STEP 2: Convert to cylindrical coordinates ---
# theta = np.arctan2(y, x)       # Angle around the pole, range (-π, π)
# theta = (theta + 2 * np.pi) % (2 * np.pi)  # Optional: wrap to [0, 2π]
# r = np.sqrt(x**2 + y**2)       # Radius from pole center

# # Optional: sort by theta and z if you want structured output
# idx = np.lexsort((z, theta))
# theta, z, r = theta[idx], z[idx], r[idx]

# # --- STEP 3: Plot unwrapped view (theta vs z) ---
# plt.figure(figsize=(10, 6))
# plt.scatter(theta, z, c=r, cmap='viridis', s=1)
# plt.xlabel("Angle around pole (radians)")
# plt.ylabel("Height (z)")
# plt.title("Unwrapped LiDAR Pole Scan")
# plt.colorbar(label="Radius from center")
# plt.axis("equal")
# plt.tight_layout()
# plt.show()
