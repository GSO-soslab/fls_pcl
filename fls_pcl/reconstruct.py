import open3d as o3d
import numpy as np

# 1. Load your point cloud (.pcd or .ply)
pcd = o3d.io.read_point_cloud("/home/tony/auv_ws/submap.pcd")

# Optional: Estimate normals (required for Poisson)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# 2. Run Poisson surface reconstruction
print("Running Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Optional: Crop low-density areas (to remove artifacts)
print("Removing low-density vertices...")
vertices_to_keep = densities > np.quantile(densities, 0.05)
mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

# 3. Visualize the result
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# 4. Save the mesh to a file
o3d.io.write_triangle_mesh("reconstructed_mesh.ply", mesh)
print("Saved mesh to reconstructed_mesh.ply")


## To view
#pc = o3d.io.read_point_cloud("reconstructed_mesh.ply")
#o3d.visualization.draw_geometries([pc], window_name="Open3D Viewer")