cmake_minimum_required(VERSION 3.5)
project(fls_ism)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(oculus_interfaces REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)  # for Eigen::Matrix4f, Eigen::Vector3f, etc.
find_package(nav_msgs REQUIRED)

# Create executable
add_executable(voxel_log_odds_visualizer src/voxel_log_odds_visualizer.cpp)

# Link libraries / dependencies
ament_target_dependencies(voxel_log_odds_visualizer
  rclcpp
  sensor_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  Eigen3
  nav_msgs
)
target_include_directories(voxel_log_odds_visualizer   PUBLIC ${EIGEN3_INCLUDE_DIR})

# Install
install(TARGETS
  voxel_log_odds_visualizer
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(PROGRAMS
  fls_ism/fls_pcl.py
  fls_ism/depth_planner.py
  fls_ism/fls_voxels.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()