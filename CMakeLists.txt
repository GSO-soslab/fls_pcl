cmake_minimum_required(VERSION 3.5)
project(fls_ism)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(oculus_interfaces REQUIRED)

# Install
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