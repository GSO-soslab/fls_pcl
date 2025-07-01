cmake_minimum_required(VERSION 3.13)

if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

project(fls_pcl)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
# find_package(Open3D REQUIRED)

# include_directories(
#   include
#   ${Open3D_INCLUDE_DIRS}
# )

# add_executable(loop_node
#   src/loop.cpp)

# ament_target_dependencies(loop_node
#   rclcpp
#   sensor_msgs
#   geometry_msgs
#   pcl_conversions
#   pcl_ros
# )

# target_link_libraries(loop_node ${Open3D_LIBRARIES})

# install(TARGETS loop_node
#   DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(PROGRAMS
  fls_pcl/fls_pcl.py
  fls_pcl/loop.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()