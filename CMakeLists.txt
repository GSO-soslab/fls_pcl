cmake_minimum_required(VERSION 3.5)
project(fls_pcl)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(include)

add_executable(loop_node
  src/loop.cpp)

ament_target_dependencies(loop_node
  rclcpp
  sensor_msgs)

install(TARGETS loop_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(PROGRAMS
  fls_pcl/fls_pcl.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()