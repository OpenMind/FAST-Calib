cmake_minimum_required(VERSION 3.8)
project(fast_calib)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(cv_bridge REQUIRED)

set(dependencies
  rclcpp
  sensor_msgs
  pcl_conversions
  rosbag2_cpp
  rosbag2_storage
  cv_bridge
)

# The Livox CustomMsg input is optional: without livox_ros_driver2 the tool
# still builds and calibrates against any sensor_msgs/PointCloud2 LiDAR.
find_package(livox_ros_driver2 QUIET)
if(livox_ros_driver2_FOUND)
  message(STATUS "livox_ros_driver2 found: Livox CustomMsg input enabled")
  add_definitions(-DFAST_CALIB_LIVOX_ENABLED)
  list(APPEND dependencies livox_ros_driver2)
else()
  message(WARNING "livox_ros_driver2 not found: building without Livox CustomMsg bag support")
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(fast_calib src/main.cpp)
ament_target_dependencies(fast_calib ${dependencies})
target_link_libraries(fast_calib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(multi_fast_calib src/multi_scene.cpp)
ament_target_dependencies(multi_fast_calib ${dependencies})
target_link_libraries(multi_fast_calib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS fast_calib multi_fast_calib
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz_cfg calib_data output
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
