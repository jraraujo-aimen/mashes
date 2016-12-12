cmake_minimum_required(VERSION 2.8.3)
project(mashes_calibration)

find_package(catkin REQUIRED COMPONENTS
  tf
  rospy
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package(
  DEPENDS message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
