cmake_minimum_required(VERSION 3.9)

project(pragmabot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  message_filters
  topic_tools
  tf
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    sensor_msgs
    geometry_msgs
    cv_bridge
    message_filters
    topic_tools
    tf
)

catkin_install_python(
  PROGRAMS 
    nodes/pragmabot_node.py
    nodes/image_decompressor_node.py
    nodes/image_republisher_node.py
    nodes/memory_manager_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
