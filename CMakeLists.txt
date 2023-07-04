cmake_minimum_required(VERSION 3.0.2)
project(camera_apps)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
    pluginlib
    nodelet
    camera_apps_msgs
    pcl_ros
    eigen_conversions
)
find_package(Eigen3 REQUIRED)

SET(OCV_VERSION "4.5.5")
find_package(OpenCV ${OCV_VERSION})

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES camera_apps
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
)

add_executable(person_tracker_node src/person_tracker_node.cpp src/person_tracker.cpp src/hungarian.cpp)
add_dependencies(person_tracker_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    person_tracker_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Eigen3_LIBRARIES}
    )

add_executable(mask_rcnn2_node src/mask_rcnn2_node.cpp src/mask_rcnn2.cpp)
add_dependencies(mask_rcnn2_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    mask_rcnn2_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
add_library(mask_rcnn2_nodelet src/mask_rcnn2.cpp src/mask_rcnn2_nodelet.cpp)
target_link_libraries(mask_rcnn2_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)
