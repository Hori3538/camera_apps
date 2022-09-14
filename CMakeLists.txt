cmake_minimum_required(VERSION 3.0.2)
project(camera_apps)

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

add_executable(object_detector_node src/object_detector_node.cpp src/object_detector.cpp)
add_dependencies(object_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    object_detector_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
add_library(object_detector_nodelet src/object_detector.cpp src/object_detector_nodelet.cpp)
target_link_libraries(object_detector_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

add_executable(mask_rcnn_node src/mask_rcnn_node.cpp src/mask_rcnn.cpp)
add_dependencies(mask_rcnn_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    mask_rcnn_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
add_library(mask_rcnn_nodelet src/mask_rcnn.cpp src/mask_rcnn_nodelet.cpp)
target_link_libraries(mask_rcnn_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

add_executable(object_state_estimator_node src/object_state_estimator_node.cpp src/object_state_estimator.cpp)
add_dependencies(object_state_estimator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    object_state_estimator_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    )
add_library(object_state_estimator_nodelet src/object_state_estimator.cpp src/object_state_estimator_nodelet.cpp)
target_link_libraries(object_state_estimator_nodelet
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

add_executable(motion_predictor_node src/motion_predictor_node.cpp src/motion_predictor.cpp src/hungarian.cpp)
add_dependencies(motion_predictor_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    motion_predictor_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Eigen3_LIBRARIES}
    )
add_library(motion_predictor_nodelet src/motion_predictor.cpp src/motion_predictor_nodelet.cpp src/hungarian.cpp)
target_link_libraries(motion_predictor_nodelet
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Eigen3_LIBRARIES}
)

add_executable(person_recognizer_node src/person_recognizer_node.cpp src/person_recognizer.cpp)
add_dependencies(person_recognizer_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    person_recognizer_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Eigen3_LIBRARIES}
    )
add_library(person_recognizer_nodelet src/person_recognizer.cpp src/person_recognizer_nodelet.cpp)
target_link_libraries(person_recognizer_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Eigen3_LIBRARIES}
)

add_executable(hog_detector_node src/hog_detector_node.cpp src/hog_detector.cpp)
add_dependencies(hog_detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    hog_detector_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
add_library(hog_detector_nodelet src/hog_detector.cpp src/hog_detector_nodelet.cpp)
target_link_libraries(hog_detector_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
)

add_executable(yolov5_node src/yolov5_node.cpp src/yolov5.cpp)
add_dependencies(yolov5_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(
    yolov5_node
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    )
add_library(yolov5_nodelet src/yolov5.cpp src/yolov5_nodelet.cpp)
target_link_libraries(yolov5_nodelet
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
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
