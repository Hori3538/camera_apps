#include <ros/ros.h>
#include <object_detector/object_detector.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_detector_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::ObjectDetector object_detector(nh, pnh);

    ros::spin();
    return 0;
}
