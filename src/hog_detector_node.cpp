#include <ros/ros.h>
#include <hog_detector/hog_detector.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "hog_detector_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::HogDetector hog_detector(nh, pnh);

    hog_detector.process();

    ros::spin();
    return 0;
}
