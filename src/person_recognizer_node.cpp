#include <ros/ros.h>
#include <person_recognizer/person_recognizer.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "person_recognizer_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::PersonRecognizer person_recognizer(nh, pnh);

    ros::spin();
    return 0;
}
