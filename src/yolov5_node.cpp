#include <ros/ros.h>
#include <yolov5/yolov5.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolov5_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::Yolov5 yolov5(nh, pnh);

    ros::spin();
    return 0;
}
