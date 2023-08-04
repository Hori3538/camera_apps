#include <ros/ros.h>
#include <mask_rcnn/mask_rcnn.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mask_rcnn_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::MaskRcnn mask_rcnn(nh, pnh);

    mask_rcnn.process();
    return 0;
}
