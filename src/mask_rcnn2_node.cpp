#include <ros/ros.h>
#include <mask_rcnn2/mask_rcnn2.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mask_rcnn2_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::MaskRcnn2 mask_rcnn2(nh, pnh);

    ros::spin();
    return 0;
}
