#include <ros/ros.h>
#include <object_state_estimator/object_state_estimator.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_state_estimator_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::ObjectStateEstimator object_state_estimator(nh, pnh);

    ros::spin();
    return 0;
}
