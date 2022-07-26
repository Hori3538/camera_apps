#include <ros/ros.h>
#include <motion_predictor/motion_predictor.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "motion_predictor");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    camera_apps::MotionPredictor motion_predictor(nh, pnh);

    ros::spin();
    return 0;
}
