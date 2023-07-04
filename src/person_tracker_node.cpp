#include <ros/ros.h>
#include <person_tracker/person_tracker.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "person_tracker");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    PersonTracker person_tracker(nh, pnh);

    person_tracker.process();

    return 0;
}
