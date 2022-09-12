#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <object_detector/object_detector.h>


namespace camera_apps{
    class ObjectDetectorNodelet : public nodelet::Nodelet
    {
        public:
            ObjectDetectorNodelet() = default;
            ~ObjectDetectorNodelet() {
        if (object_detector_) delete object_detector_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                object_detector_ = new camera_apps::ObjectDetector(nh, pnh);
            }
            camera_apps::ObjectDetector *object_detector_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::ObjectDetectorNodelet, nodelet::Nodelet);
