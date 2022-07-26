#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <hog_detector/hog_detector.h>


namespace camera_apps{
    class HogDetectorNodelet : public nodelet::Nodelet
    {
        public:
            HogDetectorNodelet() = default;
            ~HogDetectorNodelet() {
        if (hog_detector_) delete hog_detector_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                hog_detector_ = new camera_apps::HogDetector(nh, pnh);
                hog_detector_->process();
            }
            camera_apps::HogDetector *hog_detector_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::HogDetectorNodelet, nodelet::Nodelet);
