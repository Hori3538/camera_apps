#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <motion_predictor/motion_predictor.h>


namespace camera_apps{
    class MotionPredictorNodelet : public nodelet::Nodelet
    {
        public:
            MotionPredictorNodelet() = default;
            ~MotionPredictorNodelet() {
        if (motion_predictor_) delete motion_predictor_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                motion_predictor_ = new camera_apps::MotionPredictor(nh, pnh);
            }
            camera_apps::MotionPredictor *motion_predictor_;
    };
}
PLUGINLIB_EXPORT_CLASS(camera_apps::MotionPredictorNodelet, nodelet::Nodelet);
