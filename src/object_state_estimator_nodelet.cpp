#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <object_state_estimator/object_state_estimator.h>


namespace camera_apps{
    class ObjectStateEstimatorNodelet : public nodelet::Nodelet
    {
        public:
            ObjectStateEstimatorNodelet() = default;
            ~ObjectStateEstimatorNodelet() {
        if (object_state_estimator_) delete object_state_estimator_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                object_state_estimator_ = new camera_apps::ObjectStateEstimator(nh, pnh);
            }
            camera_apps::ObjectStateEstimator *object_state_estimator_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::ObjectStateEstimatorNodelet, nodelet::Nodelet);
