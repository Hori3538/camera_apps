#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <mask_rcnn/mask_rcnn.h>


namespace camera_apps{
    class MaskRcnnNodelet : public nodelet::Nodelet
    {
        public:
            MaskRcnnNodelet() = default;
            ~MaskRcnnNodelet() {
        if (mask_rcnn_) delete mask_rcnn_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                mask_rcnn_ = new camera_apps::MaskRcnn(nh, pnh);
            }
            camera_apps::MaskRcnn *mask_rcnn_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::MaskRcnnNodelet, nodelet::Nodelet);
