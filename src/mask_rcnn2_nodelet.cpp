#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <mask_rcnn2/mask_rcnn2.hpp>


namespace camera_apps{
    class MaskRcnn2Nodelet : public nodelet::Nodelet
    {
        public:
            MaskRcnn2Nodelet() = default;
            ~MaskRcnn2Nodelet() {
        if (mask_rcnn2_) delete mask_rcnn2_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                mask_rcnn2_ = new camera_apps::MaskRcnn2(nh, pnh);
            }
            camera_apps::MaskRcnn2 *mask_rcnn2_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::MaskRcnn2Nodelet, nodelet::Nodelet);
