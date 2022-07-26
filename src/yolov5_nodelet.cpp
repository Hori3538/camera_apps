#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <yolov5/yolov5.h>


namespace camera_apps{
    class Yolov5Nodelet : public nodelet::Nodelet
    {
        public:
            Yolov5Nodelet() = default;
            ~Yolov5Nodelet() {
        if (yolov5_) delete yolov5_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                yolov5_ = new camera_apps::Yolov5(nh, pnh);
            }
            camera_apps::Yolov5 *yolov5_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::Yolov5Nodelet, nodelet::Nodelet);
