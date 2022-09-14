#ifndef MASK_RCNN2
#define MASK_RCNN2

#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <sstream>


#include <camera_apps_msgs/BoundingBox.h>
#include <camera_apps_msgs/BoundingBoxes.h>
#include <camera_apps_msgs/Mask.h>
#include <camera_apps_msgs/Masks.h>

namespace camera_apps
{
    class MaskRcnn2
    {
        public:
            MaskRcnn2(ros::NodeHandle &nh, ros::NodeHandle &pnh);
        private:
            void image_callback(const sensor_msgs::ImageConstPtr &msg);
            std::vector<std::string> read_file(std::string filename, char delimiter='\n');

            void set_network();
            void object_detect(cv::Mat &image);
            void draw_bbox(cv::Mat &image, cv::Rect rect, int id, float conf, cv::Mat& object_mask);
            void set_mask(cv::Rect rect, int id, float conf, cv::Mat& mask, std::string class_name);

            std::string camera_topic_name_;
            std::string model_path_;
            double conf_threshold_;
            double mask_threshold_;
            bool detect_only_person_;

            std::vector<std::string> class_names_;
            std::vector<cv::Scalar> colors_;

            cv::Mat input_image_;
            cv::Mat detection_image_;
            cv::dnn::Net net_;
            camera_apps_msgs::Masks masks_;

            image_transport::Subscriber image_sub_;
            image_transport::Publisher image_pub_;
            ros::Publisher masks_pub_;
    };
}
#endif
