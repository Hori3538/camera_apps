#ifndef YOLOV5
#define YOLOV5

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>

#include <fstream>
#include <sstream>

#include <camera_apps_msgs/BoundingBox.h>
#include <camera_apps_msgs/BoundingBoxes.h>

namespace camera_apps
{
    class Yolov5
    {
        public:
            Yolov5(ros::NodeHandle &nh, ros::NodeHandle &pnh);
        private:
            void image_callback(const sensor_msgs::ImageConstPtr &msg);
            std::vector<std::string> read_file(std::string filename, char delimiter='\n');
            void set_network();
            std::vector<cv::Mat> pre_process(cv::Mat &input_image);
            cv::Mat post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name);
            void draw_label(cv::Mat& input_image, std::string label, int left, int top);
            void object_detect(cv::Mat &image);
            void set_bbox(int x0, int x1, int y0, int y1, float conf, int id, std::string class_name);

            std::string camera_topic_name_;
            std::string model_path_;
            double conf_threshold_;
            double score_threshold_;
            double nms_threshold_;

            const int input_width_ = 640;
            const int input_height_ = 640;
            const int up_width_ = 960;
            const int up_height_ = 720;
            const double font_scale_ = 0.7;
            const int font_face_ = cv::FONT_HERSHEY_SIMPLEX;
            const int thickness_ = 1;

            std::vector<std::string> class_names_;

            cv::Mat input_image_;
            cv::Mat detection_image_;
            cv::dnn::Net net_;
            camera_apps_msgs::BoundingBoxes bboxes_;

            image_transport::Subscriber image_sub_;
            image_transport::Publisher image_pub_;
            ros::Publisher bboxes_pub_;
    };
}
#endif
