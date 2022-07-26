#ifndef HOG_DETECTOR
#define HOG_DETECTOR

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/objdetect.hpp>

#include <fstream>
#include <sstream>

#include <camera_apps_msgs/BoundingBox.h>
#include <camera_apps_msgs/BoundingBoxes.h>

namespace camera_apps
{
    class HogDetector
    {
        public:
            HogDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh);
            void process();
        private:
            void image_callback(const sensor_msgs::ImageConstPtr &msg);
            std::vector<std::string> read_file(std::string filename, char delimiter='\n');
            void set_network();
            void object_detect(cv::Mat &image);
            void draw_bbox(cv::Mat &image, cv::Rect& rect);
            // void draw_bbox(cv::Mat &image, int x0, int y0, int x1, int y1, std::string label);
            void set_bbox(int x0, int x1, int y0, int y1, float conf, int id, std::string class_name);
            void set_bbox(cv::Rect& rect, float weight);
            void resize_boxes(cv::Rect& box);
            // void send_bbox(int x0, int x1, int y0, int y1, float conf, int id, std::string class_name);

            std::string camera_topic_name_;
            std::string model_path_;
            double conf_threshold_;
            double hit_threshold_;

            std::vector<std::string> class_names_;

            cv::Mat input_image_;
            // ros::Time msg_stamp_;
            cv::Mat detection_image_;
            cv::dnn::Net net_;
            camera_apps_msgs::BoundingBoxes bboxes_;
            // camera_apps_msgs::BoundingBox bbox_;

            image_transport::Subscriber image_sub_;
            image_transport::Publisher image_pub_;
            ros::Publisher bboxes_pub_;
            // ros::Publisher bbox_pub_;
    };
}
#endif
