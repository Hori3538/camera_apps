#ifndef PERSON_RECOGNIZER
#define PERSON_RECOGNIZER

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>

#include <fstream>
#include <sstream>


#include <camera_apps_msgs/BoundingBox.h>
#include <camera_apps_msgs/BoundingBoxes.h>
#include <camera_apps_msgs/ObjectState.h>
#include <camera_apps_msgs/ObjectStates.h>

namespace camera_apps
{
    struct DetectInfo
    {
        std::string label;
        double conf;
        cv::Mat mask;
        cv::Rect roi;
    };

    class PersonRecognizer
    {
        public:
            PersonRecognizer(ros::NodeHandle &nh, ros::NodeHandle &pnh);
            ~PersonRecognizer();
        private:
            // void sync_callback(const sensor_msgs::ImageConstPtr &image_msg, const sensor_msgs::PointCloud2ConstPtr &pc_msg);
            void image_callback(const sensor_msgs::ImageConstPtr &msg);
            void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg);
            std::vector<std::string> read_file(std::string filename, char delimiter='\n');

            void set_network();
            void object_detect(cv::Mat &image);
            void draw_bbox(cv::Mat &image, cv::Rect rect, int id, float conf, cv::Mat& object_mask);
            cv::Mat calc_hist(cv::Rect roi, cv::Mat mask);
            std::vector<float> convert_hist_to_vector(cv::Mat& hist);
            // void set_bbox(int x0, int x1, int y0, int y1, float conf, int id, std::string class_name);

            void create_object_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc, cv::Rect& roi, cv::Mat& mask);
            void downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in);
            bool clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in);
            geometry_msgs::PointStamped caluculate_centroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in);
            void through_filtering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in);
            
            // void send_bbox(int x0, int x1, int y0, int y1, float conf, int id, std::string class_name);

            std::string camera_topic_name_;
            std::string model_path_;
            double conf_threshold_;
            double mask_threshold_;

            int points_limit_;
            double cluster_tolerance_;
            int min_cluster_size_;
            int max_cluster_size_;
            double through_th_z_min_;
            double through_th_z_max_;

            bool pc_get_flag_ = false;

            std::vector<std::string> class_names_;
            std::vector<cv::Scalar> colors_;
            std::vector<DetectInfo> detect_list_;

            cv::Mat input_image_;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_pc_ {new pcl::PointCloud<pcl::PointXYZRGB>};
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pcs_ {new pcl::PointCloud<pcl::PointXYZRGB>};
            pcl::PointCloud<pcl::PointXYZ>::Ptr centroids_ {new pcl::PointCloud<pcl::PointXYZ>};
            camera_apps_msgs::ObjectStates object_states_;
            // ros::Time msg_stamp_;
            cv::Mat detection_image_;
            cv::dnn::Net net_;
            camera_apps_msgs::BoundingBoxes bboxes_;

            // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
            
            // message_filters::Subscriber<sensor_msgs::Image> *image_sub_;
            // message_filters::Subscriber<sensor_msgs::PointCloud2> *pc_sub_;
            // message_filters::Synchronizer<MySyncPolicy> *sync_;

            image_transport::Subscriber image_sub_;
            ros::Subscriber pc_sub_;
            image_transport::Publisher image_pub_;
            ros::Publisher object_pc_pub_;
            ros::Publisher centroids_pub_;
            ros::Publisher object_states_pub_;
            ros::Publisher bboxes_pub_;

            tf2_ros::Buffer tf_buffer_;
            tf2_ros::TransformListener* tf2_listener_;
    };
}
#endif
