#ifndef MOTION_PREDICTOR
#define MOTION_PREDICTOR

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>

#include <Eigen/Dense>

#include <camera_apps_msgs/ObjectState.h>
#include <camera_apps_msgs/ObjectStates.h>
#include <camera_apps_msgs/ObjectInfo.h>
#include <camera_apps_msgs/ObjectsInfo.h>

#include "hungarian.h"

namespace camera_apps
{
    struct PersonInfo
    {
        int id;
        ros::Time latest_time;
        geometry_msgs::PointStamped centroid;
        geometry_msgs::PoseStamped filtered_pose;
        nav_msgs::Path trajectory;
        nav_msgs::Path filtered_trajectory;
        nav_msgs::Path future_trajectory;
        Eigen::VectorXd X;
        Eigen::MatrixXd P;
        Eigen::MatrixXd K;
        std::vector<float> color_hist;
        
        bool update_flag; 
    };

    class MotionPredictor
    {
        public:
            MotionPredictor(ros::NodeHandle &nh, ros::NodeHandle &pnh);
            ~MotionPredictor();
        private:
            void object_states_callback(const camera_apps_msgs::ObjectStatesConstPtr &msg);
            void register_person(camera_apps_msgs::ObjectState& object_state);
            void update_person(int id, camera_apps_msgs::ObjectState& object_state);
            void update_trajectory(nav_msgs::Path& trajectory, geometry_msgs::PointStamped centroid);
            // void update_trajectory(nav_msgs::Path& trajectory, geometry_msgs::PoseStamped filtered_pose);
            void visualize_trajectory();
            void visualize_trajectory2();
            void visualize_filtered_trajectory();
            void visualize_future_trajectory();
            void visualize_filtered_pose();
            void delete_person(int id);
            void lost_judge();
            int id_to_index(int id);
            double adjust_yaw(double yaw);
            Eigen::VectorXd adjust_X(Eigen::VectorXd X);
            geometry_msgs::PoseStamped create_pose_from_X(Eigen::MatrixXd);
            void set_invariable_matrix();
            Eigen::VectorXd calculate_X_hat(Eigen::VectorXd X, double dt);
            Eigen::MatrixXd calculate_F(Eigen::VectorXd X, double dt);
            Eigen::MatrixXd calculate_P_hat(Eigen::VectorXd X, Eigen::MatrixXd P, double dt);
            Eigen::VectorXd update_X(Eigen::VectorXd X_hat, Eigen::MatrixXd K, double Z_x, double Z_y);
            Eigen::MatrixXd update_P(Eigen::MatrixXd K, Eigen::MatrixXd P_hat);
            Eigen::MatrixXd update_K(Eigen::MatrixXd P_hat, double Z_x, double Z_y);
            double calculate_euclidean_distance(PersonInfo registered_info, geometry_msgs::PointStamped input_centroid);
            double calculate_mahalanobis_distance(PersonInfo registered_info, geometry_msgs::PointStamped input_centroid);
            void calculate_future_trajectory(PersonInfo& person_info);
            void add_to_tracking_result(PersonInfo person_info);
            
            std::vector<std::vector<double>> create_cost_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list);
            std::vector<std::vector<double>> create_hist_cost_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list);

            std::vector<std::vector<double>> create_transpose_mat(std::vector<std::vector<double>> mat);
            std::vector<std::vector<double>> create_integrated_cost_mat(std::vector<std::vector<double>>& dist_cost_mat, std::vector<std::vector<double>>& hist_cost_mat);
            std::vector<std::vector<double>> create_velocity_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list);
        
            std::vector<int> create_map(std::vector<std::vector<double>> cost_mat);
            double compare_hist(std::vector<float> hist1, std::vector<float> hist2);
            // bool is_duplicate(camera_apps_msgs::ObjectState& object_state);
            bool is_duplicate(int index);


            double error_threshold_;
            double hist_similarity_threshold_;
            double time_threshold_;
            int past_path_threshold_;
            int person_num_limit_;
            bool colorful_trajectory_flag_;
            double observation_noise_ratio_;
            double trajectory_z_;
            int data_num_th_visualize_;
            int data_num_th_mahalanovis_;
            double predict_time_;
            double predict_dt_;
            bool calc_future_trajectory_flag_;
            bool visualize_future_trajectory_flag_;
            bool visualize_past_trajectory_flag_;
            double register_th_;
            double duplicate_th_;
            double velocity_th_;
            double dist_cost_ratio_;
            double hist_cost_ratio_;

            double sigma_initial_P_theta_;
            double sigma_initial_P_velocity_;
            double sigma_initial_P_omega_;
            double sigma_Q_x_;
            double sigma_Q_y_;
            double sigma_Q_theta_;
            double sigma_Q_velocity_;
            double sigma_Q_omega_;

            int id_now_ = 0;

            std::vector<int> free_id_list_;
            std::vector<int> valid_id_list_;
            std::vector<PersonInfo> person_list_;
            camera_apps_msgs::ObjectStates object_states_;
            camera_apps_msgs::ObjectsInfo tracking_result_;

            Eigen::MatrixXd Q_;
            Eigen::MatrixXd H_;

            ros::Subscriber object_states_sub_;
            ros::Publisher past_trajectory_pub_;
            ros::Publisher filtered_past_trajectory_pub_;
            ros::Publisher future_trajectory_pub_;
            ros::Publisher filtered_pose_pub_;
            ros::Publisher filtered_pose_array_pub_;
            ros::Publisher tracking_result_pub_;

            ros::Publisher past_trajectory_pub1_;
            ros::Publisher past_trajectory_pub2_;
            ros::Publisher past_trajectory_pub3_;
            ros::Publisher past_trajectory_pub4_;
            ros::Publisher past_trajectory_pub5_;

            tf2_ros::Buffer tf_buffer_;
            tf2_ros::TransformListener* tf2_listener_;
    };
}


#endif
