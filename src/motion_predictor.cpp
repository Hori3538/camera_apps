#include <motion_predictor/motion_predictor.h>

namespace camera_apps
{
    MotionPredictor::MotionPredictor(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("error_threshold", error_threshold_, 0.5);
        pnh.param("hist_similarity_threshold", hist_similarity_threshold_, 0.3);
        pnh.param("time_threshold", time_threshold_, 2.0);
        pnh.param("past_path_threshold", past_path_threshold_, 50);
        pnh.param("person_num_limit", person_num_limit_, 10);
        pnh.param("colorful_trajectory_flag", colorful_trajectory_flag_, false);
        pnh.param("observation_noise_ratio", observation_noise_ratio_, 0.05);
        pnh.param("sigma_initial_P_theta", sigma_initial_P_theta_, 2 * M_PI);
        pnh.param("sigma_initial_P_velocity", sigma_initial_P_velocity_, 3.0);
        pnh.param("sigma_initial_P_omega", sigma_initial_P_omega_, M_PI);
        pnh.param("sigma_Q_x", sigma_Q_x_, 1.0);
        pnh.param("sigma_Q_y", sigma_Q_y_, 1.0);
        pnh.param("sigma_Q_theta", sigma_Q_theta_, M_PI);
        pnh.param("sigma_Q_velocity", sigma_Q_velocity_, 3.0);
        pnh.param("sigma_Q_omega", sigma_Q_omega_, M_PI);
        pnh.param("trajectory_z", trajectory_z_, 0.0);
        pnh.param("data_num_th_visualize", data_num_th_visualize_, 2);
        pnh.param("data_num_th_mahalanovis", data_num_th_mahalanovis_, 2);
        pnh.param("predict_time", predict_time_, 2.0);
        pnh.param("predict_dt", predict_dt_, 0.1);
        pnh.param("calc_future_trajectory_flag", calc_future_trajectory_flag_, false);
        pnh.param("visualize_future_trajectory_flag", visualize_future_trajectory_flag_, false);
        pnh.param("visualize_past_trajectory_flag", visualize_past_trajectory_flag_, true);
        pnh.param("register_th", register_th_, 0.9);
        pnh.param("duplicate_th", duplicate_th_, 0.2);
        pnh.param("velocity_th", velocity_th_, 5.0);
        pnh.param("dist_cost_ratio", dist_cost_ratio_, 1.0);
        pnh.param("hist_cost_ratio", hist_cost_ratio_, 1.0);

        object_states_sub_ = nh.subscribe("/object_states", 5, &MotionPredictor::object_states_callback, this);
        past_trajectory_pub_ = nh.advertise<nav_msgs::Path>("/past_trajectory", 20);
        filtered_past_trajectory_pub_ = nh.advertise<nav_msgs::Path>("/filtered_past_trajectory", 20);
        future_trajectory_pub_ = nh.advertise<nav_msgs::Path>("/future_trajectory", 20);
        filtered_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/filtered_pose", 1);
        filtered_pose_array_pub_ = nh.advertise<geometry_msgs::PoseArray>("/filtered_pose_array", 1);
        tracking_result_pub_ = nh.advertise<camera_apps_msgs::ObjectsInfo>("/tracking_result", 1);

        past_trajectory_pub1_ = nh.advertise<nav_msgs::Path>("/past_trajectory1", 10);
        past_trajectory_pub2_ = nh.advertise<nav_msgs::Path>("/past_trajectory2", 10);
        past_trajectory_pub3_ = nh.advertise<nav_msgs::Path>("/past_trajectory3", 10);
        past_trajectory_pub4_ = nh.advertise<nav_msgs::Path>("/past_trajectory4", 10);
        past_trajectory_pub5_ = nh.advertise<nav_msgs::Path>("/past_trajectory5", 10);
        
        tf2_listener_ = new tf2_ros::TransformListener(tf_buffer_);

        for(int i=0; i<person_num_limit_; i++) free_id_list_.push_back(i);

        set_invariable_matrix();
    }

    MotionPredictor::~MotionPredictor()
    {
        delete this->tf2_listener_;
    }

    // void MotionPredictor::object_states_callback(const camera_apps_msgs::ObjectStatesConstPtr &msg)
    // {
    //     object_states_ = *msg;
    //
    //     tracking_result_.objects_info.clear();
    //     tracking_result_.header =  object_states_.header;
    //
    //     for(auto& object_state: object_states_.object_states){
    //         int best_id = -1;
    //         double best_error = 1000000000;
    //         for(const auto& id: valid_id_list_){
    //             int index = id_to_index(id);
    //             double mahalanobis_error = calculate_mahalanobis_distance(person_list_[index], object_state.centroid);
    //             double error = calculate_euclidean_distance(person_list_[index], object_state.centroid);
    //             if(error > mahalanobis_error) error = mahalanobis_error;
    //             // if(person_list_[index].trajectory.poses.size() >= data_num_th_mahalanovis_){
    //             //     error = mahalanobis_error;
    //             // }
    //             if(error < best_error){
    //                 best_id = id;
    //                 best_error = error;
    //             }
    //
    //         }
    //         if(best_error > error_threshold_){
    //             if(free_id_list_.size() != 0){
    //                 register_person(object_state);
    //             }
    //             else std::cout << "cannot add more person" << std::endl;
    //         }
    //         else{
    //             update_person(best_id, object_state);
    //         }
    //     }
    //     lost_judge();
    //     if(!colorful_trajectory_flag_){
    //         if(visualize_past_trajectory_flag_) visualize_trajectory();
    //     }
    //     else{
    //         visualize_trajectory2();
    //     }
    //     // visualize_trajectory2();
    //     visualize_filtered_trajectory();
    //     if(visualize_future_trajectory_flag_) visualize_future_trajectory();
    //     visualize_filtered_pose();
    //
    //     for(auto& person_info: person_list_){
    //         if(person_info.update_flag){
    //             add_to_tracking_result(person_info);
    //             person_info.update_flag = false;
    //         }
    //     }
    //     tracking_result_pub_.publish(tracking_result_);
    // }

    void MotionPredictor::object_states_callback(const camera_apps_msgs::ObjectStatesConstPtr &msg)
    {
        object_states_ = *msg;
        if(object_states_.object_states.size() == 0) return;

        tracking_result_.objects_info.clear();
        tracking_result_.header =  object_states_.header;

        std::vector<std::vector<double>> dist_cost_mat = create_cost_mat(object_states_, person_list_);

        // std::vector<std::vector<double>> hist_cost_mat = create_hist_cost_mat(object_states_, person_list_);
        // std::vector<std::vector<double>> integrated_cost_mat = create_integrated_cost_mat(dist_cost_mat, hist_cost_mat);
        // std::vector<std::vector<double>> velocity_mat = create_velocity_mat(object_states_, person_list_);
        std::vector<int> map = create_map(dist_cost_mat); 
        // std::vector<int> map = create_map(integrated_cost_mat); 
        for(int i=0; i<map.size(); i++){
            if(map[i] == -1){
                if(object_states_.object_states[i].confidence >= register_th_ && !is_duplicate(i)){
                    // std::cout << "register" << std::endl;
                    register_person(object_states_.object_states[i]);
                }
            }
            else if(dist_cost_mat[i][map[i]] < error_threshold_ && !is_duplicate(i)){
            // else if(dist_cost_mat[i][map[i]] < error_threshold_ || (hist_cost_mat[i][map[i]] < hist_similarity_threshold_ && velocity_mat[i][map[i]] <= velocity_th_)){
                // std::cout << "dist error " << dist_cost_mat[i][map[i]] << std::endl;
                // std::cout << "update" << std::endl;
                update_person(person_list_[map[i]].id, object_states_.object_states[i]);
            }
            else{
                if(object_states_.object_states[i].confidence >= register_th_ && !is_duplicate(i)){
                    // std::cout << "register2" << std::endl;
                    register_person(object_states_.object_states[i]);
                }
            }
        }

        lost_judge();
        if(!colorful_trajectory_flag_){
            if(visualize_past_trajectory_flag_) visualize_trajectory();
        }
        else{
            visualize_trajectory2();
        }
        // visualize_trajectory2();
        visualize_filtered_trajectory();
        if(visualize_future_trajectory_flag_) visualize_future_trajectory();
        visualize_filtered_pose();

        // tracking_result_pub_.publish(tracking_result_);
        for(auto& person_info: person_list_){
            if(person_info.update_flag){
                add_to_tracking_result(person_info);
                person_info.update_flag = false;
            }
        }
        tracking_result_pub_.publish(tracking_result_);
    }

    void MotionPredictor::register_person(camera_apps_msgs::ObjectState& object_state)
    {
        PersonInfo new_person;
        int new_id = free_id_list_[0];
        free_id_list_.erase(free_id_list_.begin());

        new_person.id = new_id;
        new_person.centroid = object_state.centroid;
        new_person.filtered_pose.pose.position = object_state.centroid.point;
        new_person.filtered_pose.header = object_state.centroid.header;
        new_person.latest_time = object_state.centroid.header.stamp;

        nav_msgs::Path trajectory;
        // trajectory.poses.clear();
        trajectory.header = object_state.centroid.header;
        update_trajectory(trajectory, object_state.centroid);
        new_person.trajectory = trajectory;
        new_person.filtered_trajectory.header = trajectory.header;
        // new_person.filtered_trajectory = trajectory;

        new_person.update_flag = true;

        Eigen::VectorXd X(5);
        X(0) = object_state.centroid.point.x;
        X(1) = object_state.centroid.point.y;
        X(2) = 0;
        X(3) = 0;
        X(4) = 0;
        new_person.X = X;

        Eigen::MatrixXd P(5,5);
        P.setZero();
        P(0,0) = X(0) * observation_noise_ratio_;
        P(1,1) = X(1) * observation_noise_ratio_;
        P(2,2) = sigma_initial_P_theta_;
        P(3,3) = sigma_initial_P_velocity_;
        P(4,4) = sigma_initial_P_omega_;
        P *= P;
        new_person.P = P;

        new_person.color_hist = object_state.color_hist;

        person_list_.push_back(new_person);
        valid_id_list_.push_back(new_id);
        // add_to_tracking_result(new_person);
    }

    void MotionPredictor::update_person(int id, camera_apps_msgs::ObjectState& object_state)
    {
        int index = id_to_index(id);
        person_list_[index].centroid = object_state.centroid;
        double dt = (object_state.centroid.header.stamp - person_list_[index].latest_time).nsec
            / std::pow(10, 9);
        person_list_[index].latest_time = object_state.centroid.header.stamp;

        update_trajectory(person_list_[index].trajectory, object_state.centroid);

        Eigen::MatrixXd X_hat = calculate_X_hat(person_list_[index].X, dt);
        Eigen::MatrixXd P_hat = calculate_P_hat(person_list_[index].X, person_list_[index].P, dt);
        person_list_[index].X = X_hat;
        person_list_[index].P = P_hat;

        double Z_x = object_state.centroid.point.x;
        double Z_y = object_state.centroid.point.y;
        Eigen::MatrixXd K = update_K(P_hat, Z_x, Z_y);
        person_list_[index].K = K;
        person_list_[index].X = update_X(X_hat, K, Z_x, Z_y);
        person_list_[index].P = update_P(K, P_hat);

        geometry_msgs::PoseStamped filtered_pose = create_pose_from_X(person_list_[index].X);
        filtered_pose.header = object_state.centroid.header;

        double dist = std::sqrt(std::pow(person_list_[index].filtered_pose.pose.position.x - filtered_pose.pose.position.x, 2) + std::pow(person_list_[index].filtered_pose.pose.position.y - filtered_pose.pose.position.y, 2));
        // std::cout << "traj gap: " << dist << std::endl;
        if(dist > 1.0){
            // std::cout << "data num: " << person_list_[index].trajectory.poses.size() << std::endl;
        }

        person_list_[index].filtered_pose = filtered_pose;
        person_list_[index].filtered_trajectory.poses.push_back(filtered_pose);

        person_list_[index].color_hist = object_state.color_hist;
        person_list_[index].update_flag = true;

        // add_to_tracking_result(person_list_[index]);
        if(calc_future_trajectory_flag_) calculate_future_trajectory(person_list_[index]);

        if(person_list_[index].trajectory.poses.size() > past_path_threshold_){
            person_list_[index].trajectory.poses.erase(person_list_[index].trajectory.poses.begin());
            person_list_[index].filtered_trajectory.poses.erase(person_list_[index].filtered_trajectory.poses.begin());
        }
    }

    void MotionPredictor::update_trajectory(nav_msgs::Path& trajectory, geometry_msgs::PointStamped centroid)
    {
        geometry_msgs::PoseStamped pose;
        pose.header = centroid.header;
        pose.pose.position.x = centroid.point.x;
        pose.pose.position.y = centroid.point.y;
        pose.pose.position.z = trajectory_z_;
        trajectory.poses.push_back(pose);
    }

    void MotionPredictor::delete_person(int id)
    {
        int index = id_to_index(id);
        for(int i=0; i<valid_id_list_.size(); i++){
            if(valid_id_list_[i] == id){
                // std::cout << "delete id: " << id << std::endl;
                // person_list_[index].trajectory.poses.clear();
                person_list_.erase(person_list_.begin() + index);
                valid_id_list_.erase(valid_id_list_.begin() + i);
                free_id_list_.push_back(id);

                return;
            }
        }
    }

    void MotionPredictor::lost_judge()
    {
        ros::Time current_time = object_states_.header.stamp; 
        for(const auto& id: valid_id_list_){
            int index = id_to_index(id);
            ros::Duration time_blank = current_time - person_list_[index].latest_time;
            if(std::abs(time_blank.sec) > time_threshold_) delete_person(id);
        }
    }
    
    int MotionPredictor::id_to_index(int id)
    {
        for(int index=0; index<person_list_.size(); index++){
            if(person_list_[index].id == id) return index;
        }
        return -1;
    }

    void MotionPredictor::visualize_trajectory()
    {
        for(const auto& person_info: person_list_){
            if(person_info.trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub_.publish(person_info.trajectory);
            }
        }
    }
    void MotionPredictor::visualize_trajectory2()
    {
        // for(const auto& person_info: person_list_){
        //     past_trajectory_pub_.publish(person_info.trajectory);
        // }
        int limit = 5;
        // if(limit > person_list_.size()) limit = person_list_.size();
        int i=0;
        if(i>limit-1) return;
        // if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub1_.publish(person_list_[id_to_index(i)].filtered_trajectory);
        // }
        if(id_to_index(i) != -1){
            if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub1_.publish(person_list_[id_to_index(i)].filtered_trajectory);
            }
        }
        // past_trajectory_pub1_.publish(person_list_[0].trajectory);
        i++;
        if(i>limit-1) return;
        // if(person_list_[1].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub2_.publish(person_list_[1].filtered_trajectory);
        // }
        if(id_to_index(i) != -1){
            if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub2_.publish(person_list_[id_to_index(i)].filtered_trajectory);
            }
        }
        // past_trajectory_pub2_.publish(person_list_[1].trajectory);
        i++;
        if(i>limit-1) return;
        // if(person_list_[2].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub3_.publish(person_list_[2].filtered_trajectory);
        // }
        // if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub3_.publish(person_list_[id_to_index(i)].filtered_trajectory);
        // }
        if(id_to_index(i) != -1){
            if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub3_.publish(person_list_[id_to_index(i)].filtered_trajectory);
            }
        }
        // past_trajectory_pub3_.publish(person_list_[2].trajectory);
        i++;
        if(i>limit-1) return;
        // if(person_list_[3].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub4_.publish(person_list_[3].filtered_trajectory);
        // }
        // if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub4_.publish(person_list_[id_to_index(i)].filtered_trajectory);
        // }
        if(id_to_index(i) != -1){
            if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub4_.publish(person_list_[id_to_index(i)].filtered_trajectory);
            }
        }
        // past_trajectory_pub4_.publish(person_list_[3].trajectory);
        i++;
        if(i>limit-1) return;
        // if(person_list_[4].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub5_.publish(person_list_[4].filtered_trajectory);
        // }
        // if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
        //     past_trajectory_pub5_.publish(person_list_[id_to_index(i)].filtered_trajectory);
        // }
        if(id_to_index(i) != -1){
            if(person_list_[id_to_index(i)].filtered_trajectory.poses.size() >= data_num_th_visualize_){
                past_trajectory_pub5_.publish(person_list_[id_to_index(i)].filtered_trajectory);
            }
        }
        // past_trajectory_pub5_.publish(person_list_[4].trajectory);
    }

    void MotionPredictor::visualize_filtered_trajectory()
    {
        for(const auto& person_info: person_list_){
            if(person_info.filtered_trajectory.poses.size() >= data_num_th_visualize_){

                filtered_past_trajectory_pub_.publish(person_info.filtered_trajectory);
            }
        }
    }

    void MotionPredictor::visualize_future_trajectory()
    {
        for(const auto& person_info: person_list_){
            if(person_info.future_trajectory.poses.size() >= 1){
                future_trajectory_pub_.publish(person_info.future_trajectory);
            }
        }

    }

    void MotionPredictor::visualize_filtered_pose()
    {
        geometry_msgs::PoseArray filtered_pose_array;
        if(person_list_.size() == 0) return;
        filtered_pose_array.header.frame_id = person_list_[0].trajectory.header.frame_id;
        for(const auto& person_info: person_list_){
            if(person_info.filtered_trajectory.poses.size() >= data_num_th_visualize_){
                filtered_pose_array.poses.push_back(person_info.filtered_pose.pose);
            }
        }
        filtered_pose_array_pub_.publish(filtered_pose_array);
    }

    double MotionPredictor::adjust_yaw(double yaw)
    {
        if(yaw > M_PI){yaw -= 2*M_PI;}
        if(yaw < -M_PI){yaw += 2*M_PI;}

        return yaw;
    }

    Eigen::VectorXd MotionPredictor::adjust_X(Eigen::VectorXd X)
    {
        Eigen::VectorXd X_out = X;
        if(X_out(3) < 0){
            X_out(3) *= -1;
            X_out(2) = adjust_yaw(X_out(2) + M_PI);
        }
        return X_out;
    }

    geometry_msgs::PoseStamped MotionPredictor::create_pose_from_X(Eigen::MatrixXd X)
    {
        geometry_msgs::PoseStamped pose;
        X = adjust_X(X);
        pose.pose.position.x = X(0);
        pose.pose.position.y = X(1);
        pose.pose.position.z = trajectory_z_;
        quaternionTFToMsg(tf::createQuaternionFromYaw(X(2)), pose.pose.orientation);

        return pose;
    }

    void MotionPredictor::set_invariable_matrix()
    {
        // Eigen::MatrixXd Q(5,5);
        // Q.setZero();
        // Q(0,0) = sigma_Q_x_;
        // Q(1,1) = sigma_Q_y_;
        // Q(2,2) = sigma_Q_theta_;
        // Q(3,3) = sigma_Q_velocity_;
        // Q(4,4) = sigma_Q_omega_;
        // Q_ = Q;

        Eigen::MatrixXd H(2,5);
        H.setIdentity();
        H_ = H;
    }

    Eigen::VectorXd MotionPredictor::calculate_X_hat(Eigen::VectorXd X, double dt)
    {
        Eigen::VectorXd X_hat(5);
        X_hat(0) = X(0) + X(3) * dt * std::cos(X(2) + X(4) * dt/2);
        X_hat(1) = X(1) + X(3) * dt * std::sin(X(2) + X(4) * dt/2);
        X_hat(2) = adjust_yaw(X(2) + X(4) * dt);
        X_hat(3) = X(3);
        X_hat(4) = X(4);

        return X_hat;
    }

    Eigen::MatrixXd MotionPredictor::calculate_F(Eigen::VectorXd X, double dt)
    {
        Eigen::MatrixXd F(5,5);
        F.setIdentity();
        F(0,2) = -X(3) * dt * std::sin(X(2) + X(4) * dt/2);
        F(0,3) = dt * std::cos(X(2) + X(4) * dt/2);
        F(0,4) = -X(3) * dt*dt/2 * std::sin(X(2) + X(4) * dt/2);
        F(1,2) = X(3) * dt * std::cos(X(2) + X(4) * dt/2);
        F(1,3) = dt * std::sin(X(2) + X(4) * dt/2);
        F(1,4) = X(3) * dt*dt/2 * std::cos(X(2) + X(4) * dt/2);
        F(2,4) = dt;

        return F;
    }

    Eigen::MatrixXd MotionPredictor::calculate_P_hat(Eigen::VectorXd X, Eigen::MatrixXd P, double dt)
    {
        Eigen::MatrixXd Q(5,5);
        Q.setZero();
        Q(0,0) = sigma_Q_x_ * dt;
        Q(1,1) = sigma_Q_y_ * dt;
        Q(2,2) = sigma_Q_theta_ * dt;
        Q(3,3) = sigma_Q_velocity_ * dt;
        Q(4,4) = sigma_Q_omega_ * dt;
        Q *= Q;
        Eigen::MatrixXd P_hat;
        Eigen::MatrixXd F = calculate_F(X, dt);
        P_hat = F * P * F.transpose() + Q;

        return P_hat;
    }
    Eigen::VectorXd MotionPredictor::update_X(Eigen::VectorXd X_hat, Eigen::MatrixXd K, double Z_x, double Z_y)
    {
        Eigen::Vector2d Z(Z_x,Z_y);
        Eigen::VectorXd X;
        X = X_hat + K * (Z - H_ * X_hat);
        X(2) = adjust_yaw(X(2));

        return X;
    }

    Eigen::MatrixXd MotionPredictor::update_P(Eigen::MatrixXd K, Eigen::MatrixXd P_hat)
    {
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(5,5);
        Eigen::MatrixXd P;
        P = (I - K * H_) * P_hat;

        return P;
    }

    Eigen::MatrixXd MotionPredictor::update_K(Eigen::MatrixXd P_hat, double Z_x, double Z_y)
    {
        Eigen::MatrixXd R(2,2);
        R.setZero();
        R(0,0) = Z_x * observation_noise_ratio_;
        R(1,1) = Z_y * observation_noise_ratio_;
        R *= R;

        Eigen::MatrixXd K = P_hat * H_.transpose() * (H_ * P_hat * H_.transpose() + R).inverse();

        return K;
    }

    double MotionPredictor::calculate_euclidean_distance(PersonInfo registered_info, geometry_msgs::PointStamped input_centroid)
    {
        double registered_x = registered_info.centroid.point.x;
        double registered_y = registered_info.centroid.point.y;
        double input_x = input_centroid.point.x;
        double input_y = input_centroid.point.y;

        double dist = std::sqrt(std::pow(input_x - registered_x, 2) + std::pow(input_y - registered_y, 2));

        return dist;
    }

    double MotionPredictor::calculate_mahalanobis_distance(PersonInfo registered_info, geometry_msgs::PointStamped input_centroid)
    {
        double dt = (input_centroid.header.stamp - registered_info.latest_time).nsec / std::pow(10, 9);
        // std::cout << "dt " << dt << std::endl;
        Eigen::MatrixXd X = registered_info.X;
        // std::cout << "X " << X << std::endl;
        Eigen::MatrixXd P = registered_info.P;
        Eigen::MatrixXd X_hat = calculate_X_hat(X, dt);
        // std::cout << "X hat " << X_hat << std::endl;
        Eigen::MatrixXd P_hat = calculate_P_hat(X, P, dt);
        // std::cout << "P hat " << P_hat << std::endl;

        Eigen::Vector2d Z_hat(X_hat(0), X_hat(1));
        // std::cout << "Z hat " << Z_hat << std::endl;
        Eigen::Vector2d Z(input_centroid.point.x, input_centroid.point.y);
        // std::cout << "Z " << Z << std::endl;
        Eigen::MatrixXd sigma = P_hat.block(0, 0, 2, 2);
        // std::cout << "sigma " << sigma << std::endl;

        double dist = std::sqrt((Z - Z_hat).transpose() * sigma.inverse() * (Z - Z_hat));
        double dist2 = std::sqrt((Z_hat - Z).transpose() * sigma.inverse() * (Z_hat - Z));
        // std::cout << "mahalanobis dist " << dist << std::endl;

        return dist;
    }

    void MotionPredictor::calculate_future_trajectory(PersonInfo& person_info)
    {
        nav_msgs::Path future_trajectory;
        future_trajectory.header.frame_id = person_info.trajectory.header.frame_id;

        Eigen::MatrixXd X = person_info.X;
        geometry_msgs::PoseStamped pose = create_pose_from_X(X);
        pose.header = person_info.filtered_pose.header;
        future_trajectory.poses.push_back(pose);

        for(double t=predict_dt_; t<=predict_time_; t+=predict_dt_){
            X = calculate_X_hat(X, predict_dt_);
            pose = create_pose_from_X(X);
            pose.header.frame_id = person_info.filtered_pose.header.frame_id;
            pose.header.stamp = person_info.latest_time + ros::Duration(t);
            future_trajectory.poses.push_back(pose);
        }
        person_info.future_trajectory = future_trajectory;
    }

    void MotionPredictor::add_to_tracking_result(PersonInfo person_info)
    {
        camera_apps_msgs::ObjectInfo object_info;
        object_info.id = person_info.id;
        object_info.centroid.x = person_info.X(0);
        object_info.centroid.y = person_info.X(1);
        // object_info.centroid = person_info.centroid.point;
        
        tracking_result_.objects_info.push_back(object_info);
    }
    std::vector<std::vector<double>> MotionPredictor::create_cost_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list)
    {
        std::vector<std::vector<double>> cost_mat;
        for(const auto& object_state: object_states.object_states){
            std::vector<double> cost_row;
            for(const auto& person_info: person_list){
                double mahalanobis_error = calculate_mahalanobis_distance(person_info, object_state.centroid);
                double error = calculate_euclidean_distance(person_info, object_state.centroid);
                // std::cout << "eu dist: " << error << " maha dist: " << mahalanobis_error << std::endl;
                if(error > mahalanobis_error) error = mahalanobis_error;
                cost_row.push_back(error);
            }
            cost_mat.push_back(cost_row);
        }
        // std::cout << "dist mat" << std::endl;
        // for(const auto& row: cost_mat){
        //     for(const auto& num: row){
        //         std::cout << std::setw(5) << num << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        return cost_mat;
    }

    std::vector<std::vector<double>> MotionPredictor::create_hist_cost_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list)
    {
        std::vector<std::vector<double>> cost_mat;
        for(const auto& object_state: object_states.object_states){
            std::vector<double> cost_row;
            for(const auto& person_info: person_list){
                double similarity = compare_hist(person_info.color_hist, object_state.color_hist);
                double error = 1.0 - similarity;
                cost_row.push_back(error);
            }
            cost_mat.push_back(cost_row);
        }
        // std::cout << "hist mat" << std::endl;
        // for(const auto& row: cost_mat){
        //     for(const auto& num: row){
        //         std::cout << std::setw(5) << num << " ";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
        return cost_mat;
    }
    std::vector<std::vector<double>> MotionPredictor::create_transpose_mat(std::vector<std::vector<double>> mat)
    {
        std::vector<std::vector<double>> transpose_mat(mat[0].size(), std::vector<double>(mat.size()));
        for(int i=0; i<mat.size(); i++){
            for(int j=0; j<mat[0].size(); j++){
                transpose_mat[j][i] = mat[i][j];
            }
        }
        return transpose_mat;
    }

    std::vector<int> MotionPredictor::create_map(std::vector<std::vector<double>> cost_mat)
    {

        bool transpose_flag = false;
        if(cost_mat.size() < cost_mat[0].size()){
            cost_mat = create_transpose_mat(cost_mat);
            transpose_flag = true;
        }

        Hungarian<double> hungarian(cost_mat);
        std::vector<int> alloc = hungarian.solve().second;

        std::vector<int> map(object_states_.object_states.size());
        for(int i=0; i<alloc.size(); i++){
            if(!transpose_flag){
                map[i] = alloc[i];
            }
            if(transpose_flag){
                if(alloc[i] != -1){
                    map[alloc[i]] = i;
                }
            }
        }
        // std::cout << "map: ";
        // for(const auto& ma: map){
        //     std::cout << ma  << " ";
        // }
        // std::cout << std::endl;
        return map;
    }

    double MotionPredictor::compare_hist(std::vector<float> hist1, std::vector<float> hist2)
    {
        double similarity;
        for(int i=0; i<hist1.size(); i++){
            similarity += std::min(hist1[i], hist2[i]);
        }
        return similarity;
    }

    // bool MotionPredictor::is_duplicate(camera_apps_msgs::ObjectState& object_state)
    // {
    //    for(const auto& state: object_states_.object_states){
    //        double dist = std::sqrt(std::pow(state.centroid.point.x - object_state.centroid.point.x, 2) + std::pow(state.centroid.point.y - object_state.centroid.point.y, 2));
    //        if(dist <= duplicate_th_){
    //            std::cout << "duplicate!" << std::endl;
    //            return true;
    //        }
    //    } 
    //    return false;
    // }

    bool MotionPredictor::is_duplicate(int index)
    {

        double x_parent = object_states_.object_states[index].centroid.point.x;
        double y_parent = object_states_.object_states[index].centroid.point.y;
        for(int i=0; i<object_states_.object_states.size(); i++){
            if(i == index) continue;
            double x_child = object_states_.object_states[i].centroid.point.x;
            double y_child = object_states_.object_states[i].centroid.point.y;
            double dist = std::sqrt(std::pow(x_parent - x_child, 2) + std::pow(y_parent - y_child, 2));
            if(dist <= duplicate_th_){
               // std::cout << "duplicate!" << std::endl;
               return true;
            }
        }
        return false;
    }

    std::vector<std::vector<double>> MotionPredictor::create_integrated_cost_mat(std::vector<std::vector<double>>& dist_cost_mat, std::vector<std::vector<double>>& hist_cost_mat)
    {
        std::vector<std::vector<double>> integrated_cost_mat(dist_cost_mat.size(), std::vector<double>(dist_cost_mat[0].size()));
        for(int i=0; i<dist_cost_mat.size(); i++){
            for(int j=0; j<dist_cost_mat[0].size(); j++){
                integrated_cost_mat[i][j] = dist_cost_mat[i][j] * dist_cost_ratio_ + 
                    hist_cost_mat[i][j] * hist_cost_ratio_;
            }
        }
        return integrated_cost_mat;
    }
    std::vector<std::vector<double>> MotionPredictor::create_velocity_mat(camera_apps_msgs::ObjectStates& object_states, std::vector<PersonInfo>& person_list)
    {
        std::vector<std::vector<double>> velocity_mat;
        for(const auto& object_state: object_states.object_states){
            std::vector<double> velocity_row;
            for(const auto& person_info: person_list){
                double dt = (object_state.centroid.header.stamp - person_info.latest_time).nsec / std::pow(10, 9);
                double dist = calculate_euclidean_distance(person_info, object_state.centroid);
                double velocity = dist / dt;
                velocity_row.push_back(dt);
                // std::cout << "velo: " << velocity << std::endl;
            }
            velocity_mat.push_back(velocity_row);
        }
        return velocity_mat;
    }
}

