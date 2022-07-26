#include <person_recognizer/person_recognizer.h>

namespace camera_apps
{
    PersonRecognizer::PersonRecognizer(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("camera_topic_name", camera_topic_name_, std::string("/camera/color/image_raw"));
        pnh.getParam("model_path", model_path_);
        pnh.param("conf_threshold", conf_threshold_, 0.4);
        pnh.param("mask_threshold", mask_threshold_, 0.4);
        pnh.param("points_limit", points_limit_, 10000);
        pnh.param("cluster_tolerance", cluster_tolerance_, 0.02);
        pnh.param("min_cluster_size", min_cluster_size_, 1000);
        pnh.param("max_cluster_size", max_cluster_size_, 20000);
        pnh.param("through_th_z_min", through_th_z_min_, -70.0);
        pnh.param("through_th_z_max", through_th_z_max_, 2.0);
        
        image_transport::ImageTransport it(nh);
        image_sub_ = it.subscribe(camera_topic_name_, 1, &PersonRecognizer::image_callback, this);
        pc_sub_ = nh.subscribe("/camera/depth_registered/points", 1, &PersonRecognizer::pc_callback, this);

        image_pub_ = it.advertise("/detected_image", 1);
        object_pc_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object_state/object_pc", 1);
        centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object_state/centroids", 1);
        object_states_pub_ = nh.advertise<camera_apps_msgs::ObjectStates>("/object_states", 1);
        bboxes_pub_ = nh.advertise<camera_apps_msgs::BoundingBoxes>("/bounding_boxes", 1);

        tf2_listener_ = new tf2_ros::TransformListener(tf_buffer_);
        set_network();
    }

    PersonRecognizer::~PersonRecognizer()
    {
        delete this->tf2_listener_;
    }

    void PersonRecognizer::image_callback(const sensor_msgs::ImageConstPtr &msg)
    {

        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            input_image_ = cv_ptr->image.clone();
            bboxes_.header.stamp = msg->header.stamp;
            // msg_stamp_ = msg->header.stamp;
            // object_detect(cv_ptr->image);
            // cv::imshow("input image", input_image_);
            // int key = cv::waitKey(5);
        }
        catch(cv_bridge::Exception &e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        if(pc_get_flag_){
            pcl_conversions::fromPCL(input_pc_->header, object_states_.header);
            object_states_.object_states.clear();
            centroids_->points.clear();
            object_pcs_->points.clear();
            object_pcs_->header = input_pc_->header;

            object_detect(cv_ptr->image);
            for(const auto& detect_info: detect_list_){
                camera_apps_msgs::ObjectState object_state;
                object_state.label = detect_info.label;
                object_state.confidence = detect_info.conf;

                cv::Mat hist = calc_hist(detect_info.roi, detect_info.mask);
                std::vector<float> hist_vec = convert_hist_to_vector(hist);
                object_state.color_hist = hist_vec;

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc {new pcl::PointCloud<pcl::PointXYZRGB>};
                cv::Rect roi = detect_info.roi;
                cv::Mat mask = detect_info.mask;
                create_object_pc(object_pc, roi, mask);
                through_filtering(object_pc);
                downsampling(object_pc);
                if(!clustering(object_pc)){
                    // std::cout << "cannot clustering!!" << std::endl;
                    continue;
                }

                for(const auto& point: object_pc->points) object_pcs_->points.push_back(point);
                geometry_msgs::PointStamped centroid = caluculate_centroid(object_pc);
                object_state.centroid = centroid;

                object_states_.object_states.push_back(object_state);

            }
            object_pc_pub_.publish(object_pcs_);
            centroids_pub_.publish(centroids_);
            object_states_pub_.publish(object_states_);
        }
    }

    void PersonRecognizer::pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        try{
            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_.lookupTransform("camera_fixed_frame", "camera_color_optical_frame", ros::Time(0));
            Eigen::Matrix4f mat = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();

            sensor_msgs::PointCloud2 msg_temp;
            pcl_ros::transformPointCloud(mat, *msg, msg_temp);
            pcl::fromROSMsg(msg_temp, *input_pc_);
            // if(input_pc_->points.size() != 480 * 640){
            //     std::cout << "pc_size is wrong" << std::endl;
            // }
            input_pc_->header.frame_id = "camera_fixed_frame";
        }
        catch(tf2::TransformException &ex){
            ROS_WARN("%s", ex.what());
            return;
        }
        pc_get_flag_ = true;
    }

    std::vector<std::string> PersonRecognizer::read_file(std::string filename, char delimiter)
    {
        std::vector<std::string> result;
        std::ifstream fin(filename);
        std::string line;
        while (getline(fin, line)) {
            std::istringstream stream(line);
            std::string field;
            while (getline(stream, field, delimiter)) {
                result.push_back(field);
            }
        }
        fin.close();
        return result;
    }

    void PersonRecognizer::set_network()
    {
        std::string proto_path = model_path_ + "/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
        std::string weight_path = model_path_ + "/frozen_inference_graph.pb";
        std::string label_path = model_path_ + "/object_detection_classes_coco.txt";

        net_ = cv::dnn::readNet(proto_path, weight_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        class_names_ = read_file(label_path);

        std::string colorsFile = model_path_ + "/colors.txt";
        std::ifstream colorFptr(colorsFile.c_str());
        std::string line;
        while (getline(colorFptr, line)) {
            char* pEnd;
            double r, g, b;
            r = strtod (line.c_str(), &pEnd);
            g = strtod (pEnd, NULL);
            b = strtod (pEnd, NULL);
            cv::Scalar color = cv::Scalar(r, g, b, 255.0);
            colors_.push_back(cv::Scalar(r, g, b, 255.0));
        }
    }

    void PersonRecognizer::object_detect(cv::Mat &image)
    {
        bboxes_.bounding_boxes.clear();
        detect_list_.clear();

        cv::Mat blob = cv::dnn::blobFromImage(image, 1, cv::Size(image.cols, image.rows), cv::Scalar());
        net_.setInput(blob);
        std::vector<std::string> outNames(2);
        outNames[0] = "detection_out_final";
        outNames[1] = "detection_masks";
        std::vector<cv::Mat> pred;
        net_.forward(pred, outNames);
        cv::Mat pred_detections = pred[0];
        cv::Mat pred_masks = pred[1];
        // Output size of masks is NxCxHxW where
        // N - number of detected boxes
        // C - number of classes (excluding background)
        // HxW - segmentation shape
        int num_detections = pred_detections.size[2];
        int num_classes = pred_masks.size[1];

        pred_detections = pred_detections.reshape(1, pred_detections.total() / 7);

        // for(int i=0; i<1; i++){
        for(int i=0; i<num_detections; i++){
            float conf = pred_detections.at<float>(i, 2);

            if(conf > conf_threshold_){
                int x0 = int(pred_detections.at<float>(i, 3) * image.cols);
                int y0 = int(pred_detections.at<float>(i, 4) * image.rows);
                int x1 = int(pred_detections.at<float>(i, 5) * image.cols);
                int y1 = int(pred_detections.at<float>(i, 6) * image.rows);

                x0 = std::max(0, std::min(x0, image.cols - 1));
                y0 = std::max(0, std::min(y0, image.rows - 1));
                x1 = std::max(0, std::min(x1, image.cols - 1));
                y1 = std::max(0, std::min(y1, image.rows - 1));
                cv::Rect rect(x0, y0, x1-x0+1, y1-y0+1);

                int id = int(pred_detections.at<float>(i, 1));
                if(id != 0) continue;
                std::string class_name = class_names_[id];
                std::string label = class_name + ":" + std::to_string(conf).substr(0, 4);

                cv::Mat object_mask(pred_masks.size[2], pred_masks.size[3], CV_32F, pred_masks.ptr<float>(i, id));
                cv::resize(object_mask, object_mask, cv::Size(rect.width, rect.height));
                cv::Mat mask = (object_mask > mask_threshold_);
                mask.convertTo(mask, CV_8U);
                // cv::imshow("mask", mask);
                // int key = cv::waitKey(5);
                // cv::imshow("roi", input_image_(rect));
                // int key = cv::waitKey(5);
                draw_bbox(image, rect, id, conf, mask);

                DetectInfo detect_info;
                detect_info.label = class_name;
                detect_info.conf = conf;
                detect_info.mask = mask;
                detect_info.roi = rect;
                detect_list_.push_back(detect_info);
                // draw_bbox(image, rect, id, conf, object_mask);
                // if(id == 1){
                //     // set_bbox(x0, x1, y0, y1, conf, id, class_name);
                //     // send_bbox(x0, x1, y0, y1, conf, id, class_name);
                //     draw_bbox(image, x0, y0, x1, y1, label);
                // }
            }
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        // bboxes_pub_.publish(bboxes_);
        image_pub_.publish(msg);
    }

    void PersonRecognizer::draw_bbox(cv::Mat &image, cv::Rect rect, int id, float conf, cv::Mat& mask)
    {
        // cv::Rect object(x0, y0, x1-x0, y1-y0);
        cv::rectangle(image, rect, cv::Scalar(255, 255, 255), 2);

        int baseline = 0;
        std::string label = class_names_[id] + ":" + std::to_string(conf).substr(0, 4);
        cv::Size  label_size = cv::getTextSize(label,
                cv::FONT_HERSHEY_SIMPLEX,0.5, 1, &baseline);
        cv::putText(image, label, cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        cv::Scalar color = colors_[id % colors_.size()];

        // cv::imshow("object_mask", object_mask);
        // int key = cv::waitKey(5);
        // cv::resize(object_mask, object_mask, cv::Size(rect.width, rect.height));
        // cv::Mat mask = (object_mask > mask_threshold_);
        // mask.convertTo(mask, CV_8U);
        cv::Mat colored_roi = (0.3 * color + 0.7 * image(rect));
        colored_roi.convertTo(colored_roi, CV_8UC3);
        // cv::imshow("roi", colored_roi);
        // int key = cv::waitKey(5);
        // cv::imshow("mask", mask);
        // int key = cv::waitKey(5);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        // mask.convertTo(mask, CV_8U);
        cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_roi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);
        colored_roi.copyTo(image(rect), mask);
    }

    // void PersonRecognizer::set_bbox(int x0, int x1, int y0, int y1, float conf,
    //         int id, std::string class_name)
    // {
    //     camera_apps_msgs::BoundingBox bbox;
    //     
    //     bbox.confidence = conf;
    //     bbox.xmin = x0;
    //     bbox.xmax = x1; 
    //     bbox.ymin = y0;
    //     bbox.ymax = y1;
    //     bbox.id = id;
    //     bbox.label = class_name;
    //
    //     bboxes_.bounding_boxes.push_back(bbox);
    // }

    void PersonRecognizer::create_object_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc, cv::Rect& roi, cv::Mat& mask)
    {

        object_pc->header = input_pc_->header;
        object_pc->is_dense = 0;

        object_pc->points.clear();
        // for(int y=bbox.ymin; y<bbox.ymax; y++){
        //     for(int x=bbox.xmin; x < bbox.xmax; x++)
        //     {
        //         object_pc->points.push_back(input_pc_->points[y * input_pc_->width + x]);
        //     }
        // }
        for(int i=0; i<mask.rows; i++){
            for(int j=0; j<mask.cols; j++){
                if((int)mask.at<unsigned char>(i, j) == 255){
                    object_pc->points.push_back(input_pc_->points[(roi.y + i) * input_pc_->width + (roi.x + j)]);
                }
            }
        }

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*object_pc, *object_pc, indices);
    }

    geometry_msgs::PointStamped PersonRecognizer::caluculate_centroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        geometry_msgs::PointStamped centroid_msg;
        pcl::PointXYZ centroid_pcl;

        pcl::computeCentroid(*pc_in, centroid_pcl);
        centroids_->points.push_back(centroid_pcl);
        centroids_->header = pc_in->header;
        
        pcl_conversions::fromPCL(pc_in->header, centroid_msg.header);
        centroid_msg.point.x = centroid_pcl.x;
        centroid_msg.point.y = centroid_pcl.y;
        centroid_msg.point.z = centroid_pcl.z;

        return centroid_msg;
    }

    void PersonRecognizer::downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        double step = pc_in->points.size() / (double)points_limit_;
        if(step < 1) step = 1;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

        tmp_pc->header = pc_in->header;
        tmp_pc->is_dense = true;

        for(double i=0; i<pc_in->points.size(); i+=step){
            tmp_pc->points.push_back(pc_in->points[(int)i]);
        }
        tmp_pc->width = tmp_pc->points.size();
        tmp_pc->height = 1;

        *pc_in = *tmp_pc;
    }

    bool PersonRecognizer::clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud (pc_in);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance (cluster_tolerance_);
        ec.setMinClusterSize (min_cluster_size_);//最小のクラスターの値を設定
        ec.setMaxClusterSize (max_cluster_size_);//最大のクラスターの値を設定
        ec.setSearchMethod (tree);//検索に使用する手法を指定
        ec.setInputCloud (pc_in);//点群を入力
        ec.extract (cluster_indices);//クラスター情報を出力
        
        if(cluster_indices.size() == 0) return false;

        std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin ();
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
            cloud_cluster->points.push_back (pc_in->points[*pit]); 
        }

        cloud_cluster->header = pc_in->header;
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        *pc_in = *cloud_cluster;

        return true;
    }

    cv::Mat PersonRecognizer::calc_hist(cv::Rect roi, cv::Mat mask)
    {
        cv::Mat hsv_image;
        cv::cvtColor(input_image_(roi), hsv_image, cv::COLOR_BGR2HSV);
        cv::Mat hist;
        std::vector<cv::Mat> images = {hsv_image};
        std::vector<int> channels = {0, 1};
        std::vector<int> hist_size = {45, 64};
        std::vector<float> ranges = {0, 180, 0, 256};
        // std::cout << "hsv row: " << hsv_image.rows << "hsv col: " << hsv_image.cols << std::endl;
        // std::cout << "mask row: " << mask.rows << "mask col: " << mask.cols << std::endl;
        cv::calcHist(images, channels, mask, hist, hist_size, ranges);
        // cv::calcHist(hsv_image, channels, mask, hist, hist_size, ranges);
        // std::cout << "hist sum: " << cv::sum(hist)[0] << std::endl;
        // std::cout << "hist type: " << hist.type() << std::endl;
        hist /= (float)cv::sum(hist)[0];

        return hist;
    }

    std::vector<float> PersonRecognizer::convert_hist_to_vector(cv::Mat& hist)
    {
        std::vector<float> hist_vector;
        for(int i=0; i<hist.rows; i++){
            for(int j=0; j<hist.cols; j++){
                hist_vector.push_back(hist.at<float>(i, j));
            }
        }
        return hist_vector;

    }

    void PersonRecognizer::through_filtering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(pc_in);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(through_th_z_min_, through_th_z_max_);
        pass.filter(*pc_in);
    }
}

