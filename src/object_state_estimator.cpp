#include <object_state_estimator/object_state_estimator.h>

namespace camera_apps
{
    ObjectStateEstimator::ObjectStateEstimator(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("img_width", img_width_, 640);
        pnh.param("img_height", img_height_, 480);
        pnh.param("mean_k", mean_k_, 50);
        pnh.param("std_dev_th", std_dev_th_, 0.1);
        pnh.param("leafsize", leafsize_, 0.01);
        pnh.param("points_limit", points_limit_, 10000);
        pnh.param("cluster_tolerance", cluster_tolerance_, 0.02);
        pnh.param("min_cluster_size", min_cluster_size_, 1000);
        pnh.param("max_cluster_size", max_cluster_size_, 20000);
        pnh.param("publish_object_pc_flag", publish_object_pc_flag_, false);
        pnh.param("ransac_dist_th", ransac_dist_th_, 0.01);
        pnh.param("through_th_z_min", through_th_z_min_, -70.0);
        pnh.param("through_th_z_max", through_th_z_max_, 2.0);
        pnh.param("downsample_for_visualize", downsample_for_visualize_, true);
        pnh.param("leafsize_for_visualize", leafsize_for_visualize_, 0.04);

        object_pc_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object_state/object_pc", 1);
        centroids_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object_state/centroids", 1);

        bboxes_sub_  = new message_filters::Subscriber<camera_apps_msgs::BoundingBoxes> (nh, "/bounding_boxes", 5);
        pc_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/camera/depth_registered/points", 5);
        sync_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *bboxes_sub_, *pc_sub_);
        sync_->registerCallback(&ObjectStateEstimator::sync_callback, this);

        object_states_pub_ = nh.advertise<camera_apps_msgs::ObjectStates>("/object_states", 1);

        tf2_listener_ = new tf2_ros::TransformListener(tf_buffer_);
    }

    ObjectStateEstimator::~ObjectStateEstimator()
    {
        delete this->bboxes_sub_;
        delete this->pc_sub_;
        delete this->sync_;
        delete this->tf2_listener_;
    }

    void ObjectStateEstimator::sync_callback(const camera_apps_msgs::BoundingBoxesConstPtr &bboxes_msg,
            const sensor_msgs::PointCloud2ConstPtr &pc_msg)
    {
        try{
            geometry_msgs::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_.lookupTransform("camera_fixed_frame", "camera_color_optical_frame", ros::Time(0));
            Eigen::Matrix4f mat = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();

            sensor_msgs::PointCloud2 pc_msg_temp;
            pcl_ros::transformPointCloud(mat, *pc_msg, pc_msg_temp);
            pcl::fromROSMsg(pc_msg_temp, *input_pc_);
            input_pc_->header.frame_id = "camera_fixed_frame";
        }
        catch(tf2::TransformException &ex){
            ROS_WARN("%s", ex.what());
            return;
        }

        bboxes_ = *bboxes_msg;

        pcl_conversions::fromPCL(input_pc_->header, object_states_.header);
        object_states_.object_states.clear();
        object_pcs_->points.clear();
        centroids_->points.clear();
        for(auto bbox: bboxes_msg->bounding_boxes){
            camera_apps_msgs::ObjectState object_state;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc {new pcl::PointCloud<pcl::PointXYZRGB>};

            object_state.label = bbox.label;
            object_state.confidence = bbox.confidence;

            adjust_bbox(bbox);
            create_object_pc(object_pc, bbox);
            through_filtering(object_pc);
            downsampling(object_pc);
            // std::cout << " after: " << object_pc->points.size() << std::endl;
            // std::cout << "before: " << object_pc->points.size();
            // surface_segmentation(object_pc);
            if(!clustering(object_pc)){
                // std::cout << "cannot clustering!!" << std::endl;
                continue;
            }
            // remove_outlier(object_pc);
            for(const auto& point: object_pc->points) object_pcs_->points.push_back(point);
            object_pcs_->header = object_pc->header;

            geometry_msgs::PointStamped centroid = caluculate_centroid(object_pc);
            object_state.centroid = centroid;

            if(publish_object_pc_flag_) pcl::toROSMsg(*object_pc, object_state.object_pc);
            object_states_.object_states.push_back(object_state);
        }
        if(downsample_for_visualize_) downsampling_pcl(object_pcs_, leafsize_for_visualize_);
        object_pc_pub_.publish(object_pcs_);
        centroids_pub_.publish(centroids_);
        object_states_pub_.publish(object_states_);
    }

    void ObjectStateEstimator::adjust_bbox(camera_apps_msgs::BoundingBox& bbox)
    {
        if(bbox.xmin < 0) bbox.xmin = 0;
        if(bbox.ymin < 0) bbox.ymin = 0;
        if(bbox.xmax >= img_width_) bbox.xmax = img_width_ - 1;
        if(bbox.ymax >= img_height_) bbox.ymax = img_height_ - 1;
    }

    void ObjectStateEstimator::create_object_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc, camera_apps_msgs::BoundingBox bbox)
    {

        object_pc->header = input_pc_->header;
        object_pc->is_dense = 0;


        object_pc->points.clear();
        for(int y=bbox.ymin; y<bbox.ymax; y++){
            for(int x=bbox.xmin; x < bbox.xmax; x++)
            {
                object_pc->points.push_back(input_pc_->points[y * input_pc_->width + x]);
            }
        }

        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*object_pc, *object_pc, indices);
        
        // downsampling(object_pc);
    }

    void ObjectStateEstimator::downsampling(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
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
    void ObjectStateEstimator::downsampling_pcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in, double leafsize)
    {
        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(pc_in);
        vg.setLeafSize(leafsize, leafsize, leafsize);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
        vg.filter(*tmp);
        *pc_in = *tmp;
    }

    void ObjectStateEstimator::remove_outlier(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(pc_in);
        sor.setMeanK (mean_k_);
        sor.setStddevMulThresh (std_dev_th_);
        sor.setNegative (false);
        sor.filter (*pc_in);
    }

    void ObjectStateEstimator::coloring_pc(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in, int red, int green, int blue)
    {
        for(auto &p: pc_in->points){
            p.r = red;
            p.g = green;
            p.b = blue;
        }
    }

    bool ObjectStateEstimator::clustering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
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

    geometry_msgs::PointStamped ObjectStateEstimator::caluculate_centroid(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
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

    void ObjectStateEstimator::surface_segmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

        pcl::SACSegmentation<pcl::PointXYZRGB> seg;

        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setInputCloud(pc_in);
        seg.setModelType (pcl::SACMODEL_PLANE);//検出するモデルのタイプを指定
        seg.setMethodType (pcl::SAC_RANSAC);//検出に使用する方法を指定
        seg.setDistanceThreshold (ransac_dist_th_);//RANSACの最小二乗法の許容誤差範囲
        seg.setMaxIterations(50);
        seg.setProbability(0.8);
        seg.segment(*inliers, *coefficients);

        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(pc_in);
        extract.setIndices(inliers);
        extract.setNegative(true);//trueの場合出力は検出された平面以外のデータ falseの場合は平面のデータ
        extract.filter(*pc_in);
    }

    void ObjectStateEstimator::through_filtering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_in)
    {
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(pc_in);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(through_th_z_min_, through_th_z_max_);
        pass.filter(*pc_in);
    }
}
