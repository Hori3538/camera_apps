#include <mask_rcnn/mask_rcnn.hpp>

namespace camera_apps
{
    MaskRcnn::MaskRcnn(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.getParam("model_path", model_path_);
        pnh.param<std::string>("camera_topic_name", camera_topic_name_, "/camera/color/image_raw");
        pnh.param<double>("conf_threshold", conf_threshold_, 0.4);
        pnh.param<double>("mask_threshold", mask_threshold_, 0.4);
        pnh.param<bool>("detect_only_person", detect_only_person_, true);
        pnh.param<int>("hz", hz_, 10);
        
        image_transport::ImageTransport it(nh);
        image_sub_ = it.subscribe(camera_topic_name_, 1, &MaskRcnn::image_callback, this);

        image_pub_ = it.advertise("/mask_rcnn/detected_image", 1);
        masks_pub_ = nh.advertise<camera_apps_msgs::Masks>("/mask_rcnn/masks", 1);

        set_network();
    }

    void MaskRcnn::image_callback(const sensor_msgs::ImageConstPtr &msg)
    {

        // cv_bridge::CvImagePtr cv_ptr;
        try
        {
            input_image_cvptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            // input_image_ = cv_ptr->image.clone();
            masks_.header.stamp = msg->header.stamp;
            // object_detect(cv_ptr->image);
            masks_.height = msg->height;
            masks_.width = msg->width;
        }
        catch(cv_bridge::Exception &e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }


    std::vector<std::string> MaskRcnn::read_file(std::string filename, char delimiter)
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

    void MaskRcnn::set_network()
    {
        std::string proto_path = model_path_ + "/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
        std::string weight_path = model_path_ + "/frozen_inference_graph.pb";
        std::string label_path = model_path_ + "/object_detection_classes_coco.txt";

        net_ = cv::dnn::readNet(proto_path, weight_path);

        //GPU
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        //NCS2
        // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        // net_.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
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

    void MaskRcnn::object_detect(cv::Mat &image)
    {
        masks_.masks.clear();

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
            if(conf < conf_threshold_) continue;

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
            if(detect_only_person_ && id != 0) continue;
            std::string class_name = class_names_[id];
            std::string label = class_name + ":" + std::to_string(conf).substr(0, 4);

            cv::Mat object_mask(pred_masks.size[2], pred_masks.size[3], CV_32F, pred_masks.ptr<float>(i, id));
            cv::resize(object_mask, object_mask, cv::Size(rect.width, rect.height));
            cv::Mat mask = (object_mask > mask_threshold_);
            mask.convertTo(mask, CV_8U);

            draw_bbox(image, rect, id, conf, mask);
            set_mask(rect, id, conf, mask, class_name);


        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        image_pub_.publish(msg);
        masks_pub_.publish(masks_);
    }

    void MaskRcnn::draw_bbox(cv::Mat &image, cv::Rect rect, int id, float conf, cv::Mat& mask)
    {
        cv::rectangle(image, rect, cv::Scalar(255, 255, 255), 2);

        int baseline = 0;
        std::string label = class_names_[id] + ":" + std::to_string(conf).substr(0, 4);
        cv::Size  label_size = cv::getTextSize(label,
                cv::FONT_HERSHEY_SIMPLEX,0.5, 1, &baseline);
        cv::putText(image, label, cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

        cv::Scalar color = colors_[id % colors_.size()];

        cv::Mat colored_roi = (0.3 * color + 0.7 * image(rect));
        colored_roi.convertTo(colored_roi, CV_8UC3);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_roi, contours, -1, color, 5, cv::LINE_8, hierarchy, 100);
        colored_roi.copyTo(image(rect), mask);
    }

    void MaskRcnn::set_mask(cv::Rect rect, int id, float conf, cv::Mat& mask, std::string class_name)
    {

        camera_apps_msgs::Mask mask_msg;
        camera_apps_msgs::BoundingBox bbox;
        
        bbox.confidence = conf;
        bbox.id = id;
        bbox.label = class_name;
        bbox.xmin = rect.x;
        bbox.xmax = rect.x + rect.width; 
        bbox.ymin = rect.y;
        bbox.ymax = rect.y + rect.height;

        mask_msg.bounding_box = bbox;
        mask_msg.mask = *cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
        masks_.masks.push_back(mask_msg);
    }

    void MaskRcnn::process()
    {
        ros::Rate loop_rate(hz_);
        
        while(ros::ok())
        {
            if(input_image_cvptr_.has_value())
            {
                object_detect(input_image_cvptr_.value()->image);
                input_image_cvptr_.reset();
            }
            ros::spinOnce();
            loop_rate.sleep();
        }
    }
}
