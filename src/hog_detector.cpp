#include <hog_detector/hog_detector.h>

namespace camera_apps
{
    HogDetector::HogDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("camera_topic_name", camera_topic_name_, std::string("/camera/color/image_raw"));
        pnh.getParam("model_path", model_path_);
        pnh.param("conf_threshold", conf_threshold_, 0.4);
        pnh.param("hit_threshold", hit_threshold_, 1.0);

        image_transport::ImageTransport it(nh);
        image_sub_ = it.subscribe(camera_topic_name_, 1, &HogDetector::image_callback, this);
        image_pub_ = it.advertise("/detected_image", 1);
        bboxes_pub_ = nh.advertise<camera_apps_msgs::BoundingBoxes>("/bounding_boxes", 1);
        // bbox_pub_ = nh.advertise<camera_apps_msgs::BoundingBox>("/bounding_box", 1);

        set_network();
    }

    void HogDetector::image_callback(const sensor_msgs::ImageConstPtr &msg)
    {

        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            input_image_ = cv_ptr->image;
            bboxes_.header.stamp = msg->header.stamp;
            // msg_stamp_ = msg->header.stamp;
            object_detect(input_image_);
        }
        catch(cv_bridge::Exception &e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    std::vector<std::string> HogDetector::read_file(std::string filename, char delimiter)
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

    void HogDetector::set_network()
    {
        std::string proto_path = model_path_ + "/ssd_mobilenet_v2_coco.pbtxt";
        std::string weight_path = model_path_ + "/frozen_inference_graph.pb";
        std::string label_path = model_path_ + "/object_detection_classes_coco.txt";

        net_ = cv::dnn::readNet(proto_path, weight_path);
        class_names_ = read_file(label_path);
    }

    // void HogDetector::object_detect(cv::Mat &image)
    // {
    //     bboxes_.bounding_boxes.clear();
    //     // bbox_.header.stamp = msg_stamp_;
    //
    //     cv::Mat blob = cv::dnn::blobFromImage(image, 1, cv::Size(300, 300));
    //     // cv::Mat blob = cv::dnn::blobFromImage(image, 1, cv::Size(image.cols, image.rows), cv::Scalar());
    //     net_.setInput(blob);
    //     cv::Mat pred = net_.forward();
    //     cv::Mat pred_mat(pred.size[2], pred.size[3], CV_32F, pred.ptr<float>());
    //
    //     // for(int i=0; i<1; i++){
    //     for(int i=0; i<pred_mat.rows; i++){
    //         float conf = pred_mat.at<float>(i, 2);
    //
    //         if(conf > conf_threshold_){
    //             int x0 = int(pred_mat.at<float>(i, 3) * image.cols);
    //             int y0 = int(pred_mat.at<float>(i, 4) * image.rows);
    //             int x1 = int(pred_mat.at<float>(i, 5) * image.cols);
    //             int y1 = int(pred_mat.at<float>(i, 6) * image.rows);
    //
    //             int id = int(pred_mat.at<float>(i, 1));
    //             std::string class_name = class_names_[id-1];
    //             std::string label = class_name + ":" + std::to_string(conf).substr(0, 4);
    //             if(id == 1){
    //             // if(true){
    //                 set_bbox(x0, x1, y0, y1, conf, id, class_name);
    //                 // send_bbox(x0, x1, y0, y1, conf, id, class_name);
    //                 draw_bbox(image, x0, y0, x1, y1, label);
    //             }
    //         }
    //     }
    //     sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    //     bboxes_pub_.publish(bboxes_);
    //     image_pub_.publish(msg);
    // }

    void HogDetector::object_detect(cv::Mat &image)
    {
        bboxes_.bounding_boxes.clear();
        // std::cout << "start detect!" << std::endl;

        //
        // // for(int i=0; i<1; i++){
        // for(int i=0; i<pred_mat.rows; i++){
        //     float conf = pred_mat.at<float>(i, 2);
        //
        //     if(conf > conf_threshold_){
        //         int x0 = int(pred_mat.at<float>(i, 3) * image.cols);
        //         int y0 = int(pred_mat.at<float>(i, 4) * image.rows);
        //         int x1 = int(pred_mat.at<float>(i, 5) * image.cols);
        //         int y1 = int(pred_mat.at<float>(i, 6) * image.rows);
        //
        //         int id = int(pred_mat.at<float>(i, 1));
        //         std::string class_name = class_names_[id-1];
        //         std::string label = class_name + ":" + std::to_string(conf).substr(0, 4);
        //         if(id == 1){
        //         // if(true){
        //             set_bbox(x0, x1, y0, y1, conf, id, class_name);
        //             // send_bbox(x0, x1, y0, y1, conf, id, class_name);
        //             draw_bbox(image, x0, y0, x1, y1, label);
        //         }
        //     }
        // }
        // bboxes_pub_.publish(bboxes_);
        //


        cv::HOGDescriptor hog;
        hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

        std::vector<cv::Rect> detections;
        std::vector<double> weights;
        // hog.detectMultiScale(image, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
        hog.detectMultiScale(image, detections, weights, hit_threshold_, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
        // for(const auto& weight: weights){
        //     std::cout << "weight: " << weight << std::endl;
        // }
        // std::cout << std::endl;
        // hog.detectMultiScale(image, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.2, 2);

        int i=0;
        for (auto& detection : detections) {
            resize_boxes(detection);
            draw_bbox(image, detection);
            set_bbox(detection, weights[i]);
            // std::cout << "weight: " << weights[i] << std::endl;
            // cv::rectangle(image, detection.tl(), detection.br(), cv::Scalar(255, 0, 0), 2);
            i++;
        }
        // for(int i=0; i<detections.size(); i++){
        //     resize_boxes(detections[i]);
        //     draw_bbox(image, detections[i]);
        //     set_bbox(detections[i], )
        // }

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        image_pub_.publish(msg);
        bboxes_pub_.publish(bboxes_);
    }

    // void HogDetector::draw_bbox(cv::Mat &image, int x0, int y0, int x1, int y1, std::string label)
    // {
    //     cv::Rect object(x0, y0, x1-x0, y1-y0);
    //     cv::rectangle(image, object, cv::Scalar(255, 255, 255), 2);
    //
    //     int baseline = 0;
    //     cv::Size  label_size = cv::getTextSize(label,
    //             cv::FONT_HERSHEY_SIMPLEX,0.5, 1, &baseline);
    //     cv::putText(image, label, cv::Point(x0, y0),
    //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    // }
    void HogDetector::draw_bbox(cv::Mat &image, cv::Rect& rect)
    // void HogDetector::draw_bbox(cv::Mat &image, cv::Rect& rect, std::string label)
    {
        cv::rectangle(image, rect, cv::Scalar(255, 255, 255), 2);

        int baseline = 0;
        // cv::Size  label_size = cv::getTextSize(label,
                // cv::FONT_HERSHEY_SIMPLEX,0.5, 1, &baseline);
        // cv::putText(image, label, cv::Point(x0, y0),
                // cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    }

    // void HogDetector::send_bbox(int x0, int x1, int y0, int y1, float conf,
    //         int id, std::string class_name)
    // {
    //     bbox_.confidence = conf;
    //     bbox_.xmin = x0;
    //     bbox_.xmax = x1; 
    //     bbox_.ymin = y0;
    //     bbox_.ymax = y1;
    //     bbox_.id = id;
    //     bbox_.label = class_name;
    //     bbox_pub_.publish(bbox_);
    // }

    void HogDetector::set_bbox(cv::Rect& rect, float weight)
    {
        camera_apps_msgs::BoundingBox bbox;
        
        bbox.confidence = weight;
        bbox.xmin = rect.x;
        bbox.xmax = rect.x + rect.width; 
        // bbox.ymin = rect.y - rect.height;
        // bbox.ymax = rect.y;
        bbox.ymin = rect.y;
        bbox.ymax = rect.y + rect.height;
        // bbox.id = id;
        // bbox.label = class_name;

        bboxes_.bounding_boxes.push_back(bbox);

        // std::cout << "xmin: " << bbox.xmin << std::endl;
        // std::cout << "xmax: " << bbox.xmax << std::endl;
        // std::cout << "ymin: " << bbox.ymin << std::endl;
        // std::cout << "ymax: " << bbox.ymax << std::endl;
        // std::cout << std::endl;
    }
    // void HogDetector::set_bbox(int x0, int x1, int y0, int y1, float conf,
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
    void HogDetector::resize_boxes(cv::Rect& box) {
        box.x += cvRound(box.width*0.1);
        box.width = cvRound(box.width*0.8);
        box.y += cvRound(box.height*0.06);
        box.height = cvRound(box.height*0.8);
    }

    void HogDetector::process()
    {
        std::cout << "start process" << std::endl;

    }
}

