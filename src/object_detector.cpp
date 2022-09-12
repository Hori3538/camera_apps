#include <object_detector/object_detector.h>

namespace camera_apps
{
    ObjectDetector::ObjectDetector(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("camera_topic_name", camera_topic_name_, std::string("/camera/color/image_raw"));
        pnh.getParam("model_path", model_path_);
        pnh.param("conf_threshold", conf_threshold_, 0.4);

        image_transport::ImageTransport it(nh);
        image_sub_ = it.subscribe(camera_topic_name_, 1, &ObjectDetector::image_callback, this);
        image_pub_ = it.advertise("/detected_image", 1);
        bboxes_pub_ = nh.advertise<camera_apps_msgs::BoundingBoxes>("/bounding_boxes", 1);

        set_network();
    }

    void ObjectDetector::image_callback(const sensor_msgs::ImageConstPtr &msg)
    {

        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            input_image_ = cv_ptr->image;
            bboxes_.header.stamp = msg->header.stamp;
            object_detect(input_image_);
        }
        catch(cv_bridge::Exception &e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    std::vector<std::string> ObjectDetector::read_file(std::string filename, char delimiter)
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

    void ObjectDetector::set_network()
    {
        std::string proto_path = model_path_ + "/ssd_mobilenet_v2_coco.pbtxt";
        std::string weight_path = model_path_ + "/frozen_inference_graph.pb";
        std::string label_path = model_path_ + "/object_detection_classes_coco.txt";

        net_ = cv::dnn::readNet(proto_path, weight_path);
        class_names_ = read_file(label_path);
    }

    void ObjectDetector::object_detect(cv::Mat &image)
    {
        bboxes_.bounding_boxes.clear();

        cv::Mat blob = cv::dnn::blobFromImage(image, 1, cv::Size(300, 300));
        net_.setInput(blob);
        cv::Mat pred = net_.forward();
        cv::Mat pred_mat(pred.size[2], pred.size[3], CV_32F, pred.ptr<float>());

        for(int i=0; i<pred_mat.rows; i++){
            float conf = pred_mat.at<float>(i, 2);

            if(conf > conf_threshold_){
                int x0 = int(pred_mat.at<float>(i, 3) * image.cols);
                int y0 = int(pred_mat.at<float>(i, 4) * image.rows);
                int x1 = int(pred_mat.at<float>(i, 5) * image.cols);
                int y1 = int(pred_mat.at<float>(i, 6) * image.rows);

                int id = int(pred_mat.at<float>(i, 1));
                std::string class_name = class_names_[id-1];
                std::string label = class_name + ":" + std::to_string(conf).substr(0, 4);
                if(id == 1){
                // if(true){
                    set_bbox(x0, x1, y0, y1, conf, id, class_name);
                    draw_bbox(image, x0, y0, x1, y1, label);
                }
            }
        }
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        bboxes_pub_.publish(bboxes_);
        image_pub_.publish(msg);
    }

    void ObjectDetector::draw_bbox(cv::Mat &image, int x0, int y0, int x1, int y1, std::string label)
    {
        cv::Rect object(x0, y0, x1-x0, y1-y0);
        cv::rectangle(image, object, cv::Scalar(255, 255, 255), 2);

        int baseline = 0;
        cv::Size  label_size = cv::getTextSize(label,
                cv::FONT_HERSHEY_SIMPLEX,0.5, 1, &baseline);
        cv::putText(image, label, cv::Point(x0, y0),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
    }


    void ObjectDetector::set_bbox(int x0, int x1, int y0, int y1, float conf,
            int id, std::string class_name)
    {
        camera_apps_msgs::BoundingBox bbox;
        
        bbox.confidence = conf;
        bbox.xmin = x0;
        bbox.xmax = x1; 
        bbox.ymin = y0;
        bbox.ymax = y1;
        bbox.id = id;
        bbox.label = class_name;

        bboxes_.bounding_boxes.push_back(bbox);
    }
}

