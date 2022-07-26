#include <yolov5/yolov5.h>

namespace camera_apps
{
    Yolov5::Yolov5(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        pnh.param("camera_topic_name", camera_topic_name_, std::string("/camera/color/image_raw"));
        pnh.getParam("model_path", model_path_);
        pnh.param("conf_threshold", conf_threshold_, 0.45);
        pnh.param("score_threshold", score_threshold_, 0.5);
        pnh.param("nms_threshold", nms_threshold_, 0.45);

        image_transport::ImageTransport it(nh);
        image_sub_ = it.subscribe(camera_topic_name_, 1, &Yolov5::image_callback, this);
        image_pub_ = it.advertise("/detected_image", 1);
        bboxes_pub_ = nh.advertise<camera_apps_msgs::BoundingBoxes>("/bounding_boxes", 1);

        set_network();
    }

    void Yolov5::image_callback(const sensor_msgs::ImageConstPtr &msg)
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

    std::vector<std::string> Yolov5::read_file(std::string filename, char delimiter)
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

    void Yolov5::set_network()
    {
        std::string net_path = model_path_ + "/YOLOv5x.onnx";
        std::string label_path = model_path_ + "/coco_classes.txt";

        net_ = cv::dnn::readNet(net_path);

        //GPU
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        class_names_ = read_file(label_path);
    }

    void Yolov5::object_detect(cv::Mat &image)
    {
        bboxes_.bounding_boxes.clear();
        std::vector<cv::Mat> detections;     // Process the image.
        detections = pre_process(image);
        cv::Mat img = post_process(image, detections, class_names_);

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
        bboxes_pub_.publish(bboxes_);
        image_pub_.publish(msg);
    }


    void Yolov5::set_bbox(int x0, int x1, int y0, int y1, float conf,
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

    std::vector<cv::Mat> Yolov5::pre_process(cv::Mat &input_image)
    {
        cv::resize(input_image, input_image, cv::Size(up_width_, up_height_), cv::INTER_LINEAR);
        // Convert to blob.
        cv::Mat blob;
        cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(input_width_, input_height_), cv::Scalar(), true, false);

        net_.setInput(blob);

        // Forward propagate.
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        return outputs;
    }
    cv::Mat Yolov5::post_process(cv::Mat &input_image, std::vector<cv::Mat> &outputs, const std::vector<std::string> &class_name)
    {
        // Initialize std::vectors to hold respective outputs while unwrapping     detections.
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        // Resizing factor.
        float x_factor = (float)input_image.cols / input_width_;
        float y_factor = (float)input_image.rows / input_height_;
        float *data = (float *)outputs[0].data;
        const int dimensions = 85;
        // 25200 for default size 640.
        const int rows = 25200;
        // Iterate through 25200 detections.
        for (int i = 0; i < rows; ++i)
        {
            float confidence = data[4];
            // Discard bad detections and continue.
            if (confidence >= conf_threshold_)
            {
                float * classes_scores = data + 5;
                // Create a 1x85 cv::Mat and store class scores of 80 classes.
                cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
                // Perform minMaxLoc and acquire the index of best class  score.
                cv::Point class_id;
                double max_class_score;
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                // Continue if the class score is above the threshold.
                if (max_class_score > score_threshold_)
                {
                    // Store class ID and confidence in the pre-defined respective std::vectors.
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    // Center.
                    float cx = data[0];
                    float cy = data[1];
                    // Box dimension.
                    float w = data[2];
                    float h = data[3];
                    // Bounding box coordinates.
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    // Store good detections in the boxes std::vector.
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            // Jump to the next row.
            data += 85;
        }
        // Perform Non-Maximum Suppression and draw predictions.
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);
        for (int i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            // Draw bounding box.
            rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(255,255,255), 3*thickness_);
            // Get the label for the class name and its confidence.
            std::string label = cv::format("%.2f", confidences[idx]);
            label = class_name[class_ids[idx]] + ":" + label;
            // Draw class labels.
            draw_label(input_image, label, left, top);
        }
        return input_image;
    }
    void Yolov5::draw_label(cv::Mat& input_image, std::string label, int left, int top)
    {
        // Display the label at the top of the bounding box.
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, font_face_, font_scale_, thickness_, &baseLine);
        top = cv::max(top, label_size.height);
        // Top left corner.
        cv::Point tlc = cv::Point(left, top);
        // Bottom right corner.
        cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);
        // Draw white rectangle.
        rectangle(input_image, tlc, brc, cv::Scalar(0, 0, 0), cv::FILLED);
        // Put the label on the black rectangle.
        putText(input_image, label, cv::Point(left, top + label_size.height), font_face_, font_scale_, cv::Scalar(255, 255, 0), thickness_);

    }

}

