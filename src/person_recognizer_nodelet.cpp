#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <person_recognizer/person_recognizer.h>


namespace camera_apps{
    class PersonRecognizerNodelet : public nodelet::Nodelet
    {
        public:
            PersonRecognizerNodelet() = default;
            ~PersonRecognizerNodelet() {
        if (person_recognizer_) delete person_recognizer_;
            }
        private:
            virtual void onInit() {
                ros::NodeHandle nh;
                ros::NodeHandle pnh("~");
                pnh = getPrivateNodeHandle();
                person_recognizer_ = new camera_apps::PersonRecognizer(nh, pnh);
            }
            camera_apps::PersonRecognizer *person_recognizer_;
    };
}
// Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(camera_apps::PersonRecognizerNodelet, nodelet::Nodelet);
