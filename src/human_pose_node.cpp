#include <vector>
#include <chrono>
#include <ros/ros.h>
#include <inference_engine.hpp>

#include "../monitors/presenter.h"
#include "ocv_common.hpp"

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

using namespace InferenceEngine;
using namespace human_pose_estimation;

cv::Mat Color_pic;
bool init_flag = false;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
   cv_bridge::CvImagePtr cam_img;
   //std::cout<<"received pic!"<<std::endl;

   try {
       cam_img = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
   }catch(cv_bridge::Exception& e){
       ROS_ERROR("cv_bridge exception: %s",e.what());
       return;
   }
   if(cam_img)
   {
       Color_pic = cam_img->image.clone();

   }

}
int main(int argc, char** argv) {
    ros::init(argc, argv, "human_pose_node");
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("/camera/color/image_raw",1,imageCallback);
    image_transport::Publisher img_pub = it.advertise("human_pose_show",1);



    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    HumanPoseEstimator estimator("/home/zoo/openvino/open_model_zoo/tools/downloader/intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml",
                                     "CPU", false);

    int delay = 33;
    cv::Mat curr_frame;
        //curr_frame = Color_pic.clone();
    cv::Mat next_frame;

        //estimator.reshape(curr_frame);  // Do not measure network reshape, if it happened

    std::cout << "To close the application, press 'CTRL+C' here";
    std::cout << std::endl;

    cv::Size graphSize{(int)640 / 4, 60};
    Presenter presenter("", (int)480 - graphSize.height - 10, graphSize);
    std::vector<HumanPose> poses;
    bool isLastFrame = false;
    bool isAsyncMode = false; // execution is always started in SYNC mode
    bool isModeChanged = false; // set to true when execution mode is changed (SYNC<->ASYNC)
    bool blackBackground = false;

    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    auto total_t0 = std::chrono::high_resolution_clock::now();
    auto wallclock = std::chrono::high_resolution_clock::now();
    double render_time = 0;
        //init_flag =true;
    ros::Rate loop_rate(30);
    cv::namedWindow("HUMAN_POSE");

    while (ros::ok()) {
            if(!init_flag)
            {
                if(!Color_pic.empty())
                {
                    curr_frame = Color_pic.clone();
                    estimator.reshape(curr_frame);  // Do not measure network reshape, if it happened
                    init_flag =true;
                }
            }
            else{
                auto t0 = std::chrono::high_resolution_clock::now();
                //here is the first asynchronus point:
                //in the async mode we capture frame to populate the NEXT infer request
                //in the regular mode we capture frame to the current infer request
                next_frame = Color_pic.clone();
                std::cout<<"received pic!"<<std::endl;


                if (isAsyncMode) {
                    if (isModeChanged) {
                        estimator.frameToBlobCurr(curr_frame);
                    }
                    if (!isLastFrame) {
                        estimator.frameToBlobNext(next_frame);
                    }
                } else if (!isModeChanged) {
                    estimator.frameToBlobCurr(curr_frame);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

                t0 = std::chrono::high_resolution_clock::now();
                // Main sync point:
                // in the trully Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
                // in the regular mode we start the CURRENT request and immediately wait for it's completion
                if (isAsyncMode) {
                    if (isModeChanged) {
                        estimator.startCurr();
                    }
                    if (!isLastFrame) {
                        estimator.startNext();
                    }
                } else if (!isModeChanged) {
                    estimator.startCurr();
                }

                if (estimator.readyCurr()) {
                    t1 = std::chrono::high_resolution_clock::now();
                    ms detection = std::chrono::duration_cast<ms>(t1 - t0);
                    t0 = std::chrono::high_resolution_clock::now();
                    ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                    wallclock = t0;

                    t0 = std::chrono::high_resolution_clock::now();

                    if (true) {
                        if (blackBackground) {
                            curr_frame = cv::Mat::zeros(curr_frame.size(), curr_frame.type());
                        }
                        std::ostringstream out;
                        out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                            << (decode_time + render_time) << " ms";

                        cv::putText(curr_frame, out.str(), cv::Point2f(0, 25),
                                    cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                        out.str("");
                        out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                        out << std::fixed << std::setprecision(2) << wall.count()
                            << " ms (" << 1000.f / wall.count() << " fps)";
                        cv::putText(curr_frame, out.str(), cv::Point2f(0, 50),
                                    cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                        if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                            out.str("");
                            out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                            << " ms ("
                            << 1000.f / detection.count() << " fps)";
                            cv::putText(curr_frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                                cv::Scalar(255, 0, 0));
                        }
                    }

                    poses = estimator.postprocessCurr();

                    if (false) {
                        if (!poses.empty()) {
                            std::time_t result = std::time(nullptr);
                            char timeString[sizeof("2020-01-01 00:00:00: ")];
                            std::strftime(timeString, sizeof(timeString), "%Y-%m-%d %H:%M:%S: ", std::localtime(&result));
                            std::cout << timeString;
                         }

                        for (HumanPose const& pose : poses) {
                            std::stringstream rawPose;
                            rawPose << std::fixed << std::setprecision(0);
                            for (auto const& keypoint : pose.keypoints) {
                                rawPose << keypoint.x << "," << keypoint.y << " ";
                            }
                            rawPose << pose.score;
                            std::cout << rawPose.str() << std::endl;
                        }
                    }

                    if (true) {
                        presenter.drawGraphs(curr_frame);
                        renderHumanPose(poses, curr_frame);
                        cv::imshow("Human Pose Estimation on CPU", curr_frame);
                        cv::waitKey(3);
                        //sensor_msgs::ImagePtr
                        //img_pub.publish();
                        t1 = std::chrono::high_resolution_clock::now();
                        render_time = std::chrono::duration_cast<ms>(t1 - t0).count();
                    }
                }

                if (isLastFrame) {
                    break;
                }

                if (isModeChanged) {
                    isModeChanged = false;
                }

                // Final point:
                // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
                curr_frame = next_frame.clone();
                next_frame = cv::Mat();
                if (isAsyncMode) {
                    estimator.swapRequest();
                }
            }

            ros::spinOnce();
            loop_rate.sleep();

     }




        return 0;

}
