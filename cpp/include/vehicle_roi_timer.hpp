#pragma once

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include "yolov11.hpp"

struct Detection
{
    cv::Scalar color;
    int frame_count;
    cv::Point2f center;
    size_t bbox_idx;
};

void process_video(YOLOv11 &model, const std::string &video_path, const std::string &output_dir);