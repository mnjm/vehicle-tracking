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
    ObjectBBox bbox;
    int dangling_frame_count; // Frame count where detections are not found
    Detection(cv::Scalar color_, int frame_count_, ObjectBBox bbox_, int dangling_frame_count_=0) : color(color_), frame_count(frame_count_), bbox(bbox_), dangling_frame_count(dangling_frame_count_) {};
};

void process_video(YOLOv11 &model, const std::string &video_path, const std::string &output_dir="", bool display=true);