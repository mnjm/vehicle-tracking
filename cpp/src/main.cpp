#include "yolov11.hpp"
#include "vehicle_roi_timer.hpp"
#include <iostream>

int main()
{
    std::string input_dir = "../input_videos";
    std::string output_dir = "./out_vid_dir";

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);

    YOLOv11 model("../data/yolo11s.onnx");

    // Process all MP4 files in input directory
    for (const auto &entry : std::filesystem::directory_iterator(input_dir))
    {
        if (entry.path().extension() == ".mp4")
        {
            process_video(model, entry.path().string(), output_dir);
        }
    }

    return 0;
}