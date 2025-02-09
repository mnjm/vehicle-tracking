#include "yolov11.hpp"
#include "vehicle_roi_timer.hpp"
#include <iostream>
#include "argparse.hpp"
#include "debug.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser parser("Vehicle ROI Timer");
    parser.add_argument("--video").help("Path to video file to process").required();
    parser.add_argument("--model").help("Path to the YOLO model file").required();

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << "Error: " << err.what() << "\n";
        std::cerr << parser;
        return 1;
    }
    std::string video_path = parser.get<std::string>("--video");
    std::string model_path = parser.get<std::string>("--model");
    assert(std::filesystem::exists(video_path) && "Video file does not exists!");

    std::string output_dir = "./out_vid_dir";

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    YOLOv11 model(model_path);
    process_video(model, video_path, output_dir);

    return 0;
}