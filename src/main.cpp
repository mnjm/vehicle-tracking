#include "yolov11.hpp"
#include "vehicle_roi_timer.hpp"
#include <iostream>
#include "argparse.hpp"
#include "debug.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser parser("Vehicle ROI Timer");
    parser.add_argument("--video").help("Path to video file to process").required();
    parser.add_argument("--model").help("Path to the YOLO model file").default_value("./data/yolo11s.onnx");
    parser.add_argument("--output_dir").help("Path to output directory").default_value("./output");
    parser.add_argument("--no_display").help("Do not display").default_value(false).implicit_value(true);
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
    std::string output_dir = parser.get<std::string>("--output_dir");
    bool display_b = ! parser.get<bool>("--no_display");
    assert(std::filesystem::exists(video_path) && "Video file does not exists!");

    // Create output directory if it doesn't exist
    if ("" != output_dir) {
        std::filesystem::create_directories(output_dir);
    }
    YOLOv11 model(
        model_path,
        0.45f,
        0.45f,
        [](int lbl_id, const std::string lbl)
        { return lbl_id >= 0 && lbl_id <= 8; });
    process_video(model, video_path, output_dir, display_b);

    return 0;
}