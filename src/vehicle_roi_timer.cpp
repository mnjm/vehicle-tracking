#include "vehicle_roi_timer.hpp"
#include "debug.hpp"

#define TOTAL_COLORS 30
#define IOU_TRESH 0.45f
#define HISTORY_LIMIT 5

cv::Scalar generate_unique_colors()
{
    static int color_idx = 0;
    const int total_colors = TOTAL_COLORS;
    int hue = (color_idx * 180 / total_colors) % 180;
    color_idx++;
    cv::Mat color(1, 1, CV_8UC3, cv::Scalar(hue, 200, 255));
    cv::cvtColor(color, color, cv::COLOR_HSV2BGR);
    return cv::Scalar(color.at<cv::Vec3b>(0, 0));
}

std::string format_time(float secs)
{
    int mins = static_cast<int>(secs / 60);
    float remaining_secs = std::fmod(secs, 60.0f);
    std::string ret_str = std::to_string(remaining_secs);
    ret_str = ret_str.substr(0, ret_str.find('.') + 3);
    ret_str = std::to_string(mins) + ":" + ret_str;
    return ret_str;
}

void process_video(YOLOv11 &model, const std::string &video_path, const std::string &output_dir, bool display)
{
    cv::VideoCapture cap(video_path);
    assert(cap.isOpened() && "Error reading video file");

    DEBUG_PRINT("Processing " << video_path);

    int vid_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int vid_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    DEBUG_PRINT("Video WxH:" << vid_w << "x" << vid_h << " FPS:" << fps);

    // Read ROI from file
    std::string roi_path = video_path.substr(0, video_path.find_last_of('.')) + "_roi.txt";
    assert(std::filesystem::exists(roi_path) && "ROI file not found");
    std::vector<int> roi(4);
    std::ifstream roi_file(roi_path);
    for (int i = 0; i < 4; i++)
    {
        roi_file >> roi[i];
    }
    DEBUG_PRINT("ROI Read" << roi[0] << " " << roi[1] << " " << roi[2] << " " << roi[3]);

    // Setup output video
    cv::VideoWriter out_vid;
    bool save_video_b = output_dir != "";
    if (true == save_video_b) {
        std::filesystem::path output_path = std::filesystem::path(output_dir) /
                                            std::filesystem::path(video_path).filename();
        out_vid.open(output_path.string(),
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                fps, cv::Size(vid_w, vid_h));
        
        assert(out_vid.isOpened() && "Error Could not create output video");
    }

    std::vector<Detection> det_l;
    cv::Scalar roi_color(255, 0, 0);

    cv::Mat frame;
    while (cap.read(frame))
    {
        cv::Mat img = frame(cv::Rect(roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]));
        auto bbox_l = model.detect(img);

        // Draw ROI
        cv::rectangle(frame, cv::Point(roi[0], roi[1]),
                        cv::Point(roi[2], roi[3]), roi_color, 2);

        std::vector<Detection> new_det_l;
        for (size_t bbox_idx = 0; bbox_idx < bbox_l.size(); bbox_idx++)
        {
            const auto &bbox = bbox_l[bbox_idx];

            // Find closest existing detection
            float max_iou = 0.0f;
            int max_idx = -1;
            for (size_t idx = 0; idx < det_l.size(); idx++)
            {
                float iou = calculateIoU(bbox, det_l[idx].bbox);
                if (iou > max_iou)
                {
                    max_iou = iou;
                    max_idx = idx;
                }
            }

            if (max_idx >= 0 && max_iou >= IOU_TRESH) {
                det_l[max_idx].bbox = bbox;
                ++det_l[max_idx].frame_count;
                det_l[max_idx].dangling_frame_count = 0;
                new_det_l.push_back(det_l[max_idx]);
                det_l.erase(det_l.begin() + max_idx);
            } else {
                cv::Scalar color = generate_unique_colors();
                Detection new_det(
                    color,
                    1,
                    bbox
                );
                new_det_l.push_back(new_det);
            }
        }

        for (auto det : det_l) {
            ++det.dangling_frame_count;
            ++det.frame_count;
            if (det.dangling_frame_count <= HISTORY_LIMIT) {
                new_det_l.push_back(det);
            }
        }
        det_l.clear();
        det_l = new_det_l;

        // Draw detections
        for (const auto &det : det_l)
        {
            if (det.dangling_frame_count > 0) continue;
            const auto &bbox = det.bbox;
            cv::Point p1(roi[0] + bbox.x1, roi[1] + bbox.y1);
            cv::Point p2(roi[0] + bbox.x2, roi[1] + bbox.y2);
            cv::rectangle(frame, p1, p2, det.color, 2);

            // Draw time
            float t = static_cast<float>(det.frame_count) / fps;
            std::string t_str = format_time(t);

            cv::Point bbox_center((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
            int bbox_height = p2.y - p1.y;
            double font_scale = std::max(0.5, bbox_height / 200.0);
            int font_thickness = std::max(1, static_cast<int>(font_scale * 2));

            int baseline;
            cv::Size txt_size = cv::getTextSize(t_str, cv::FONT_HERSHEY_SIMPLEX,
                                                font_scale, font_thickness, &baseline);
            cv::Point txt_pos(bbox_center.x - txt_size.width / 2,
                                bbox_center.y + txt_size.height / 2);
            cv::Point bgP1(txt_pos.x, txt_pos.y - txt_size.height - baseline);
            cv::Point bgP2(txt_pos.x + txt_size.width, txt_pos.y + baseline);
            cv::rectangle(frame, bgP1, bgP2, cv::Scalar(0, 0, 0), cv::FILLED);

            cv::putText(frame, t_str, txt_pos, cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
        }

        if (true == save_video_b) {
            out_vid.write(frame);
        }

        if (display) {
            cv::imshow("show", frame);
            char key = cv::waitKey(1);
            if (key == 'q')
                break;
        }
    }

    cap.release();
    if (true == save_video_b) {
        out_vid.release();
    }
    cv::destroyAllWindows();
}