#include "vehicle_roi_timer.hpp"

#define TOTAL_COLORS 30
#define DIST_THRESH 40.0f

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

void process_video(YOLOv11 &model, const std::string &video_path, const std::string &output_dir)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Reading " << video_path << " video failed." << std::endl;
        return;
    }

    int vid_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int vid_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    std::cout << "Video WxH:" << vid_w << "x" << vid_h << " FPS:" << fps << std::endl;

    // Read ROI from file
    std::string roi_path = video_path.substr(0, video_path.find_last_of('.')) + "_roi.txt";
    std::vector<int> roi(4);
    std::ifstream roi_file(roi_path);
    for (int i = 0; i < 4; i++)
    {
        roi_file >> roi[i];
    }

    // Setup output video
    std::filesystem::path output_path = std::filesystem::path(output_dir) /
                                        std::filesystem::path(video_path).filename();
    cv::VideoWriter out_vid(output_path.string(),
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps, cv::Size(vid_w, vid_h));

    std::vector<Detection> det_l;
    const float dist_thresh = DIST_THRESH;
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
            cv::Point2f center((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2);

            // Find closest existing detection
            float min_dist = std::numeric_limits<float>::max();
            int min_idx = -1;
            for (size_t idx = 0; idx < det_l.size(); idx++)
            {
                float dist = cv::norm(center - det_l[idx].center);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    min_idx = idx;
                }
            }

            Detection new_det;
            new_det.center = center;
            new_det.bbox_idx = bbox_idx;

            if (min_dist < dist_thresh && min_idx >= 0)
            {
                new_det.color = det_l[min_idx].color;
                new_det.frame_count = det_l[min_idx].frame_count + 1;
            }
            else
            {
                new_det.color = generate_unique_colors();
                new_det.frame_count = 1;
            }
            new_det_l.push_back(new_det);
        }

        det_l = new_det_l;

        // Draw detections
        for (const auto &det : det_l)
        {
            const auto &bbox = bbox_l[det.bbox_idx];
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
                        font_scale, det.color, font_thickness, cv::LINE_AA);
        }

        out_vid.write(frame);
        cv::imshow("show", frame);

        char key = cv::waitKey(1);
        if (key == 'n')
            break;
        if (key == 'q')
            return;
    }

    cap.release();
    out_vid.release();
    cv::destroyAllWindows();
}