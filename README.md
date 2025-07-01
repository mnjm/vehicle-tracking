# Vehicle ROI Timer

A tool to detect vehicles within a Region of Interest (ROI) in videos and display their dwell time.

## Objective

- Detect vehicles within a predefined ROI in video footage
- Calculate and display the time each vehicle spends in the ROI
- Generate an output video showing:
  - The ROI boundary
  - Vehicle detections
  - Timers for each vehicle

## Demo

### Example 1
<p styles="font-size: 2em; font-weight:bold;" align="center"><a href="https://youtu.be/qanK6EtrPQE">Click to play</a></p>

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=qanK6EtrPQE" target="_blank">
 <img src="http://img.youtube.com/vi/qanK6EtrPQE/mqdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a></p>


### Example 2
<p styles="font-size: 2em; font-weight:bold;" align="center"><a href="https://www.youtube.com/watch?v=195PMzpf240">Click to play</a></p>

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=195PMzpf240" target="_blank">
 <img src="http://img.youtube.com/vi/195PMzpf240/mqdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a></p>

## Installation & Usage

### Option 1: Native Build

#### Requirements
- OpenCV (4.11.0+) with DNN support
- GCC
- CMake
- Make

#### Build Instructions
```bash
# Fresh build (with optional flags)
./build.sh fresh [--debug] [--cuda] [--opencl]

# Clean build artifacts
./build.sh clean
```

**Build Options:**
- `--debug`: Enable debug mode
- `--cuda`: Enable CUDA acceleration
- `--opencl`: Enable OpenCL acceleration

### Option 2: Docker
```bash
docker compose up --build -d
docker compose logs -f  # View logs
```

**Note:** Docker build includes OpenCV compilation and may take significant time.

## Running the Application

```bash
./vehicle_roi_timer --video <path_to_video> [options]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--video` | Path to input video (required) | - |
| `--model` | Path to YOLO model file | `./data/yolo11s.onnx` |
| `--output_dir` | Output directory | `./output` |
| `--no_display` | Disable video display | False |
| `-h, --help` | Show help message | - |
| `-v, --version` | Show version info | - |

**Model Notes:**
- Use `yolo11n.onnx` for low-power devices
- Model files should be placed in `./data/`

## Implementation Details

### Tracking Approaches
1. **Geometric-based Tracking**
   - Centroid Tracking
   - ROI/IoU-based Tracking (implemented)

2. **Feature-based Tracking**
   - Color Histogram Matching
   - Feature Descriptors (SIFT, ORB)
   - Deep Learning Embeddings

3. **Ensemble Methods**
   - Hybrid Approaches
   - Kalman Filter-based
   - DeepSORT

### Current Implementation
- Uses YOLO for vehicle detection
- Implements IoU-based geometric tracking
- Time calculation based on video FPS

## Known Limitations

| Issue | Example | Impact |
|-------|---------|--------|
| YOLO detection errors | False negatives/positives | Missed vehicles or incorrect timers |
| Vehicle occlusions | `overlapping.mp4` | Tracking failures |
| Fast-moving vehicles | `high-speed.mp4` | Tracking inaccuracies |
| Variable FPS | Frame drops | Time calculation errors |
| Poor visibility | `out-of-focus.mp4` | Reduced detection accuracy |

## Additional Resources

- [Python prototype](./python/vehicle_roi_timer.ipynb) with centroid tracking
- [OpenCV DNN documentation](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
- [YOLO model information](https://github.com/ultralytics/yolov5)
