# Vehicle ROI Timer

## Objective

- Given a video, detect vehicles inside a ROI in the video and display the time that they are present in the ROI.
    - ROI is read from a text file using the video path. (`<video_path>_roi.txt`)
    - Output should a video file, with ROI, Vehicle detection and timer being displayed.

## How to Build and Run?

You can build and run the app in 2 ways
1. Natively building the app
2. Using Docker

### Natively Building the app

#### Requirements

- Prebuilt Open-CV with DNN support
- GCC
- CMake
- Make

#### Build

I have provided a handy [script](./build.sh) to build the app. Given the requirements are satisfied, just run the script to build the app.

The script supports a few modifiers
- `./build.sh fresh` - Fresh builds the app
- `./build.sh clean` - Cleans the app's build artifacts

Options
- `./build.sh <fresh> --debug` - Enables debug mode
- `./build.sh <fresh> <--debug> --cuda` - Enables CUDA Acceleration in OpenCv's DNN
- `./build.sh <fresh> <--debug> --opencl` - Enables OPENCL Acceleration in OpenCv's DNN

This creates a `vehicle_roi_timer` binary.

#### To Run

```bash
./vehicle_roi_timer [--help] [--version] --video VAR [--model VAR] [--output_dir VAR] [--no_display]
```

**Arguments**
- `--video VAR`
  Path to video file to process **[required]**.
- `--model VAR`
  Path to the YOLO model file
  **Default:** `./data/yolo11s.onnx`
- `--output_dir VAR`
  Path to output directory
  **Default:** `./output`
- `--no_display`
  Do not display output.
- `-h, --help`
  Shows help message and exits.
- `-v, --version`
  Prints version information and exits.

### Docker

To build and run

```
docker-compose up --build -d
```

**Note**: Input videos and Model files are fed from `./data/` which is mounted as a docker volume. Output videos are also mounted to docker from `./output`. If you have videos in `./output` it will be overwritten.

## Implementation details

Problem breakdown
1. Detect Vehicles in the ROI
2. Track the vehicles
3. Calculate the time that vehicle is present in the ROI

- (1) can use solved by using a object detection networks (like YOLO).
- (2) There are number of ways to track vehicles in subsequent frames, which is described below.
- (3) Since it is a video source with consistent FPS, time can be calculated based the FPS rate of the video
    - `time (in sec) = frame count / fps`

### Tracking Vehicles

#### Approaches
1. Geometric-based Tracking
2. Feature-based Tracking
3. Ensemble Methods

Due to the time crunch, The one I have implemented here is Geometric-based **ROI / IoU Based Tracking**. I did try **Centroid Tracking** which did not work well for the videos I run on.

### 1. Geometric-based Tracking
This approach relies on object position, movement, and shape.

#### Centroid Tracking
- Tracks objects by computing their centroids in consecutive frames.
- Pros:
  - Simple and efficient.
  - Works well in less crowded scenes.
- Cons:
  - Fails when objects overlap significantly.
  - Cannot handle occlusions well.

#### ROI / IoU-based Tracking
- Uses Intersection over Union (IoU) to match objects across frames.
- Pros:
  - Effective in structured environments with minimal occlusion.
  - Simple implementation.
- Cons:
  - Struggles with occlusion and fast-moving objects.
  - May fail when objects move significantly between frames.

### 2. Feature-based Tracking
This method relies on identifying and matching key object features.

- Color Histogram Matching
  - Uses color distribution to track objects across frames.
  - Pros:
    - Robust to slight variations in object position.
    - Works well for objects with unique colors.
  - Cons:
    - Fails when objects have similar colors.
    - Sensitive to lighting changes.
- Using Feature Descriptors
  - Extracts keypoints (e.g., SIFT, ORB) and matches them between frames.
  - Pros:
    - Works well in cluttered scenes.
    - Can handle slight occlusions.
  - Cons:
    - Computationally expensive.
    - Struggles with drastic appearance changes.
- Using Deep Learning Features/Embeddings
  - Extracts high-level features using deep learning models for better tracking in complex scenes.
  - Pros:
    - More robust to occlusions and appearance variations.
    - Works well in complex environments.
  - Cons:
    - Requires a pre-trained model.
    - Slower than traditional methods.

### 3. Ensemble Methods
Combines multiple approaches to improve robustness.

- Hybrid Approaches
  - Uses a mix of geometric and feature-based tracking for better accuracy.
- Kalman Filter-based Tracking
  - Predicts object motion using state estimation and corrects based on observations.
  - Pros:
    - Handles motion smoothly.
    - Works well in moderate occlusions.
  - Cons:
    - Assumes constant motion, which may not always hold true.
    - Requires accurate initial detection.
- DeepSORT (Simple Online and Realtime Tracker)
  - A state-of-the-art multi-object tracker combining deep learning embeddings and Kalman filtering.
  - Pros:
    - Robust and widely used in real-time tracking applications.
    - Handles occlusions better than traditional methods.
  - Cons:
    - Computationally intensive.

## Edge cases

There are the edge cases that plagues the current implementation

1. Irregularities in YOLO detections
    - YOLO maynot detect all vehicles in the ROI (False Negatives)
    - YOLO may generate inaccurate bounding boxes
    - False Positives
2. Tracking Edge cases
    - Occlusions and Overlapping of Vehicles
        - This can be observed in this video `./roi_iou_output/overlappinp.mp4`
    - Fast moving vehicles - Geometric tracking is prone to fail if the vehicles move fast
        - This can be observed in this video `./roi_iou_output/high-speed.mp4`
3. Irregularities in FPS can cause miscalculations in time
    - Inconsistent Frame Rates
    - Frame lag
4. Other
    - Camera Angles / Placement
    - Vehicle is partially seen in the ROI
    - Camera conditions - Low resolution, out of focus, blurryness
        - Example `./roi_iou_output/out-of-focus.mp4`
    - Changes in Lighting conditions
    - Rain, Fog or dust etc

## Addendum

- I initially wrote a rough version of this app in python, which you can see here ./python/vehicle_roi_timer.ipynb
