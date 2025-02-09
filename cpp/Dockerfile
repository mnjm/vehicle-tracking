FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libopencv-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p build

RUN cd build && cmake .. && make

./vehicle_roi_timer --video ./data/9521384-uhd_3840_2160_15fps.mp4 --model ./data/yolo11s.onnx

CMD ["./build/vehicle_roi_timer", "--video", "/app/data/5473765-uhd_3840_2160_24fps.mp4", "--model", "/app/data/yolo11s.onnx"]