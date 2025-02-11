FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    build-essential cmake git wget unzip pkg-config \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libopenexr-dev libdc1394-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libx265-dev \
    libgtk-3-dev libcanberra-gtk3-dev \
    libtbb-dev gfortran \
    python3 python3-pip python3-dev python3-numpy \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION=4.11.0
RUN mkdir /opencv && cd /opencv && \
    git clone --depth=1 https://github.com/opencv/opencv.git && \
    git clone --depth=1 https://github.com/opencv/opencv_contrib.git && \
    mkdir /opencv/build && cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib/modules \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_OPENGL=ON \
          -D WITH_LAPACK=ON \
          -D WITH_OPENBLAS=ON \
          -D OPENCV_DNN_OPENVINO=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=ON \
          -D OPENCV_DNN_CPU_BASELINE=AVX2 \
          ../opencv && \
    make -j$(nproc) && make install && ldconfig

RUN rm -rf /opencv

RUN python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

WORKDIR /app

COPY CMakeLists.txt .
COPY build.sh .
COPY include/ include/
COPY src/ src/

RUN chmod +x build.sh

RUN ./build.sh

CMD bash -c 'for video in /app/data/*.mp4; do \
    if [[ -f "$video" ]]; then \
        echo "Processing: $video"; \
        ./vehicle_roi_timer --video "$video"; \
    fi \
done'
