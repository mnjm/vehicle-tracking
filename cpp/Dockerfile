FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt .
COPY build.sh .
COPY include/ include/
COPY src/ src/

RUN chmod +x build.sh

RUN ./build.sh

CMD bash -c 'for video in /app/data/*; do \
    if [[ -f "$video" ]]; then \
        echo "Processing: $video"; \
        ./build/vehicle_roi_timer --video "$video"; \
    fi \
done'
