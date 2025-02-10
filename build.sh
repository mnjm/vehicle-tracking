#!/bin/bash

BUILD_TYPE="Release"
MACRO_DEFINITION=""
FRESH_BUILD=false
CLEAN_BUILD=false

for arg in "$@"; do
    case "$arg" in
        fresh)
            FRESH_BUILD=true
            ;;
        clean)
            CLEAN_BUILD=true
            ;;
        --debug)
            BUILD_TYPE="Debug"
            ;;
        --cuda)
            if [[ "$MACRO_DEFINITION" == "-DOPENCL_ACC=ON" ]]; then
                echo "Error: --cuda and --opencl are mutually exclusive."
                exit 1
            fi
            MACRO_DEFINITION="-DCUDA_ACC=ON"
            ;;
        --opencl)
            if [[ "$MACRO_DEFINITION" == "-DCUDA_ACC=ON" ]]; then
                echo "Error: --cuda and --opencl are mutually exclusive."
                exit 1
            fi
            MACRO_DEFINITION="-DOPENCL_ACC=ON"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [fresh | clean] [--debug] [--cuda | --opencl]"
            exit 1
            ;;
    esac
done

if $FRESH_BUILD && $CLEAN_BUILD; then
    echo "Error: 'fresh' and 'clean' cannot be used together."
    exit 1
fi

if $CLEAN_BUILD; then
    echo "Cleaning build directory..."
    rm -rf build
    exit 0
fi

if $FRESH_BUILD; then
    echo "Performing a fresh build..."
    rm -rf build
fi

mkdir -p build
cd build || exit 1

echo "Building in $BUILD_TYPE mode..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE $MACRO_DEFINITION ..
make -j$(nproc)

mv vehicle_roi_timer ../

cd ..