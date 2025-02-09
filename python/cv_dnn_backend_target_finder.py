import cv2


# backend_map = {
#     cv2.dnn.DNN_BACKEND_OPENCV: "OpenCV",
#     cv2.dnn.DNN_BACKEND_CUDA: "CUDA",
#     cv2.dnn.DNN_BACKEND_TIMVX: "TIM-VX",
#     cv2.dnn.DNN_BACKEND_VKCOM: "Vulkan",
#     cv2.dnn.DNN_BACKEND_CANN: "Huawei Ascend",
# }
#
target_map = {
    cv2.dnn.DNN_TARGET_CPU: "CPU",
    cv2.dnn.DNN_TARGET_CUDA: "CUDA",
    cv2.dnn.DNN_TARGET_CUDA_FP16: "CUDA FP16",
    cv2.dnn.DNN_TARGET_OPENCL: "OpenCL",
    cv2.dnn.DNN_TARGET_OPENCL_FP16: "OpenCL FP16",
}

targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_OPENCV)
print("  Supported Targets:")
for target in targets:
    print(f"    - {target_map.get(target, 'Unknown')}")

has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
print("Cuda:", has_cuda)

has_opencl = cv2.ocl.haveOpenCL()
print("OpenCL:", has_opencl)

backends = [ cv2.dnn.DNN_BACKEND_OPENCV ]
if has_cuda:
    backends += [ cv2.dnn.DNN_BACKEND_CUDA ]
