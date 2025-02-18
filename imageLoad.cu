#include "imageLoad.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void loadImageToGPU(const cv::Mat& img, ImageData& imgData) {
    imgData.width = img.cols;
    imgData.height = img.rows;
    imgData.channels = img.channels();

    size_t imageSize = img.total() * img.elemSize(); 
    CHECK_CUDA_ERROR(cudaMalloc(&imgData.d_image, imageSize));
    if (imgData.d_image == nullptr) {
        std::cerr << "Error: Memory allocation failed!" << std::endl;
        return;  // exit or handle the error
    }
    if (img.data == nullptr) {
        std::cerr << "Error: Image data is null!" << std::endl;
        return;  // exit or handle the error
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(imgData.d_image, img.data, imageSize, cudaMemcpyHostToDevice));


}
