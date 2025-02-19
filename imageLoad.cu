#include "imageLoad.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <opencv2/cudaimgproc.hpp>  // For cv::cuda::cvtColor

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void loadImageToGPU(const cv::Mat& imG, ImageData& imgData) {

    cv::cuda::GpuMat* gpu_img = new cv::cuda::GpuMat();
    gpu_img->upload(imG);   // Upload image to GPU

    cv::cuda::cvtColor(*gpu_img, *gpu_img, cv::COLOR_BGR2GRAY); // Convert to grayscale on GPU
    
    imgData.image_ref = gpu_img;
    
}
