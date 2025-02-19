#include "imageSave.cuh"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

void saveImageFromGPU(const ImageData& imgData) {
    if (!imgData.rotation_ref) {
        std::cerr << "Error: imgData.ref is null!" << std::endl;
        return;
    }
    if (imgData.rotation_ref->empty()) {
        std::cerr << "Error: GPU Mat is empty!" << std::endl;
        return;
    }
    
    cv::Mat img;
    imgData.rotation_ref->download(img);
    
    if (!cv::imwrite(imgData.outputPath, img)) {
        std::cerr << "Failed to save the image to: " << imgData.outputPath << std::endl;
        exit(EXIT_FAILURE);
    }

}

void freeGPUData(ImageData& imgData) {
    // Free GpuMat pointer
    if (imgData.image_ref) {
        delete imgData.image_ref;   
        imgData.image_ref = nullptr;
    }

    if (imgData.denoised_ref) {
        delete imgData.denoised_ref;
        imgData.denoised_ref = nullptr;
    }

    if (imgData.rotation_ref) {
        delete imgData.rotation_ref;
        imgData.rotation_ref = nullptr;
    }
}

