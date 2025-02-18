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
    if (!imgData.denoised_ref) {
        std::cerr << "Error: imgData.ref is null!" << std::endl;
        return;
    }
    if (imgData.denoised_ref->empty()) {
        std::cerr << "Error: GPU Mat is empty!" << std::endl;
        return;
    }
    
    cv::Mat img;
    imgData.denoised_ref->download(img);

    if (!cv::imwrite(imgData.outputPath, img)) {
        std::cerr << "Failed to save the image to: " << imgData.outputPath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Image successfully saved to: " << imgData.outputPath << std::endl;

}

void freeGPUData(ImageData& imgData) {
    if (imgData.d_image) {
        CHECK_CUDA_ERROR(cudaFree(imgData.d_image));
        imgData.d_image = nullptr;
    }
}
