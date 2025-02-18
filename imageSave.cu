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
    size_t imageSize = imgData.width * imgData.height * imgData.channels * sizeof(unsigned char);
    unsigned char* h_image = new unsigned char[imageSize];

    CHECK_CUDA_ERROR(cudaMemcpy(h_image, imgData.d_image, imageSize, cudaMemcpyDeviceToHost));

    cv::Mat img(imgData.height, imgData.width, (imgData.channels == 3) ? CV_8UC3 : CV_8UC1, h_image);

    if (!cv::imwrite(imgData.outputPath, img)) {
        std::cerr << "Failed to save the image to: " << imgData.outputPath << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Image successfully saved to: " << imgData.outputPath << std::endl;

    //std::this_thread::sleep_for(std::chrono::seconds(2));

    delete[] h_image;
}

void freeGPUData(ImageData& imgData) {
    if (imgData.d_image) {
        CHECK_CUDA_ERROR(cudaFree(imgData.d_image));
        imgData.d_image = nullptr;
    }
}
