#ifndef IMAGE_LOAD_CUH
#define IMAGE_LOAD_CUH

#include <opencv2/opencv.hpp>

struct ImageData {
    cv::Mat original_frame;
    cv::cuda::GpuMat* image_ref = nullptr;    // Original GPU image
    cv::cuda::GpuMat* denoised_ref = nullptr; // Denoised GPU image
    cv::cuda::GpuMat* rotation_ref = nullptr; // Rotated GPU image
    cv::cuda::GpuMat* binary_ref = nullptr;   // Binarized GPU image 
    cv::cuda::GpuMat* annotated_ref = nullptr;   // Visual output with annotations
    std::string outputPath;

    std::chrono::high_resolution_clock::time_point processing_start_time;

};

void loadImageToGPU(const cv::Mat& img, ImageData& imgData);

#endif // IMAGE_LOAD_CUH
