#ifndef IMAGE_LOAD_CUH
#define IMAGE_LOAD_CUH

#include <opencv2/opencv.hpp>

struct ImageData {
    cv::cuda::GpuMat* image_ref = nullptr;    // Original GPU image
    cv::cuda::GpuMat* denoised_ref = nullptr; // Denoised GPU image
    cv::cuda::GpuMat* rotation_ref = nullptr; // Rotated GPU image
    cv::cuda::GpuMat* binary_ref = nullptr;   // Binarized GPU image 
    std::string outputPath;
};

void loadImageToGPU(const cv::Mat& img, ImageData& imgData);

#endif // IMAGE_LOAD_CUH
