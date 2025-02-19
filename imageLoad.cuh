#ifndef IMAGE_LOAD_CUH
#define IMAGE_LOAD_CUH

#include <opencv2/opencv.hpp>

struct ImageData {
    cv::cuda::GpuMat* image_ref = nullptr;        // Raw image reference in GPU
    cv::cuda::GpuMat* denoised_ref = nullptr;        // Denoised image reference in GPU
    cv::cuda::GpuMat* rotation_ref = nullptr;        // Rotation corrected image reference in GPU
    std::string outputPath;
};

void loadImageToGPU(const cv::Mat& img, ImageData& imgData);

#endif // IMAGE_LOAD_CUH
