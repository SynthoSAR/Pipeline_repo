#ifndef IMAGE_LOAD_CUH
#define IMAGE_LOAD_CUH

#include <opencv2/opencv.hpp>

struct ImageData {
    unsigned char* d_image = nullptr;       // Original image in GPU
    cv::cuda::GpuMat* denoised_ref = nullptr;        // Denoised image reference in GPU
    int width = 0;
    int height = 0;
    int channels = 0;
    std::string outputPath;
};

void loadImageToGPU(const cv::Mat& img, ImageData& imgData);

#endif // IMAGE_LOAD_CUH
