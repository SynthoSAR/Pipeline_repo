#ifndef IMAGE_LOAD_CUH
#define IMAGE_LOAD_CUH

#include <opencv2/opencv.hpp>

struct ImageData {
    unsigned char* d_image = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    std::string outputPath;
};

void loadImageToGPU(const cv::Mat& img, ImageData& imgData);

#endif // IMAGE_LOAD_CUH
