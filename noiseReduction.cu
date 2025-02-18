#include "noiseReduction.cuh"

void applyNoiseReduction(ImageData& imgData) {
    cv::cuda::GpuMat d_image(imgData.height, imgData.width, (imgData.channels == 3) ? CV_8UC3 : CV_8UC1, imgData.d_image);
    cv::cuda::GpuMat d_denoised;
    
    float h = 8.0f;  // Filter strength
    int search_window = 21;  // Search window size
    int block_size = 7;  // Block size

    cv::cuda::fastNlMeansDenoising(d_image, d_denoised, h, search_window, block_size);

    imgData.denoised_ref = new cv::cuda::GpuMat();

    d_denoised.copyTo(*imgData.denoised_ref);


}

