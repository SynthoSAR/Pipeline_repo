#include "noiseReduction.cuh"

void applyNoiseReduction(ImageData& imgData) {
    cv::cuda::GpuMat* d_image = imgData.image_ref;

    cv::cuda::GpuMat* denoised_image = new cv::cuda::GpuMat();
    
    float h = 8.0f;  // Filter strength
    int search_window = 21;  // Search window size
    int block_size = 7;  // Block size

    cv::cuda::fastNlMeansDenoising(*d_image, *denoised_image, h, search_window, block_size);

    imgData.denoised_ref = denoised_image;


}

