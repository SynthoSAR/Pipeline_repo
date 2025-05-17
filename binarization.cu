#include "binarization.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define HIST_BINS 256 // Number of histogram bins (0-255 for 8-bit grayscale)

// CUDA kernel to compute histogram
__global__ void computeHistogramKernel(const unsigned char* input, int width, int height, unsigned int* histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char value = input[idx];
        atomicAdd(&histogram[value], 1);
    }
}

// Host function to compute Otsu threshold from histogram
double computeOtsuThreshold(unsigned int* hist, int total_pixels) {
    float sum = 0;
    for (int i = 0; i < HIST_BINS; ++i) {
        sum += i * hist[i];
    }

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float maxVariance = 0;
    int threshold = 0;

    for (int t = 0; t < HIST_BINS; ++t) {
        wB += hist[t];              // Weight Background
        if (wB == 0) continue;

        wF = total_pixels - wB;     // Weight Foreground
        if (wF == 0) break;

        sumB += t * hist[t];

        float mB = sumB / wB;       // Mean Background
        float mF = (sum - sumB) / wF; // Mean Foreground

        // Between-class variance
        float variance = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = t;
        }
    }

    return static_cast<double>(threshold);
}

void applyBinarization(ImageData& imgData) {
    if (imgData.rotation_ref == nullptr) {
        std::cerr << "Error: Rotation reference is null!" << std::endl;
        return;
    }

    cv::cuda::GpuMat* rotated_image = imgData.rotation_ref;
    cv::cuda::GpuMat* binary_image = new cv::cuda::GpuMat();

    // Ensure the image is grayscale
    cv::cuda::GpuMat grayscale;
    if (rotated_image->channels() > 1) {
        cv::cuda::cvtColor(*rotated_image, grayscale, cv::COLOR_BGR2GRAY);
    } else {
        grayscale = *rotated_image;
    }

    // Adjust contrast using histogram equalization
    cv::cuda::GpuMat equalized;
    cv::cuda::equalizeHist(grayscale, equalized);
    std::cout << "Contrast adjusted for " << imgData.outputPath << std::endl;

    // Allocate histogram on device
    unsigned int* d_histogram;
    cudaMalloc(&d_histogram, HIST_BINS * sizeof(unsigned int));
    cudaMemset(d_histogram, 0, HIST_BINS * sizeof(unsigned int));

    // Kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((equalized.cols + blockSize.x - 1) / blockSize.x, 
                  (equalized.rows + blockSize.y - 1) / blockSize.y);

    // Compute histogram on GPU using the equalized image
    computeHistogramKernel<<<gridSize, blockSize>>>(
        equalized.ptr<unsigned char>(), equalized.cols, equalized.rows, d_histogram);
    cudaDeviceSynchronize();

    // Copy histogram to host
    unsigned int h_histogram[HIST_BINS];
    cudaMemcpy(h_histogram, d_histogram, HIST_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Compute total pixels
    int total_pixels = equalized.cols * equalized.rows;

    // Calculate Otsu threshold on host
    double otsu_threshold = computeOtsuThreshold(h_histogram, total_pixels);
    std::cout << "Computed Otsu threshold: " << otsu_threshold << std::endl;

    // Apply threshold on GPU using the equalized image
    cv::cuda::threshold(equalized, *binary_image, otsu_threshold, 255, cv::THRESH_BINARY);

    // Free device memory
    cudaFree(d_histogram);

    imgData.binary_ref = binary_image;
    //imgData.binary_ref = imgData.rotation_ref;

    if (imgData.binary_ref == nullptr) {
        std::cerr << "Error: binary_ref is null after assignment for " << imgData.outputPath << std::endl;
    } else {
        std::cout << "binary_ref successfully set for " << imgData.outputPath << std::endl;
    }
}
