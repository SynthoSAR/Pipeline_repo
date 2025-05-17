#include "commonAreaExtraction.cuh"
#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

void applyCommonAreaExtraction(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext,
                               cv::cuda::GpuMat& commonPrev, cv::cuda::GpuMat& commonCurr, cv::cuda::GpuMat& commonNext) {
    if (imgDataPrev.rotation_ref == nullptr || imgDataCurr.rotation_ref == nullptr || imgDataNext.rotation_ref == nullptr) {
        std::cerr << "Error: One or more rotation references are null!" << std::endl;
        return;
    }

    // Ensure all images have the same dimensions initially
    if (imgDataPrev.rotation_ref->cols != imgDataCurr.rotation_ref->cols || 
        imgDataPrev.rotation_ref->rows != imgDataCurr.rotation_ref->rows ||
        imgDataCurr.rotation_ref->cols != imgDataNext.rotation_ref->cols || 
        imgDataCurr.rotation_ref->rows != imgDataNext.rotation_ref->rows) {
        std::cerr << "Error: Rotated images must have the same dimensions!" << std::endl;
        return;
    }

    int width = imgDataCurr.rotation_ref->cols;
    int height = imgDataCurr.rotation_ref->rows;

    // Threshold to create binary masks of non-zero regions (foreground)
    cv::cuda::GpuMat maskPrev, maskCurr, maskNext;
    cv::cuda::threshold(*imgDataPrev.rotation_ref, maskPrev, 1, 255, cv::THRESH_BINARY);
    cv::cuda::threshold(*imgDataCurr.rotation_ref, maskCurr, 1, 255, cv::THRESH_BINARY);
    cv::cuda::threshold(*imgDataNext.rotation_ref, maskNext, 1, 255, cv::THRESH_BINARY);

    // Compute intersection of all three masks
    cv::cuda::GpuMat tempIntersection, intersectionMask;
    cv::cuda::bitwise_and(maskPrev, maskCurr, tempIntersection);
    cv::cuda::bitwise_and(tempIntersection, maskNext, intersectionMask);

    // Find bounding rectangle of the common area
    cv::Mat intersection_cpu;
    intersectionMask.download(intersection_cpu);
    cv::Rect commonRect = cv::boundingRect(intersection_cpu);

    // Validate common area
    if (commonRect.width <= 0 || commonRect.height <= 0 || 
        commonRect.x < 0 || commonRect.y < 0 || 
        commonRect.x + commonRect.width > width || commonRect.y + commonRect.height > height) {
        std::cerr << "Error: Invalid common area computed! Using full images as fallback." << std::endl;
        commonPrev = *imgDataPrev.rotation_ref;
        commonCurr = *imgDataCurr.rotation_ref;
        commonNext = *imgDataNext.rotation_ref;
        return;
    }

    // Crop to common area only (outputs have common area dimensions)
    commonPrev = (*imgDataPrev.rotation_ref)(commonRect).clone();
    commonCurr = (*imgDataCurr.rotation_ref)(commonRect).clone();
    commonNext = (*imgDataNext.rotation_ref)(commonRect).clone();

    std::cout << "Common area cropped (x: " << commonRect.x << ", y: " << commonRect.y 
              << ", w: " << commonRect.width << ", h: " << commonRect.height 
              << ") for " << imgDataCurr.outputPath << std::endl;
}