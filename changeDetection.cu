#include "changeDetection.cuh"
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp> // For CPU operations

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, cv::cuda::GpuMat& changeMask) {
    if (imgDataPrev.binary_ref == nullptr || imgDataCurr.binary_ref == nullptr || imgDataNext.binary_ref == nullptr) {
        std::cerr << "Error: One or more binary references are null!" << std::endl;
        return;
    }

    if (imgDataPrev.binary_ref->cols != imgDataCurr.binary_ref->cols || imgDataPrev.binary_ref->rows != imgDataCurr.binary_ref->rows ||
        imgDataCurr.binary_ref->cols != imgDataNext.binary_ref->cols || imgDataCurr.binary_ref->rows != imgDataNext.binary_ref->rows) {
        std::cerr << "Error: Images must have the same dimensions for change detection!" << std::endl;
        return;
    }

    // Download to CPU
    cv::Mat prev_cpu, curr_cpu, next_cpu;
    imgDataPrev.binary_ref->download(prev_cpu);
    imgDataCurr.binary_ref->download(curr_cpu);
    imgDataNext.binary_ref->download(next_cpu);

    // Step 1: Gaussian blur (CPU)
    cv::Mat blurPrev, blurCurr, blurNext;
    cv::GaussianBlur(prev_cpu, blurPrev, cv::Size(5, 5), 0);
    cv::GaussianBlur(curr_cpu, blurCurr, cv::Size(5, 5), 0);
    cv::GaussianBlur(next_cpu, blurNext, cv::Size(5, 5), 0);
    std::cout << "Gaussian blur applied (CPU) for " << imgDataCurr.outputPath << std::endl;

    // Step 2: Three-frame difference (CPU)
    cv::Mat diff1, diff2, diffMask;
    cv::absdiff(blurCurr, blurPrev, diff1);
    cv::absdiff(blurNext, blurCurr, diff2);
    cv::bitwise_and(diff1, diff2, diffMask);

    // Step 3: Thresholding (CPU)
    cv::threshold(diffMask, diffMask, 200, 255, cv::THRESH_BINARY);
    std::cout << "Thresholding applied (CPU) for " << imgDataCurr.outputPath << std::endl;

    // Step 4: Morphological closing (CPU)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(diffMask, diffMask, cv::MORPH_CLOSE, kernel);
    std::cout << "Morphological closing applied (CPU) for " << imgDataCurr.outputPath << std::endl;

    // Step 5: Overlay (CPU)
    cv::Mat overlay_cpu = cv::Mat::zeros(curr_cpu.size(), CV_8UC3);
    cv::cvtColor(curr_cpu, overlay_cpu, cv::COLOR_GRAY2BGR);
    overlay_cpu.setTo(cv::Scalar(0, 0, 255), diffMask); // Red for changes

    // Step 6: High-density removal (CPU)
    cv::Mat red_channel = overlay_cpu.clone();
    cv::extractChannel(red_channel, red_channel, 2);
    cv::Mat red_mask;
    cv::threshold(red_channel, red_mask, 100, 255, cv::THRESH_BINARY);
    cv::bitwise_and(red_mask, diffMask, red_mask);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(red_mask, labels, stats, centroids, 8);
    cv::Mat high_density_mask = cv::Mat::zeros(red_mask.size(), CV_8UC1);
    int density_threshold = 100;
    for (int i = 1; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) > density_threshold) {
            high_density_mask.setTo(255, labels == i);
        }
    }

    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(high_density_mask, high_density_mask, dilate_kernel);
    overlay_cpu.setTo(cv::Scalar(0, 0, 0), high_density_mask);

    // Upload to GPU
    changeMask.upload(overlay_cpu);
    std::cout << "Change detection completed with overlay for " << imgDataCurr.outputPath << std::endl;
}