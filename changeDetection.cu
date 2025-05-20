#include "changeDetection.cuh"
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp> // For CPU operations
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, cv::cuda::GpuMat& changeMask, cv::cuda::GpuMat& outputAnnotated) {
    if (imgDataPrev.binary_ref == nullptr || imgDataCurr.binary_ref == nullptr || imgDataNext.binary_ref == nullptr) {
        std::cerr << "Error: One or more binary references are null!" << std::endl;
        return;
    }

    if (imgDataPrev.binary_ref->cols != imgDataCurr.binary_ref->cols || imgDataPrev.binary_ref->rows != imgDataCurr.binary_ref->rows ||
        imgDataCurr.binary_ref->cols != imgDataNext.binary_ref->cols || imgDataCurr.binary_ref->rows != imgDataNext.binary_ref->rows) {
        std::cerr << "Error: Images must have the same dimensions for change detection!" << std::endl;
        return;
    }

    // Make copies of the GPU data to work with
    cv::cuda::GpuMat prev_gpu = *imgDataPrev.binary_ref;
    cv::cuda::GpuMat curr_gpu = *imgDataCurr.binary_ref;
    cv::cuda::GpuMat next_gpu = *imgDataNext.binary_ref;

    // Ensure images are binary (0 and 1)
    cv::cuda::GpuMat prev_binary, curr_binary, next_binary;
    cv::cuda::threshold(prev_gpu, prev_binary, 128, 1, cv::THRESH_BINARY);
    cv::cuda::threshold(curr_gpu, curr_binary, 128, 1, cv::THRESH_BINARY);
    cv::cuda::threshold(next_gpu, next_binary, 128, 1, cv::THRESH_BINARY);

    // Step 1: Compute differences between frames
    cv::cuda::GpuMat diff1, diff2, diffMask;
    cv::cuda::absdiff(curr_binary, prev_binary, diff1);
    cv::cuda::absdiff(next_binary, curr_binary, diff2);
    cv::cuda::bitwise_and(diff1, diff2, diffMask);

    // Step 2: Apply Gaussian blur to reduce noise
    cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(
        diffMask.type(), diffMask.type(), cv::Size(5, 5), 1.5);
    cv::cuda::GpuMat blurredDiff;
    gaussianFilter->apply(diffMask, blurredDiff);

    // Step 3: Thresholding to highlight significant changes
    cv::cuda::GpuMat thresholdedDiff;
    cv::cuda::threshold(blurredDiff, thresholdedDiff, 0.3, 255, cv::THRESH_BINARY);

    // Step 4: Define ROI (exclude 30% from left and right, 10% from top and bottom)
    int margin_x = static_cast<int>(curr_gpu.cols * 0.3); // 30% of width
    int margin_y = static_cast<int>(curr_gpu.rows * 0.1); // 10% of height
    cv::Rect roi(margin_x, margin_y, curr_gpu.cols - 2 * margin_x, curr_gpu.rows - 2 * margin_y);

    // Create a mask for the central ROI
    cv::cuda::GpuMat mask(thresholdedDiff.size(), CV_8UC1);
    mask.setTo(cv::Scalar(0));
    cv::cuda::GpuMat mask_roi = mask(roi);
    mask_roi.setTo(cv::Scalar(255));

    // Apply the mask to keep only the central area
    cv::cuda::GpuMat filtered_roi;
    cv::cuda::bitwise_and(thresholdedDiff, mask, filtered_roi);

    // Download to CPU for morphological operations and contour detection
    cv::Mat filtered_roi_cpu;
    filtered_roi.download(filtered_roi_cpu);

    // Step 5: Morphological operations to clean up the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(filtered_roi_cpu, filtered_roi_cpu, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(filtered_roi_cpu, filtered_roi_cpu, cv::MORPH_CLOSE, kernel);

    // Step 6: Connected component analysis to filter by size
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(filtered_roi_cpu, labels, stats, centroids, 8);

    int min_size = 10; // Minimum size threshold
    int max_size = 100; // Maximum size threshold
    cv::Mat size_filtered_mask = filtered_roi_cpu.clone();

    for (int i = 1; i < num_labels; i++) { // Skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_size || area > max_size) {
            size_filtered_mask.setTo(0, labels == i);
        }
    }

    // Step 7: Find contours to highlight changes
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(size_filtered_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Step 8: Prepare visualization images
    // Download current frame for visualization
    cv::Mat curr_cpu;
    curr_gpu.download(curr_cpu);

    // Create color image for visualization
    cv::Mat result_visual;
    
    // Check if original frame exists in imgDataCurr
    bool has_original_frame = false;
    try {
        has_original_frame = !imgDataCurr.original_frame.empty();
    } catch(...) {
        // Field doesn't exist or is inaccessible
        has_original_frame = false;
    }
    
    if (!has_original_frame) {
        cv::cvtColor(curr_cpu * 255, result_visual, cv::COLOR_GRAY2BGR);
    } else {
        result_visual = imgDataCurr.original_frame.clone();
    }

    // Draw circles around changes
    for (const auto& contour : contours) {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);

        // Only draw circles if the center of the change is inside the ROI
        if (roi.contains(center)) {
            cv::circle(result_visual, center, static_cast<int>(radius * 1.5), cv::Scalar(0, 0, 255), 2); // Red circle
        }
    }

    // Create binary mask for output
    cv::Mat binary_mask = cv::Mat::zeros(curr_cpu.size(), CV_8UC1);
    for (const auto& contour : contours) {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);
        if (roi.contains(center)) {
            cv::circle(binary_mask, center, static_cast<int>(radius * 1.5), cv::Scalar(255), -1); // Filled white circle
        }
    }

    // Upload results to GPU
    changeMask.upload(binary_mask);
    outputAnnotated.upload(result_visual);

    // Display results if needed (add appropriate flag for headless operation)
    if (!result_visual.empty() && result_visual.cols > 0 && result_visual.rows > 0) {
        cv::imshow("Change Detection", result_visual);
        cv::waitKey(1); // Update display with small delay
    }

    std::cout << "Change detection completed for " << imgDataCurr.outputPath << std::endl;
}