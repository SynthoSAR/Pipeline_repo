#include "changeDetection.cuh"
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, cv::cuda::GpuMat& changeMask, cv::cuda::GpuMat& outputAnnotated) {
    try {
        // Check if CUDA is available
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cerr << "Error: No CUDA-capable device found." << std::endl;
            return;
        }

        // Input validation
        if (imgDataPrev.binary_ref == nullptr || imgDataCurr.binary_ref == nullptr || imgDataNext.binary_ref == nullptr) {
            std::cerr << "Error: One or more binary references are null!" << std::endl;
            return;
        }

        if (imgDataPrev.binary_ref->cols != imgDataCurr.binary_ref->cols || imgDataPrev.binary_ref->rows != imgDataCurr.binary_ref->rows ||
            imgDataCurr.binary_ref->cols != imgDataNext.binary_ref->cols || imgDataCurr.binary_ref->rows != imgDataNext.binary_ref->rows) {
            std::cerr << "Error: Images must have the same dimensions for change detection!" << std::endl;
            return;
        }

        // Get GPU images
        cv::cuda::GpuMat prev_gpu = *imgDataPrev.binary_ref;
        cv::cuda::GpuMat curr_gpu = *imgDataCurr.binary_ref;
        cv::cuda::GpuMat next_gpu = *imgDataNext.binary_ref;

        // Ensure images are binary (convert to 0 and 1)
        cv::cuda::GpuMat prev_binary, curr_binary, next_binary;
        cv::cuda::threshold(prev_gpu, prev_binary, 128, 1, cv::THRESH_BINARY);
        cv::cuda::threshold(curr_gpu, curr_binary, 128, 1, cv::THRESH_BINARY);
        cv::cuda::threshold(next_gpu, next_binary, 128, 1, cv::THRESH_BINARY);

        // Step 1: Compute Difference Map (XOR operation between consecutive frames)
        cv::cuda::GpuMat diff1, diff2, combined_diff;
        cv::cuda::bitwise_xor(curr_binary, prev_binary, diff1);  // Current vs Previous
        cv::cuda::bitwise_xor(next_binary, curr_binary, diff2);   // Next vs Current
        
        // Combine both differences to find consistent changes
        cv::cuda::bitwise_and(diff1, diff2, combined_diff);

        // Step 2: Morphological Filtering to Remove Noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // 3x3 kernel
        cv::Ptr<cv::cuda::Filter> morph_filter = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, combined_diff.type(), kernel);
        cv::cuda::GpuMat filtered;
        morph_filter->apply(combined_diff, filtered);

        // Step 3: Exclude 30% from left and right, 10% from top and bottom
        int margin_x = static_cast<int>(curr_gpu.cols * 0.3); // 30% of width
        int margin_y = static_cast<int>(curr_gpu.rows * 0.1); // 10% of height
        cv::Rect roi(margin_x, margin_y, curr_gpu.cols - 2 * margin_x, curr_gpu.rows - 2 * margin_y);

        // Create a mask on GPU for the central ROI
        cv::cuda::GpuMat mask(filtered.size(), CV_8UC1, cv::Scalar(0)); // Initialize to zeros
        cv::cuda::GpuMat mask_roi = mask(roi);
        mask_roi.setTo(cv::Scalar(255)); // Set central region to white (255)

        // Apply the mask to keep only the central area
        cv::cuda::GpuMat filtered_roi;
        cv::cuda::bitwise_and(filtered, mask, filtered_roi);

        // Step 4: Connected Component Analysis (CPU fallback due to limited CUDA support)
        cv::Mat filtered_roi_host;
        filtered_roi.download(filtered_roi_host); // Transfer to CPU

        cv::Mat labels, stats, centroids;
        int num_labels = cv::connectedComponentsWithStats(filtered_roi_host, labels, stats, centroids);

        int min_size = 10; // Minimum size threshold
        int max_size = 100; // Maximum size threshold (increased from 20)

        for (int i = 1; i < num_labels; i++) { // Skip background (label 0)
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < min_size || area > max_size) {
                filtered_roi_host.setTo(0, labels == i);
            }
        }

        // Upload filtered result back to GPU
        filtered_roi.upload(filtered_roi_host);

        // Step 5: Find Contours to Highlight Changes (CPU-based due to OpenCV CUDA limitations)
        cv::Mat filtered_roi_contours;
        filtered_roi.download(filtered_roi_contours);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(filtered_roi_contours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Step 6: Prepare visualization
        cv::Mat result_visual;
        
        // Check if original frame exists in imgDataCurr
        bool has_original_frame = false;
        try {
            has_original_frame = !imgDataCurr.original_frame.empty();
        } catch(...) {
            // Field doesn't exist or is inaccessible
            has_original_frame = false;
        }
        
        if (has_original_frame) {
            result_visual = imgDataCurr.original_frame.clone();
        } else {
            // Convert current binary image to color for visualization
            cv::Mat curr_cpu;
            curr_gpu.download(curr_cpu);
            cv::Mat temp = curr_cpu * 255; // Convert to 0-255 range
            cv::cvtColor(temp, result_visual, cv::COLOR_GRAY2BGR);
        }

        // Draw circles around changes (only in the ROI)
        int change_count = 0;
        for (const auto& contour : contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);

            // Only draw circles if the center of the change is inside the ROI
            if (roi.contains(center)) {
                cv::circle(result_visual, center, static_cast<int>(std::max(10.0f, radius * 1.5f)), cv::Scalar(0, 0, 255), 2); // Red circle
                
                // Add a small filled circle at the center
                cv::circle(result_visual, center, 3, cv::Scalar(0, 0, 255), -1);
                
                // Add area information
                double area = cv::contourArea(contour);
                std::string area_text = std::to_string(static_cast<int>(area));
                cv::putText(result_visual, area_text, 
                           cv::Point(static_cast<int>(center.x - 10), static_cast<int>(center.y + 15)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                
                change_count++;
            }
        }

        // Draw ROI rectangle
        cv::rectangle(result_visual, roi, cv::Scalar(0, 255, 0), 1);
        
        // Add change count to the image
        cv::putText(result_visual, "Changes: " + std::to_string(change_count),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(255, 255, 255), 2);

        // Create binary mask for output
        cv::Mat binary_mask = cv::Mat::zeros(result_visual.rows, result_visual.cols, CV_8UC1);
        for (const auto& contour : contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            if (roi.contains(center)) {
                cv::circle(binary_mask, center, static_cast<int>(std::max(10.0f, radius * 1.5f)), cv::Scalar(255), -1); // Filled white circle
            }
        }

        // Upload results to GPU
        changeMask.upload(binary_mask);
        outputAnnotated.upload(result_visual);

        std::cout << "Change detection completed for " << imgDataCurr.outputPath 
                  << " - Found " << change_count << " changes" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in change detection: " << e.what() << std::endl;
        
        // Create fallback outputs to prevent crashes
        cv::Mat empty_mask = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::Mat error_img = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::putText(error_img, "Error in processing", cv::Point(10, 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                   
        changeMask.upload(empty_mask);
        outputAnnotated.upload(error_img);
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception in change detection: " << e.what() << std::endl;
        
        // Create fallback outputs
        cv::Mat empty_mask = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::Mat error_img = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::putText(error_img, "Error in processing", cv::Point(10, 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                   
        changeMask.upload(empty_mask);
        outputAnnotated.upload(error_img);
    }
    catch (...) {
        std::cerr << "Unknown error in change detection" << std::endl;
        
        // Create fallback outputs
        cv::Mat empty_mask = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::Mat error_img = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::putText(error_img, "Unknown error", cv::Point(10, 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
                   
        changeMask.upload(empty_mask);
        outputAnnotated.upload(error_img);
    }
}