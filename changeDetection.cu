#include "changeDetection.cuh"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp> // Add this include for CUDA filters
#include <chrono>  // For timing
#include <numeric> // For statistics calculation
#include <vector>  // For storing timing data
#include <cuda_runtime.h>
// Remove Eigen for now to avoid the other error
// #include <Eigen/Dense>  // For advanced PCA operations

// ADD THIS LINE - Function declaration for computeDifferenceImage
cv::Mat computeDifferenceImage(const cv::Mat& img1, const cv::Mat& img2);

// Static variable to store timing data across calls
static std::vector<double> execution_times;

// Block size for CUDA kernels
#define BLOCK_SIZE 16

// CUDA Kernel: Compute absolute difference
__global__ void absDiffKernel(const uchar* img1, const uchar* img2, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = abs(img1[idx] - img2[idx]);
    }
}

// CUDA Kernel: Threshold
__global__ void thresholdKernel(uchar* input, uchar* output, int width, int height, uchar threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

// CUDA Kernel: Apply ROI mask
__global__ void applyROIKernel(uchar* input, int width, int height, int leftMargin, 
                              int topMargin, int rightMargin, int bottomMargin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        if (x < leftMargin || x >= (width - rightMargin) || 
            y < topMargin || y >= (height - bottomMargin)) {
            input[idx] = 0;
        }
    }
}

// CUDA Kernel: Bitwise AND operation
__global__ void bitwiseAndKernel(const uchar* img1, const uchar* img2, uchar* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = img1[idx] & img2[idx];
    }
}

// Function to compute the PCA-based change detection map
cv::Mat computePCAChangeMap(const cv::Mat& diffImage) {
    // Convert to float for PCA
    cv::Mat diffFloat;
    diffImage.convertTo(diffFloat, CV_32F);
    
    // Reshape for PCA
    cv::Mat reshaped = diffFloat.reshape(1, diffFloat.rows * diffFloat.cols);
    
    // Apply PCA
    cv::PCA pca(reshaped, cv::Mat(), cv::PCA::DATA_AS_ROW, 5);  // Using 5 principal components
    cv::Mat pcaResult = pca.project(reshaped);
    
    // Reshape back to image format
    cv::Mat pcaImage = pcaResult.reshape(1, diffImage.rows);
    
    // Normalize for visualization
    cv::Mat normalized;
    cv::normalize(pcaImage, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);
    
    return normalized;
}

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, 
                         cv::cuda::GpuMat& changeMask, cv::cuda::GpuMat& outputAnnotated) {
    // Start timing the entire algorithm
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // --- Input Validation ---
        if (imgDataPrev.binary_ref == nullptr || imgDataCurr.binary_ref == nullptr || imgDataNext.binary_ref == nullptr) {
            std::cerr << "Error: Binary references are null!" << std::endl;
            return;
        }

        // Get dimensions
        int width = imgDataCurr.binary_ref->cols;
        int height = imgDataCurr.binary_ref->rows;
        size_t imageSize = width * height * sizeof(uchar);

        // --- GPU Data Prep ---
        // FIXED: Use proper pointers that are accessible from CUDA kernels
        cv::cuda::GpuMat prev_gray, curr_gray, next_gray;
        
        // Copy data to ensure contiguous memory layout 
        cv::cuda::GpuMat prev_copy, curr_copy, next_copy;
        imgDataPrev.binary_ref->copyTo(prev_copy);
        imgDataCurr.binary_ref->copyTo(curr_copy);
        imgDataNext.binary_ref->copyTo(next_copy);
        
        if (prev_copy.channels() > 1) {
            cv::cuda::cvtColor(prev_copy, prev_gray, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(curr_copy, curr_gray, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(next_copy, next_gray, cv::COLOR_BGR2GRAY);
        } else {
            prev_gray = prev_copy;
            curr_gray = curr_copy;
            next_gray = next_copy;
        }

        // === APPROACH 1: STANDARD FRAME DIFFERENCING ===
        cv::cuda::GpuMat diff1, diff2;
        cv::cuda::absdiff(curr_gray, prev_gray, diff1);
        cv::cuda::absdiff(next_gray, curr_gray, diff2);
        
        // Apply Gaussian blur to reduce noise - CORRECTED API usage
        // Create Gaussian filter and apply it
        cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(
            diff1.type(), diff1.type(), cv::Size(5, 5), 1.5);
        cv::cuda::GpuMat blurred_diff1, blurred_diff2;
        gaussianFilter->apply(diff1, blurred_diff1);
        gaussianFilter->apply(diff2, blurred_diff2);
        
        // === APPROACH 2: PCA-BASED CHANGE DETECTION ===
        // Download images for CPU-based PCA processing
        cv::Mat h_prev, h_curr, h_next;
        prev_gray.download(h_prev);
        curr_gray.download(h_curr);
        next_gray.download(h_next);
        
        // Compute difference images for PCA
        cv::Mat diff_prev_curr = computeDifferenceImage(h_prev, h_curr);
        cv::Mat diff_curr_next = computeDifferenceImage(h_curr, h_next);
        
        // Apply Gaussian blur to smooth difference images
        cv::Mat smoothed_diff1, smoothed_diff2;
        cv::GaussianBlur(diff_prev_curr, smoothed_diff1, cv::Size(5, 5), 0);
        cv::GaussianBlur(diff_curr_next, smoothed_diff2, cv::Size(5, 5), 0);
        
        // Combine differences for PCA analysis
        cv::Mat combined_diff;
        cv::bitwise_and(smoothed_diff1, smoothed_diff2, combined_diff);
        
        // Apply PCA to extract features from the combined difference
        cv::Mat pca_result = computePCAChangeMap(combined_diff);
        
        // Threshold PCA result to get binary change map
        cv::Mat pca_binary;
        cv::threshold(pca_result, pca_binary, 30, 255, cv::THRESH_BINARY);
        
        // Upload PCA result to GPU
        cv::cuda::GpuMat pca_change_map;
        pca_change_map.upload(pca_binary);
        
        // === COMBINE BOTH APPROACHES ===
        // Standard approach thresholding
        cv::cuda::GpuMat std_thresh1, std_thresh2, std_combined;
        cv::cuda::threshold(blurred_diff1, std_thresh1, 20, 255, cv::THRESH_BINARY);
        cv::cuda::threshold(blurred_diff2, std_thresh2, 20, 255, cv::THRESH_BINARY);
        cv::cuda::bitwise_and(std_thresh1, std_thresh2, std_combined);
        
        // Combine standard and PCA results
        cv::cuda::GpuMat final_change_map;
        cv::cuda::bitwise_or(std_combined, pca_change_map, final_change_map);
        
        // Define ROI
        int leftMargin = static_cast<int>(width * 0.3);
        int rightMargin = static_cast<int>(width * 0.3);
        int topMargin = static_cast<int>(height * 0.1);
        int bottomMargin = static_cast<int>(height * 0.1);
        cv::Rect roi(leftMargin, topMargin, width - leftMargin - rightMargin, height - topMargin - bottomMargin);
        
        // Create mask for ROI
        cv::cuda::GpuMat roi_mask(height, width, CV_8UC1, cv::Scalar(0));
        cv::cuda::GpuMat roi_region = roi_mask(roi);
        roi_region.setTo(cv::Scalar(255));
        
        // Apply ROI mask
        cv::cuda::GpuMat masked_diff;
        cv::cuda::bitwise_and(final_change_map, roi_mask, masked_diff);
        
        // Download for CPU processing
        cv::Mat threshold_cpu;
        masked_diff.download(threshold_cpu);
        
        
        // Morphological operations to clean up noise
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(threshold_cpu, threshold_cpu, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(threshold_cpu, threshold_cpu, cv::MORPH_CLOSE, kernel);
        
        // Debug: Save post-morphology result

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(threshold_cpu, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Debug: Print number of detected contours
        std::cout << "Detected " << contours.size() << " initial contours" << std::endl;
        
        // Filter contours by size - ADJUSTED threshold values for PCA approach
        std::vector<std::vector<cv::Point>> filtered_contours;
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            // Using a wider range for PCA-enhanced detection
            if (area >= 30 && area <= 50) { // Increased upper limit to catch PCA-detected changes
                filtered_contours.push_back(contour);
            }
        }
        
        std::cout << "After filtering: " << filtered_contours.size() << " contours" << std::endl;
        
        // --- Create Visualization ---
        cv::Mat result_visual;
        if (!imgDataCurr.original_frame.empty()) {
            result_visual = imgDataCurr.original_frame.clone();
        } else {
            cv::Mat temp;
            curr_gray.download(temp);
            cv::cvtColor(temp, result_visual, cv::COLOR_GRAY2BGR);
        }

        // Draw ROI rectangle for debugging
        cv::rectangle(result_visual, roi, cv::Scalar(0, 255, 0), 1);
        
        // Draw contours and changes
        int change_count = 0;
        for (const auto& contour : filtered_contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            
            if (roi.contains(center)) {
                // Check if this is primarily a PCA-detected change
                cv::Mat contour_mask = cv::Mat::zeros(threshold_cpu.size(), CV_8UC1);
                cv::drawContours(contour_mask, std::vector<std::vector<cv::Point>>{contour}, 0, 255, -1);
                
                // Check overlap with PCA change map
                cv::Mat pca_overlap;
                cv::bitwise_and(pca_binary, contour_mask, pca_overlap);
                double pca_area = cv::countNonZero(pca_overlap);
                double contour_area = cv::contourArea(contour);
                bool isPCAChange = (pca_area / contour_area) > 0.3;  // >30% overlap is considered PCA change
                
                // Fill the contour with red
                cv::drawContours(result_visual, std::vector<std::vector<cv::Point>>{contour}, 0, 
                                cv::Scalar(0, 0, 255), -1);
                
                // Add border in a different color for PCA changes
                if (isPCAChange) {
                    cv::drawContours(result_visual, std::vector<std::vector<cv::Point>>{contour}, 0, 
                                   cv::Scalar(0, 140, 255), 2);
                }
                
                // Add contour area as text
                double area = cv::contourArea(contour);
                std::string area_text = std::to_string(static_cast<int>(area));
                std::string method = isPCAChange ? " PCA" : "";
                cv::putText(result_visual, area_text + method, 
                           cv::Point(static_cast<int>(center.x - 15), static_cast<int>(center.y + 15)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                           
                change_count++;
            }
        }

        // Create binary mask for output
        cv::Mat binary_mask = cv::Mat::zeros(result_visual.rows, result_visual.cols, CV_8UC1);
        for (const auto& contour : filtered_contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            if (roi.contains(center)) {
                cv::circle(binary_mask, center, static_cast<int>(std::max(10.0f, radius * 1.5f)), 
                          cv::Scalar(255), -1);
            }
        }

        // Upload results to GPU
        changeMask.upload(binary_mask);
        outputAnnotated.upload(result_visual);
        
        // Debug: Save final output to check

        // End timing after all processing is complete
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        // Store the timing for statistics
        execution_times.push_back(execution_time);
        if (execution_times.size() > 100) { // Keep only the last 100 measurements
            execution_times.erase(execution_times.begin());
        }
        
        // Calculate statistics
        double avg_time = 0.0;
        if (!execution_times.empty()) {
            avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
        }

        // Add timing and change count information to the output
        cv::putText(result_visual, "Time: " + std::to_string(execution_time) + " ms", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                   
        cv::putText(result_visual, "Avg: " + std::to_string(static_cast<int>(avg_time)) + " ms", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                   
        cv::putText(result_visual, "PCA Changes: " + std::to_string(change_count),
                   cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Upload the annotated image again with the timing information
        outputAnnotated.upload(result_visual);

        // Log the timing information
        std::cout << "PCA-enhanced change detection completed for " << imgDataCurr.outputPath 
                  << " - Found " << change_count << " changes"
                  << " - Total execution time: " << execution_time << " ms" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception in change detection" << std::endl;
    }
}

// Helper function to compute the difference image (used in PCA approach)
cv::Mat computeDifferenceImage(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    return diff;
}