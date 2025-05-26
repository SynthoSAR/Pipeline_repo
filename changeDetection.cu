#include "changeDetection.cuh"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>  // For timing
#include <numeric> // For statistics calculation
#include <vector>  // For storing timing data

// Static variable to store timing data across calls
static std::vector<double> execution_times;

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

        if (imgDataPrev.binary_ref->size() != imgDataCurr.binary_ref->size() || 
            imgDataCurr.binary_ref->size() != imgDataNext.binary_ref->size()) {
            std::cerr << "Error: Image dimensions mismatch!" << std::endl;
            return;
        }

        // --- GPU Data Prep ---
        cv::cuda::GpuMat prev_gpu, curr_gpu, next_gpu;
        cv::cuda::threshold(*imgDataPrev.binary_ref, prev_gpu, 128, 255, cv::THRESH_BINARY);
        cv::cuda::threshold(*imgDataCurr.binary_ref, curr_gpu, 128, 255, cv::THRESH_BINARY);
        cv::cuda::threshold(*imgDataNext.binary_ref, next_gpu, 128, 255, cv::THRESH_BINARY);

        // Convert to grayscale (if not already)
        cv::cuda::GpuMat prev_gray, curr_gray, next_gray;
        if (prev_gpu.channels() > 1) {
            cv::cuda::cvtColor(prev_gpu, prev_gray, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(curr_gpu, curr_gray, cv::COLOR_BGR2GRAY);
            cv::cuda::cvtColor(next_gpu, next_gray, cv::COLOR_BGR2GRAY);
        } else {
            prev_gray = prev_gpu;
            curr_gray = curr_gpu;
            next_gray = next_gpu;
        }

        // --- Feature Detection ---
        cv::Ptr<cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(
            prev_gray.type(), 1000, 0.01, 10, 3, true, 0.04);
        
        cv::cuda::GpuMat prev_corners;
        detector->detect(prev_gray, prev_corners);

        if (prev_corners.empty()) {
            changeMask.setTo(0);
            return;
        }

        // --- Sparse Optical Flow ---
        cv::cuda::GpuMat curr_corners, next_corners, status;
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk = cv::cuda::SparsePyrLKOpticalFlow::create();
        lk->calc(prev_gray, curr_gray, prev_corners, curr_corners, status);
        lk->calc(curr_gray, next_gray, curr_corners, next_corners, status);

        // --- Download Results for CPU Processing ---
        cv::Mat h_prev_corners(prev_corners);
        cv::Mat h_curr_corners(curr_corners);
        cv::Mat h_next_corners(next_corners);
        cv::Mat h_status(status);

        // --- Motion Analysis ---
        cv::Mat change_mask_cpu(curr_gray.size(), CV_8UC1, cv::Scalar(0));
        for (int i = 0; i < h_status.cols; i++) {
            if (h_status.at<uchar>(i)) {
                cv::Point2f prev_pt = h_prev_corners.at<cv::Point2f>(i);
                cv::Point2f curr_pt = h_curr_corners.at<cv::Point2f>(i);
                cv::Point2f next_pt = h_next_corners.at<cv::Point2f>(i);

                // Compute motion vectors
                cv::Point2f flow1 = curr_pt - prev_pt;
                cv::Point2f flow2 = next_pt - curr_pt;

                // Check motion consistency
                if (cv::norm(flow1 - flow2) > 5.0) {
                    cv::circle(change_mask_cpu, curr_pt, 5, cv::Scalar(255), -1);
                }
            }
        }

        // --- ROI Masking ---
        int margin_x = static_cast<int>(curr_gray.cols * 0.3);
        int margin_y = static_cast<int>(curr_gray.rows * 0.1);
        cv::Rect roi(margin_x, margin_y, curr_gray.cols - 2 * margin_x, curr_gray.rows - 2 * margin_y);
        cv::Mat roi_mask = cv::Mat::zeros(curr_gray.size(), CV_8UC1);
        roi_mask(roi).setTo(255);
        cv::bitwise_and(change_mask_cpu, roi_mask, change_mask_cpu);

        // --- Post-Processing ---
        cv::morphologyEx(change_mask_cpu, change_mask_cpu, cv::MORPH_OPEN, 
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
                        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(change_mask_cpu, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // --- Visualization ---
        cv::Mat result_visual;
        if (!imgDataCurr.original_frame.empty()) {
            result_visual = imgDataCurr.original_frame.clone();
        } else {
            cv::Mat temp;
            curr_gray.download(temp);
            cv::cvtColor(temp, result_visual, cv::COLOR_GRAY2BGR);
        }

        // Draw the contours and changes
        int change_count = 0;
        for (const auto& contour : contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            if (roi.contains(center)) {
                cv::circle(result_visual, center, static_cast<int>(radius * 1.5), cv::Scalar(0, 0, 255), 2);
                
                // Add contour area as text
                double area = cv::contourArea(contour);
                std::string area_text = std::to_string(static_cast<int>(area));
                cv::putText(result_visual, area_text, 
                           cv::Point(static_cast<int>(center.x - 10), static_cast<int>(center.y + 15)),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                           
                change_count++;
            }
        }

        // Create binary mask for output
        cv::Mat binary_mask = cv::Mat::zeros(result_visual.rows, result_visual.cols, CV_8UC1);
        for (const auto& contour : contours) {
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);
            if (roi.contains(center)) {
                cv::circle(binary_mask, center, static_cast<int>(radius * 1.5), cv::Scalar(255), -1);
            }
        }

        // Upload results to GPU
        changeMask.upload(binary_mask);
        outputAnnotated.upload(result_visual);

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
        double min_time = execution_time;
        double max_time = execution_time;
        
        if (!execution_times.empty()) {
            avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / execution_times.size();
            auto [min_it, max_it] = std::minmax_element(execution_times.begin(), execution_times.end());
            min_time = *min_it;
            max_time = *max_it;
        }

        // Add timing information to the output
        cv::putText(result_visual, "Time: " + std::to_string(execution_time) + " ms", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                   
        cv::putText(result_visual, "Avg: " + std::to_string(static_cast<int>(avg_time)) + " ms", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Upload the annotated image again with the timing information
        outputAnnotated.upload(result_visual);

        // Log the timing information
        std::cout << "Change detection completed for " << imgDataCurr.outputPath 
                  << " - Found " << change_count << " changes"
                  << " - Execution time: " << execution_time << " ms" << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Error in change detection - Execution time: " << execution_time << " ms" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Error in change detection - Execution time: " << execution_time << " ms" << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown exception in change detection" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "Error in change detection - Execution time: " << execution_time << " ms" << std::endl;
    }
}