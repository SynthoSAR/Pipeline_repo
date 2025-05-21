#include "changeDetection.cuh"
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, 
                         cv::cuda::GpuMat& changeMask, cv::cuda::GpuMat& outputAnnotated) {
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

    // --- Feature Detection (Updated API) ---
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

    // --- Visualization ---
    cv::Mat result_visual;
    if (!imgDataCurr.original_frame.empty()) {
        result_visual = imgDataCurr.original_frame.clone();
    } else {
        cv::cvtColor(curr_gray, result_visual, cv::COLOR_GRAY2BGR);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(change_mask_cpu, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);
        if (roi.contains(center)) {
            cv::circle(result_visual, center, static_cast<int>(radius * 1.5), cv::Scalar(0, 0, 255), 2);
        }
    }

    // --- Upload Results ---
    changeMask.upload(change_mask_cpu);
    outputAnnotated.upload(result_visual);

    std::cout << "Optical flow change detection completed." << std::endl;
}