#include "rotationCorrection.cuh"

using namespace std;
using namespace cv;

void applyRotationCorrection(ImageData& imgData1, ImageData& imgData2) {
    if (imgData1.denoised_ref == nullptr || imgData2.denoised_ref == nullptr) {
        std::cerr << "Error: One or both denoised image references are null!" << std::endl;
        return;
    }

    cv::cuda::GpuMat gpu_img1 = *imgData1.denoised_ref;
    cv::cuda::GpuMat gpu_img2 = *imgData2.denoised_ref;

    // Create CUDA-based ORB detector
    Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create();

    // Detect keypoints and compute descriptors
    cv::cuda::GpuMat gpu_kp1, gpu_des1, gpu_kp2, gpu_des2;
    vector<KeyPoint> kp1, kp2;

    orb->detectAndComputeAsync(gpu_img1, noArray(), gpu_kp1, gpu_des1);
    orb->detectAndComputeAsync(gpu_img2, noArray(), gpu_kp2, gpu_des2);

    // Download keypoints to CPU
    orb->convert(gpu_kp1, kp1);
    orb->convert(gpu_kp2, kp2);

    // CUDA-based Brute-Force Matcher
    Ptr<cv::cuda::DescriptorMatcher> bf = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
    vector<DMatch> matches;
    bf->match(gpu_des1, gpu_des2, matches);

    // Sort and retain best 30 matches
    sort(matches.begin(), matches.end(), [](const DMatch &a, const DMatch &b) {
        return a.distance < b.distance;
    });
    
    // Limit to reasonable number of matches
    if (matches.size() > 30) {
        matches.resize(30);
    }

    // Extract matched keypoints
    vector<Point2f> match_points1, match_points2;
    
    if (kp1.empty() || kp2.empty()) {
        std::cerr << "Error: Keypoints not detected or conversion failed!" << std::endl;
        imgData1.rotation_ref = new cv::cuda::GpuMat();
        imgData2.rotation_ref = new cv::cuda::GpuMat();
     
        gpu_img1.copyTo(*imgData1.rotation_ref);
        gpu_img2.copyTo(*imgData2.rotation_ref);

        return;
    }

    for (const auto &m : matches) {
        match_points1.push_back(kp1[m.queryIdx].pt);
        match_points2.push_back(kp2[m.trainIdx].pt);
    }

    // Check if we have enough matches
    if (match_points1.size() < 3) {
        std::cerr << "Error: Not enough matching points for transformation!" << std::endl;
        imgData1.rotation_ref = new cv::cuda::GpuMat();
        imgData2.rotation_ref = new cv::cuda::GpuMat();
     
        gpu_img1.copyTo(*imgData1.rotation_ref);
        gpu_img2.copyTo(*imgData2.rotation_ref);

        return;
    }

    // Using partial affine transformation (rotation + translation + uniform scale)
    Mat inliers; // Define inliers variable
    Mat affine_matrix = estimateAffinePartial2D(match_points2, match_points1, inliers, RANSAC);

    cv::cuda::GpuMat gpu_aligned_img2;
    
    if (!affine_matrix.empty()) {
        // Apply transformation using the affine matrix
        cv::cuda::warpAffine(gpu_img2, gpu_aligned_img2, affine_matrix, gpu_img1.size());
        
        // Report transformation parameters
        double scale = std::sqrt(affine_matrix.at<double>(0,0)*affine_matrix.at<double>(0,0) + 
                               affine_matrix.at<double>(0,1)*affine_matrix.at<double>(0,1));
        double angle = atan2(affine_matrix.at<double>(0,1), affine_matrix.at<double>(0,0)) * 180.0 / CV_PI;
        Point2f translation(affine_matrix.at<double>(0,2), affine_matrix.at<double>(1,2));
        
        // std::cout << "Transformation applied - Scale: " << scale << ", Angle: " << angle 
                //   << "°, Translation: (" << translation.x << "," << translation.y << ")" << std::endl;
        
        // Count inliers
        int inlier_count = cv::countNonZero(inliers);
        std::cout << "Inlier count: " << inlier_count << " out of " << match_points1.size() << std::endl;
    } else {
        std::cerr << "Warning: Could not compute affine transformation. Falling back to manual rotation." << std::endl;
        
        // Fallback to manual rotation calculation
        double angle_sum = 0.0;
        int valid_pairs = 0;
        for (size_t i = 0; i < match_points1.size(); ++i) {
            // For each pair, compute the angle between vectors to the image center
            Point2f p1 = match_points1[i];
            Point2f p2 = match_points2[i];
            Point2f center(gpu_img1.cols / 2.0f, gpu_img1.rows / 2.0f); // Image center

            // Vectors from center to keypoints
            Point2f v1 = p1 - center;
            Point2f v2 = p2 - center;

            // Compute angle using dot product and cross product
            double dot = v1.x * v2.x + v1.y * v2.y;
            double det = v1.x * v2.y - v1.y * v2.x;
            double angle = atan2(det, dot) * 180.0 / CV_PI; // Convert to degrees

            if (std::abs(angle) < 45.0) { // Filter outliers (arbitrary threshold)
                angle_sum += angle;
                valid_pairs++;
            }
        }

        if (valid_pairs == 0) {
            std::cerr << "Error: No valid rotation angle computed!" << std::endl;
            imgData1.rotation_ref = new cv::cuda::GpuMat();
            imgData2.rotation_ref = new cv::cuda::GpuMat();
            gpu_img1.copyTo(*imgData1.rotation_ref);
            gpu_img2.copyTo(*imgData2.rotation_ref);
            return;
        }

        double rotation_angle = angle_sum / valid_pairs;
        std::cout << "Computed rotation angle (fallback): " << rotation_angle << " degrees" << std::endl;

        // Create rotation matrix (2x3 affine matrix)
        Point2f center(gpu_img1.cols / 2.0f, gpu_img1.rows / 2.0f);
        Mat rotation_matrix = getRotationMatrix2D(center, rotation_angle, 1.0); // Scale = 1.0 (no scaling)

        // Apply rotation using CUDA
        cv::cuda::warpAffine(gpu_img2, gpu_aligned_img2, rotation_matrix, gpu_img1.size());
    }
    
    // Allocate and store results
    imgData1.rotation_ref = new cv::cuda::GpuMat();
    imgData2.rotation_ref = new cv::cuda::GpuMat();
     
    gpu_img1.copyTo(*imgData1.rotation_ref);
    gpu_aligned_img2.copyTo(*imgData2.rotation_ref);
}
