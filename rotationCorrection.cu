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
    matches.resize(30);

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

    // Find Homography (CPU-based)
    Mat H = findHomography(match_points2, match_points1, RANSAC);

    // Apply Homography using CUDA
    cv::cuda::GpuMat gpu_aligned_img2;
    cv::cuda::warpPerspective(gpu_img2, gpu_aligned_img2, H, gpu_img1.size());

    // Compute Intersection Mask
    cv::cuda::GpuMat gpu_mask1, gpu_mask2, gpu_intersection_mask;
    cv::cuda::threshold(gpu_img1, gpu_mask1, 1, 255, THRESH_BINARY);
    cv::cuda::threshold(gpu_aligned_img2, gpu_mask2, 1, 255, THRESH_BINARY);

    // Bitwise AND on GPU
    cv::cuda::bitwise_and(gpu_mask1, gpu_mask2, gpu_intersection_mask);

    // Extract common regions on GPU
    cv::cuda::GpuMat gpu_common_img1, gpu_common_aligned_img2;
    cv::cuda::bitwise_and(gpu_img1, gpu_img1, gpu_common_img1, gpu_intersection_mask);
    cv::cuda::bitwise_and(gpu_aligned_img2, gpu_aligned_img2, gpu_common_aligned_img2,  gpu_intersection_mask);
    
    imgData1.rotation_ref = new cv::cuda::GpuMat();
    imgData2.rotation_ref = new cv::cuda::GpuMat();
     
    gpu_img1.copyTo(*imgData1.rotation_ref);
    gpu_aligned_img2.copyTo(*imgData2.rotation_ref);

    //gpu_common_img1.copyTo(*imgData1.rotation_ref);
    //gpu_common_aligned_img2.copyTo(*imgData2.rotation_ref);

    
    ///Mat common_img1, common_aligned_img2;
    //gpu_common_img1.download(common_img1);
    //gpu_common_aligned_img2.download(common_aligned_img2);


   // Mat combined_common_img;
    //hconcat(common_img1, common_aligned_img2, combined_common_img);

    //imshow("Common Region With Rotation correct", combined_common_img);
    //waitKey(0);

    //Mat common_img1, common_aligned_img2;
    //gpu_common_img1.download(common_img1);
    //gpu_common_aligned_img2.download(common_aligned_img2);
   
}
