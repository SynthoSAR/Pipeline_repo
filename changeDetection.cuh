#ifndef CHANGE_DETECTION_CUH
#define CHANGE_DETECTION_CUH

#include "imageLoad.cuh"
#include <opencv2/cudaarithm.hpp> // For cv::cuda::absdiff, bitwise_and, threshold

void applyChangeDetection(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext, cv::cuda::GpuMat& changeMask);

#endif // CHANGE_DETECTION_CUH
