#ifndef COMMON_AREA_EXTRACTION_CUH
#define COMMON_AREA_EXTRACTION_CUH

#include "imageLoad.cuh"
#include <opencv2/cudaarithm.hpp>

void applyCommonAreaExtraction(ImageData& imgDataPrev, ImageData& imgDataCurr, ImageData& imgDataNext,
                               cv::cuda::GpuMat& commonPrev, cv::cuda::GpuMat& commonCurr, cv::cuda::GpuMat& commonNext);

#endif // COMMON_AREA_EXTRACTION_CUH