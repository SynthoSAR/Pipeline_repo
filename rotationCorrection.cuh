#ifndef ROTATION_CORRECTION_CUH
#define ROTATION_CORRECTION_CUH

#include "imageLoad.cuh"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

void applyRotationCorrection(ImageData& imgData1, ImageData& imgData2);

#endif // ROTATION_CORRECTION_CUH
