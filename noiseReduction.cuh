#ifndef NOISE_REDUCTION_CUH
#define NOISE_REDUCTION_CUH

#include "imageLoad.cuh"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/opencv.hpp>


void applyNoiseReduction(ImageData& imgData);

#endif // NOISE_REDUCTION_CUH

