#ifndef BINARIZATION_CUH
#define BINARIZATION_CUH

#include "imageLoad.cuh"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgproc.hpp>     

void applyBinarization(ImageData& imgData);

#endif // BINARIZATION_CUH
