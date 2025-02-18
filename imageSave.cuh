#ifndef IMAGE_SAVE_CUH
#define IMAGE_SAVE_CUH

#include <string>
#include "imageLoad.cuh"

void saveImageFromGPU(const ImageData& imgData);
void freeGPUData(ImageData& imgData);

#endif // IMAGE_SAVE_CUH
