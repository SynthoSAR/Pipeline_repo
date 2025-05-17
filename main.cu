#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "imageLoad.cuh"
#include "imageSave.cuh"
#include "noiseReduction.cuh"
#include "rotationCorrection.cuh"
#include "commonAreaExtraction.cuh"
#include "binarization.cuh"
#include "changeDetection.cuh"

std::mutex mtx_loader;   // Mutex for loader thread
std::mutex mtx_noise;    // Mutex for noise reduction threads
std::mutex mtx_rotation;  // Mutex for rotation correction
std::mutex mtx_common;
std::mutex mtx_binarization; // Mutex for binarization
std::mutex mtx_change; // Mutex for change detection
std::mutex mtx_saver;    // Mutex for saver thread


std::condition_variable cv_loader;  // Condition variable for loader
std::condition_variable cv_noise;   // Condition variable for noise reduction threads
std::condition_variable cv_rotation;  // Condition variable for rotation correction
std::condition_variable cv_common;  // Condition variable for rotation correction
std::condition_variable cv_binarization; // Condition variable for binarization
std::condition_variable cv_change; // Condition variablr for change detection
std::condition_variable cv_saver;   // Condition variable for saver thread

bool isDataReady = false;         // Flag for when data is ready for noise reduction
bool isNoiseReductionDone = false; // Flag for when noise reduction is done
bool isRotationDone = false;  // Flag for rotation correction
bool isCommonAreaDone = false;  // Flag for rotation correction
bool isBinarizationDone = false;  // Flag for binarization
bool isChangeDetectionDone = false; // Flag for changeDetection
bool isProcessingDone = false;    // Flag for when all processing is done


std::vector<ImageData> sharedImageData(3); // For storing two frames
std::vector<ImageData> sharedImageDataNoise(3); // For storing two frames
std::vector<ImageData> sharedImageDataRotation(3); // For rotation correction
std::vector<ImageData> sharedImageDataCommon(3); // For rotation correction
std::vector<ImageData> sharedImageDataBinarized(3); // For binary


void loaderThread(const std::string& videoPath, const std::string& outputFolder) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return;
    }
    double fps = cap.get(cv::CAP_PROP_FPS); // Get frames per second
    int frameInterval = static_cast<int>(fps); // Number of frames to skip for 1 frame per second

    int frameCount = 0;
    cv::Mat frame;

    while (true) {
        std::unique_lock<std::mutex> lock(mtx_loader);
        cv_loader.wait(lock, [] { return !isDataReady; }); // Wait if data is already being processed

        sharedImageData.clear();
        for (int i = 0; i < 3; ++i) {  // Load two frames per iteration
            if (cap.read(frame)) {
                ImageData imgData;
                loadImageToGPU(frame, imgData);
                frameCount++;
                imgData.outputPath = outputFolder + "/frame_" + std::to_string(frameCount) + ".jpg";
                sharedImageData.push_back(imgData);
                std::cout << "Frame " << frameCount << " loaded to GPU." << std::endl;

                // Skip frames to get the next frame after 1 second
                cap.set(cv::CAP_PROP_POS_FRAMES, cap.get(cv::CAP_PROP_POS_FRAMES) + frameInterval - 1);
            } else {
                break;  // Exit loop if no more frames
            }
        }

        if (sharedImageData.size() < 3) {
            isProcessingDone = true;
            cv_saver.notify_all();
            return;
        }

        isDataReady = true;
        cv_noise.notify_all();
    }
}

void noiseReductionThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_noise);
        cv_noise.wait(lock, [] { return isDataReady || isProcessingDone; });
        sharedImageDataNoise.clear();

        if (isDataReady) {
            std::thread t1(applyNoiseReduction, std::ref(sharedImageData[0]));
            std::thread t2(applyNoiseReduction, std::ref(sharedImageData[1]));
            std::thread t3(applyNoiseReduction, std::ref(sharedImageData[2]));

            t1.join();
            t2.join();
            t3.join();
            sharedImageDataNoise.push_back(sharedImageData[0]);
            sharedImageDataNoise.push_back(sharedImageData[1]);
            sharedImageDataNoise.push_back(sharedImageData[2]);

            isNoiseReductionDone = true; // Indicate noise reduction is done for at least one image
            isDataReady = false; // Reset data ready flag
            cv_rotation.notify_all(); // Notify saver thread to start saving
        } else if (isProcessingDone) {
            break;
        }
    }
}

void rotationCorrectionThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_rotation);
        cv_rotation.wait(lock, [] { return isNoiseReductionDone || isProcessingDone; });

        if (isNoiseReductionDone) {
            if (sharedImageDataNoise.size() < 3) {
                std::cerr << "Error: Not enough images for rotation correction!" << std::endl;
                return;
            }
            //applyRotationCorrection(std::ref(sharedImageDataNoise[0]), std::ref(sharedImageDataNoise[1]));
            
            sharedImageDataRotation.clear();
            std::thread t1(applyRotationCorrection, std::ref(sharedImageDataNoise[1]), std::ref(sharedImageDataNoise[0]) );
            std::thread t2(applyRotationCorrection, std::ref(sharedImageDataNoise[1]), std::ref(sharedImageDataNoise[2]) );
            
            t1.join();
            t2.join();    
            sharedImageDataRotation.push_back(sharedImageDataNoise[0]);
            sharedImageDataRotation.push_back(sharedImageDataNoise[1]);
            sharedImageDataRotation.push_back(sharedImageDataNoise[2]);       


            isRotationDone = true;
            isNoiseReductionDone = false;
            cv_common.notify_all();
        } else if (isProcessingDone) {
            break;
        }
    }
}


void commonAreaExtractionThread() {
    while (!isProcessingDone) {
        std::unique_lock<std::mutex> lock(mtx_common);
        cv_common.wait(lock, [] { return isRotationDone || isProcessingDone; });
        if (isProcessingDone) break;

        std::cout << "Common area extraction starting..." << std::endl;
        if (sharedImageDataRotation.size() < 3) {
            std::cerr << "Error: Not enough rotated images for common area extraction!" << std::endl;
            isProcessingDone = true;
            cv_saver.notify_all();
            break;
        }

        sharedImageDataCommon.clear();
        cv::cuda::GpuMat commonPrev, commonCurr, commonNext;
        applyCommonAreaExtraction(sharedImageDataRotation[0], sharedImageDataRotation[1], sharedImageDataRotation[2],
                                  commonPrev, commonCurr, commonNext);

         //Update rotation_ref with cropped common areas
         if (sharedImageDataRotation[0].rotation_ref != nullptr) delete sharedImageDataRotation[0].rotation_ref;
         if (sharedImageDataRotation[1].rotation_ref != nullptr) delete sharedImageDataRotation[1].rotation_ref;
         if (sharedImageDataRotation[2].rotation_ref != nullptr) delete sharedImageDataRotation[2].rotation_ref;
         sharedImageDataRotation[0].rotation_ref = new cv::cuda::GpuMat(commonPrev);
         sharedImageDataRotation[1].rotation_ref = new cv::cuda::GpuMat(commonCurr);
         sharedImageDataRotation[2].rotation_ref = new cv::cuda::GpuMat(commonNext);

        //sharedImageDataCommon = sharedImageDataRotation;

        isCommonAreaDone = true;
        isRotationDone = false;
        cv_binarization.notify_all();
    }
}




void binarizationThread() {
    while (!isProcessingDone) {
        std::unique_lock<std::mutex> lock(mtx_binarization);
        cv_binarization.wait(lock, [] { return isCommonAreaDone || isProcessingDone; });
        if (isProcessingDone) break;

        sharedImageDataBinarized.clear();
        for (auto& imgData : sharedImageDataCommon) {
            applyBinarization(imgData);
            sharedImageDataBinarized.push_back(imgData);
        }

        isBinarizationDone = true;
        isCommonAreaDone = false;
        cv_change.notify_all();
    }
}

void changeDetectionThread() {
    while (!isProcessingDone) {
        std::unique_lock<std::mutex> lock(mtx_change);
        cv_change.wait(lock, [] { return isBinarizationDone || isProcessingDone; });
        if (isProcessingDone) break;

        if (sharedImageDataBinarized.size() < 3) {
            std::cerr << "Error: Not enough binarized images for change detection!" << std::endl;
            isProcessingDone = true;
            cv_saver.notify_all();
            break;
        }

        // Compute change mask for the middle frame (curr)
        cv::cuda::GpuMat changeMask;
        applyChangeDetection(sharedImageDataBinarized[0], sharedImageDataBinarized[1], sharedImageDataBinarized[2], changeMask);

        // Update the middle frame’s binary_ref with the change mask
        if (sharedImageDataBinarized[1].binary_ref != nullptr) {
            delete sharedImageDataBinarized[1].binary_ref; // Free old binary_ref
        }
        sharedImageDataBinarized[1].binary_ref = new cv::cuda::GpuMat(changeMask);

        isChangeDetectionDone = true;
        isBinarizationDone = false;
        cv_saver.notify_all();
    }
}



void saverThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_saver);
        cv_saver.wait(lock, [] { return isChangeDetectionDone|| isProcessingDone; });

        if (isChangeDetectionDone) {
            std::cout << "Image Saver" << std::endl;

            for (auto& imgData : sharedImageDataBinarized) {
                saveImageFromGPU(imgData);
                freeGPUData(imgData);
            }
	        isChangeDetectionDone = false;  // Reset flag for next batch
          //  isDataReady = false;
            cv_loader.notify_all(); // Notify loader thread to load new images
        } else if (isProcessingDone) {
            break;
        }
        //std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int main() {
    std::string videoPath = "/home/asith/Desktop/FYP/Testing_Pipeline_C/input_video/video1.mp4";
    std::string outputFolder = "/home/asith/Desktop/FYP/Testing_Pipeline_C/output_frames";

    std::thread loader(loaderThread, videoPath, outputFolder);
    std::thread noiseReducer(noiseReductionThread);
    std::thread rotator(rotationCorrectionThread); 
    std::thread commonExtractor(commonAreaExtractionThread);
    std::thread binarizer(binarizationThread);
    std::thread changeDetector(changeDetectionThread);
    std::thread saver(saverThread);

    loader.join();
    saver.join();

    std::cout << "All frames processed successfully!" << std::endl;

    return 0;
}

