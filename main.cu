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

std::mutex mtx_loader;   // Mutex for loader thread
std::mutex mtx_noise;    // Mutex for noise reduction threads
std::mutex mtx_rotation;  // Mutex for rotation correction
std::mutex mtx_saver;    // Mutex for saver thread

std::condition_variable cv_loader;  // Condition variable for loader
std::condition_variable cv_noise;   // Condition variable for noise reduction threads
std::condition_variable cv_rotation;  // Condition variable for rotation correction
std::condition_variable cv_saver;   // Condition variable for saver thread

bool isDataReady = false;         // Flag for when data is ready for noise reduction
bool isNoiseReductionDone = false; // Flag for when noise reduction is done
bool isRotationDone = false;  // Flag for rotation correction
bool isProcessingDone = false;    // Flag for when all processing is done


std::vector<ImageData> sharedImageData(2); // For storing two frames
std::vector<ImageData> sharedImageDataNoise(2); // For storing two frames

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
        for (int i = 0; i < 2; ++i) {  // Load two frames per iteration
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

        if (sharedImageData.size() < 2) {
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

            t1.join();
            t2.join();
            sharedImageDataNoise.push_back(sharedImageData[0]);
            sharedImageDataNoise.push_back(sharedImageData[1]);

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
            if (sharedImageDataNoise.size() < 2) {
                std::cerr << "Error: Not enough images for rotation correction!" << std::endl;
                return;
            }

            applyRotationCorrection(std::ref(sharedImageDataNoise[0]), std::ref(sharedImageDataNoise[1]));

            isRotationDone = true;
            isNoiseReductionDone = false;
            cv_saver.notify_all();
        } else if (isProcessingDone) {
            break;
        }
    }
}



void saverThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx_saver);
        cv_saver.wait(lock, [] { return isRotationDone|| isProcessingDone; });

        if (isRotationDone) {
            std::cout << "Image Saver" << std::endl;

            for (auto& imgData : sharedImageDataNoise) {
                saveImageFromGPU(imgData);
                freeGPUData(imgData);
            }
	    isRotationDone = false;  // Reset flag for next batch
          //  isDataReady = false;
            cv_loader.notify_all(); // Notify loader thread to load new images
        } else if (isProcessingDone) {
            break;
        }
        //std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int main() {
    std::string videoPath = "/home/asith/Desktop/Testing_Pipeline_C/input_video/video.mp4";
    std::string outputFolder = "/home/asith/Desktop/Testing_Pipeline_C/output_frames";

    std::thread loader(loaderThread, videoPath, outputFolder);
    std::thread noiseReducer(noiseReductionThread);
    std::thread rotator(rotationCorrectionThread); 
    std::thread saver(saverThread);

    loader.join();
    saver.join();

    std::cout << "All frames processed successfully!" << std::endl;

    return 0;
}

