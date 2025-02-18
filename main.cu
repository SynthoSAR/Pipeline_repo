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

std::mutex mtx;
std::condition_variable cv_sync;
bool isDataReady = false;
bool isProcessingDone = false;

std::vector<ImageData> sharedImageData(2); // For storing two frames

void loaderThread(const std::string& videoPath, const std::string& outputFolder) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS); // Get frames per second
    int frameInterval = static_cast<int>(fps* 0.04); // Number of frames to skip for 1 frame per second

    int frameCount = 0;
    cv::Mat frame;

    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_sync.wait(lock, [] { return !isDataReady; });

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

        if (sharedImageData.empty()) {
            isProcessingDone = true;
            cv_sync.notify_one();
            return;
        }

        isDataReady = true;
        cv_sync.notify_one();
    }
}


void saverThread() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_sync.wait(lock, [] { return isDataReady || isProcessingDone; });

        if (isDataReady) {
            for (auto& imgData : sharedImageData) {
                saveImageFromGPU(imgData);
                freeGPUData(imgData);
            }

            isDataReady = false;
            cv_sync.notify_one();
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
    std::thread saver(saverThread);

    loader.join();
    saver.join();

    std::cout << "All frames processed successfully!" << std::endl;

    return 0;
}

