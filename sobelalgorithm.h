#pragma once
#include <opencv2/opencv.hpp>
#include <omp.h>

void sobelSequential(const cv::Mat& input, cv::Mat& output);
void sobelOpenMP(const cv::Mat& input, cv::Mat& output, int numThreads);