#pragma once
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

void sobelSequential(const cv::Mat& input, cv::Mat& output);
void sobelOpenMP(const cv::Mat& input, cv::Mat& output, int numThreads);
void sobelTBB(const cv::Mat& input, cv::Mat& output);