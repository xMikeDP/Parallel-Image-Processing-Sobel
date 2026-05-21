#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include "sobelalgorithm.h"

using namespace std;

int main() {
    string datasetFolder = "./dataset/";
    vector<string> images;

    if (filesystem::exists(datasetFolder) && filesystem::is_directory(datasetFolder)) {
        for (auto& image : filesystem::directory_iterator(datasetFolder)) {
            images.push_back(image.path().string());
        }
    }

    if (images.empty()) {
        cerr << "no images found";
        return -1;
    }
  
    int threads = 4;


    // Sequential
    auto start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        cv::Mat input = cv::imread(image, cv::IMREAD_GRAYSCALE);
        cv::Mat output = cv::Mat::zeros(input.size(), input.type());
        sobelSequential(input, output);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> seqTime = end - start;
    cout << "Sequential Time: " << seqTime.count() << " ms\n" << endl;

    // OpenMP
    start = chrono::high_resolution_clock::now();
    for (auto& image : images) {
        cv::Mat input = cv::imread(image, cv::IMREAD_GRAYSCALE);
        cv::Mat output = cv::Mat::zeros(input.size(), input.type());
        sobelOpenMP(input, output, threads);
    }
    end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ompTime = end - start;
    cout << "OpenMP Time (" << threads << " threads): " << ompTime.count() << " ms" << endl;
    cout << "OpenMP Speedup: " << (seqTime.count() / ompTime.count()) << "x\n" << endl;


    return 0;
}