#include "sobelalgorithm.h"
#include <omp.h>

using namespace std;

// Sequential
void sobelSequential(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;
    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int pixel = input.at<uchar>(y + i, x + j);
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }
            int magnitude = sqrt((sumX * sumX) + (sumY * sumY));
            output.at<uchar>(y, x) = min(255, max(0, magnitude));
        }
    }
}

// OpenMP
void sobelOpenMP(const cv::Mat& input, cv::Mat& output, int numThreads) {
    int rows = input.rows;
    int cols = input.cols;
    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    omp_set_num_threads(numThreads);

    // split up for loop iterations (first come first served)
    #pragma omp parallel for schedule(dynamic)
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            int sumX = 0, sumY = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int pixel = input.at<uchar>(y + i, x + j);
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }
            int magnitude = sqrt((sumX * sumX) + (sumY * sumY));
            output.at<uchar>(y, x) = min(255, max(0, magnitude));
        }
    }
}

// Intel TBB
void sobelTBB(const cv::Mat& input, cv::Mat& output) {
    int rows = input.rows;
    int cols = input.cols;
    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // split the rows from 1 to rows-1 among the threads
    tbb::parallel_for(tbb::blocked_range<int>(1, rows - 1), [&](const tbb::blocked_range<int>& range) {
        for (int y = range.begin(); y != range.end(); ++y) {
            for (int x = 1; x < cols - 1; ++x) {
                int sumX = 0, sumY = 0;
                for (int i = -1; i <= 1; ++i) {
                    for (int j = -1; j <= 1; ++j) {
                        int pixel = input.at<uchar>(y + i, x + j);
                        sumX += pixel * Gx[i + 1][j + 1];
                        sumY += pixel * Gy[i + 1][j + 1];
                    }
                }
                int magnitude = sqrt((sumX * sumX) + (sumY * sumY));
                output.at<uchar>(y, x) = min(255, max(0, magnitude));
            }
        }
        });
}