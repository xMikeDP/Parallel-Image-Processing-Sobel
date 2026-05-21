#include "sobelalgorithm.h"

using namespace std;

// Sequential sobel algorithm
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