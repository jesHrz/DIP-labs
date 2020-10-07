#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

const float pi = acos(-1);

int gaussian(const Mat &src, Mat &dst, float sigma) {
  if (src.empty() || fabs(sigma) < 1e-6) {
    return 1;
  }

  Mat tmp = Mat::zeros(src.size(), src.type());
  dst = Mat::zeros(src.size(), src.type());

  int w = ((int)((sigma * 6 - 1) / 2) * 2 + 1) / 2;
  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();

  std::vector<float> kernel(2 * w + 1, 0);
  float sum = 0;
  for (int i = -w; i <= w; ++i) {
    kernel[i + w] = exp(-1.0 * i * i / (sigma * sigma * 2)) / (sqrt(pi * 2) * sigma);
    sum += kernel[i + w];
  }
  for (int i = -w; i <= w; ++i) {
    kernel[i + w] /= sum;
  }

  for (int ch = 0; ch < channels; ++ch) {
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        float sum = 0.5;
        for (int i = -w; i <= w; ++i) {
          if (col + i < 0) {
            sum += kernel[i + w] * src.at<Vec3b>(row, -(col + i))[ch];
          } else if (col + i >= cols) {
            sum += kernel[i + w] * src.at<Vec3b>(row, cols - (col + i - cols) - 1)[ch];
          } else {
            sum += kernel[i + w] * src.at<Vec3b>(row, col + i)[ch];
          }
        }
        tmp.at<Vec3b>(row, col)[ch] = (int)sum;
      }
    }
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        float sum = 0.5;
        for (int i = -w; i <= w; ++i) {
          if (row + i < 0) {
            sum += kernel[i + w] * tmp.at<Vec3b>(-(row + i), col)[ch];
          } else if (row + i >= rows) {
            sum += kernel[i + w] * tmp.at<Vec3b>(rows - (row + i - rows) - 1, col)[ch];
          } else {
            sum += kernel[i + w] * tmp.at<Vec3b>(row + i, col)[ch];
          }
        }
        dst.at<Vec3b>(row, col)[ch] = (int)sum;
      }
    }
  } 

  return 0;
}

int median(const Mat &src, Mat &dst, int w) {
  if (src.empty() || w < 0) {
    return 1;
  }

  if (w == 0) {
    dst = src;
    return 0;
  }

  dst = Mat::zeros(src.size(), src.type());

  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();

  std::vector<int> avg;
  for (int ch = 0; ch < channels; ++ch) {
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        avg.clear();
        for (int i = -w; i <= w; ++i) {
          int x = row + i < 0 ? -(row + i) : (row + i >= rows ? rows - (row + i - rows) - 1 : row + i);
          for (int j = -w; j <= w; ++j) {
            int y = col + j < 0 ? -(col + j) : (col + j >= cols ? cols - (col + j - cols) - 1 : col + j);
            avg.push_back(src.at<Vec3b>(x, y)[ch]);
          }
        }
        std::sort(avg.begin(), avg.end());
        dst.at<Vec3b>(row, col)[ch] = avg[avg.size() / 2];
      }
    }
  }

  return 0;
}

int mean(const Mat &src, Mat &dst, int w) {
  if (src.empty() || w < 0) {
    return 1;
  }

  if (w == 0) {
    dst = src;
    return 0;
  }

#define PIXEL(x, y) src.at<Vec3b>( \
        i - w < 0 ? w - i : (i - w >= rows ? rows - (i - w - rows) - 1 : i - w),\
        j - w < 0 ? w - j : (j - w >= cols ? cols - (j - w - cols) - 1 : j - w))[ch]
#define SUM(x, y) sum[x + w][y + w]

  dst = Mat::zeros(src.size(), src.type());
  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();
  int n = (2 * w + 1) * (2 * w + 1);

  for (int ch = 0; ch < channels; ++ch) {
    std::vector<std::vector<int>> sum(rows + 2 * w + 1, std::vector<int>(cols + 2 * w + 1, 0));
    // prefix sum
    for (int i = 0; i < rows + 2 * w; ++i) {
      for (int j = 0; j < cols + 2 * w; ++j) {
        sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + PIXEL(i, j);
      }
    }

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        dst.at<Vec3b>(row, col)[ch] = (int)((float)(SUM(row + w + 1, col + w + 1) - SUM(row + w + 1, col - w) - SUM(row - w, col + w + 1) + SUM(row - w, col - w)) / n + 0.5);
      }
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  const std::string img_name = "images/fig1.png";
  Mat src = imread(img_name);
  Mat dst_guassin, dst_median, dst_mean;

  imshow("src", src);

  if (!gaussian(src, dst_guassin, argc > 1 ? atof(argv[1]) : 1)) {
    imshow("gaussian", dst_guassin);
  }

  if (!median(src, dst_median, argc > 2 ? atoi(argv[2]) : 1)) {
    imshow("median", dst_median);
  }

  if (!mean(src, dst_mean, argc > 2 ? atoi(argv[2]) : 1)) {
    imshow("mean", dst_mean);
  }

  waitKey();
  return 0;
}