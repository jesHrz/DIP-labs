#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

int histogram(const Mat &src, Mat &dst) {
  #define L 256

  if (src.empty()) {
    return 1;
  }

  dst = Mat::zeros(src.size(), src.type());

  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();

  int n = rows * cols;
  int p[L];

  for (int ch = 0; ch < channels; ++ch) {
    memset(p, 0, sizeof p);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        p[src.at<Vec3b>(row, col)[ch]] += L - 1;
      }
    }
    for (int i = 1; i < L; ++i) {
      p[i] += p[i - 1];
    } 
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        dst.at<Vec3b>(row, col)[ch] = (int)(0.5 + p[src.at<Vec3b>(row, col)[ch]] / n);
      }
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  const std::string img_name = "images/fig1.png";
  Mat src = imread(img_name);
  Mat dst;

  imshow("src", src);

  if (!histogram(src, dst)) {
    imshow("dst", dst);
  }

  waitKey();
  return 0;
}