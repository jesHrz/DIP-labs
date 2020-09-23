#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

using namespace cv;

#define SCALE_X 0.6
#define SCALE_Y 0.6

// 两次水平一次垂直
float bilinear_x(float a, float b, float c, float d, float dx, float dy) {
  float h1 = a * (1.0 - dx) + b * dx;
  float h2 = c * (1.0 - dx) + d * dx;
  return h1 * (1 - dy) + h2 * dy;
}

// 两次垂直一次水平
float bilinear_y(float a, float b, float c, float d, float dx, float dy) {
  float h1 = a * (1.0 - dy) + c * dy;
  float h2 = b * (1.0 - dy) + d * dy;
  return h1 * (1 - dx) + h2 * dx;
}

int imscale(const Mat &src, Mat &dst, float sx, float sy,
            float (*bilinear)(float, float, float, float, float, float)) {
  if (src.empty()) 
    return 1;

  int srows = (int)(sx * src.rows);
  int scols = (int)(sy * src.cols);
  int channels = src.channels();
  dst = Mat::zeros(Size(scols, srows), src.type());

  float x, y;
  for (int row = 0; row < srows; ++row) {
    for (int col = 0; col < scols; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        // 缩放到原图对应的点
        x = (float)row / sx, y = (float)col / sy;
        if ((int)(x + 0.5) >= src.rows || (int)(y + 0.5) >= src.cols) continue;
        // 双线性差值
        dst.at<Vec3b>(row, col)[ch] = bilinear(
            src.at<Vec3b>((int)(x), (int)(y))[ch],              // left-top
            src.at<Vec3b>((int)(x), (int)(y + 0.5))[ch],        // right-top
            src.at<Vec3b>((int)(x + 0.5), (int)(y))[ch],        // left-bottom
            src.at<Vec3b>((int)(x + 0.5), (int)(y + 0.5))[ch],  // right-bottom
            x - (int)(x), y - (int)(y));
      }
    }
  }
  return 0;
}

int imwarp(const Mat &src, Mat &dst,
           float (*bilinear)(float, float, float, float, float, float)) {
  if (src.empty()) 
    return 1;

  int rows = src.rows;
  int cols = src.cols;
  int channels = src.channels();
  dst = Mat::zeros(src.size(), src.type());

  float x, y, tx, ty;
  float r, theta;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      for (int ch = 0; ch < channels; ++ch) {
        // normalize
        x = (float)(col - (float)cols / 2) / (float)cols * 2;
        y = (float)(row - (float)rows / 2) / (float)rows * 2;
        r = sqrt(x * x + y * y);
        theta = (1 - r) * (1 - r);

        tx = x, ty = y;
        if (r < 1) {
          tx = cos(theta) * x - sin(theta) * y;
          ty = sin(theta) * x + cos(theta) * y;
        }

        y = tx * (float)cols / 2 + (float)cols / 2;
        x = ty * (float)rows / 2 + (float)rows / 2;

        if ((int)(x + 0.5) >= src.rows || (int)(y + 0.5) >= src.cols) continue;
        // 双线性差值
        dst.at<Vec3b>(row, col)[ch] = bilinear(
            src.at<Vec3b>((int)(x), (int)(y))[ch],              // left-top
            src.at<Vec3b>((int)(x), (int)(y + 0.5))[ch],        // right-top
            src.at<Vec3b>((int)(x + 0.5), (int)(y))[ch],        // left-bottom
            src.at<Vec3b>((int)(x + 0.5), (int)(y + 0.5))[ch],  // right-bottom
            x - (int)(x), y - (int)(y));
      }
    }
  }
  return 0;
}

int main(int argc, char *argv[]) {
  const std::string img_name = "images/fig1.png";
  Mat src = imread(img_name);
  Mat dst;

  imshow("original", src);

  if (!imscale(src, dst, SCALE_X, SCALE_Y, bilinear_x)) 
    imshow("scaled", dst);

  if (!imwarp(src, dst, bilinear_x)) 
    imshow("warped", dst);

  waitKey();
  return 0;
}