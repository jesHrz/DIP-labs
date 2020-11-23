#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "side_window_filter.h"

void splitChannels(const uchar * const image, int height, int width, int channel, std::vector<std::vector<float>> &channels) {
  int pixel_num = height * width;
  const uchar *src = image;
  for (int i = 0; i < pixel_num; ++i) {
    for (int c = 0; c < channel; ++c) {
      channels[c][i] = (float)src[c];
    }
    src += channel;
  }
}

void mergeChannels(const std::vector<std::vector<float>> &channels, int height, int width, int channel, uchar * const image) {
  int pixel_num = height * width;

  for (int c = 0; c < channel; ++c) {
    uchar *src = image;
    for (int i = 0; i < pixel_num; ++i) {
      src[c] = (uchar)std::max(std::min(channels[c][i], 255.0f), 0.0f);
      src += channel;
    }
  }
}

void median_filter(const float * src, int height, int width, int radius, float *dst) {
  int n = (2 * radius + 1) * (2 * radius + 1);
  int pixel_num = height * width;
  std::vector<float> avg(n);
  std::vector<float> tmp(pixel_num);

  for (int h = 0; h < height; ++h) {
    int offset = h * width;
    for (int w = 0; w < width; ++w) {
      int count = 0;
      for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
          if (0 <= offset + w + (i * width + j) && offset + w + (i * width + j) < pixel_num) {
            avg[count++] = src[offset + w + (i * width + j)];
          }
        }
      }
      std::sort(&avg[0], &avg[count]);
      tmp[offset + w] = avg[count / 2];
    }
  }
  
  float *ptr = &tmp[0];
  float *dst_ptr = dst;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      dst_ptr[w] = ptr[w];
    }
    dst_ptr += width;
    ptr += width;
  }
}

void mean_filter(const float *src, int height, int width, int radius, float *dst) {
#define PIXEL(x, y) src[ \
      (h - radius < 0 ? radius - h : (h - radius >= height ? height - (h - radius - height) - 1 : h - radius)) * width + \
      (w - radius < 0 ? radius - w : (w - radius >= width ? width - (w - radius - width) - 1 : w - radius))]
#define SUM(x, y) sum[x + radius][y + radius]

  int n = (2 * radius + 1) * (2 * radius + 1);
  int pixel_num = height * width;
  std::vector<float> tmp(pixel_num);

  std::vector<std::vector<float>> sum(height + 2 * radius + 1, std::vector<float>(width + 2 * radius + 1)); 
  for (int h = 0; h < height + 2 * radius; ++h) {
      for (int w = 0; w < width + 2 * radius; ++w) {
        sum[h + 1][w + 1] = sum[h + 1][w] + sum[h][w + 1] - sum[h][w] + PIXEL(h, w);
      }
    }
  for (int h = 0; h < height; ++h) {
    int offset = h * width;
    for (int w = 0; w < width; ++w) {
      tmp[offset + w] = (SUM(h + radius + 1, w + radius + 1) - SUM(h + radius + 1, w - radius) - SUM(h - radius, w + radius + 1) + SUM(h - radius, w - radius)) / n;
    }
  }

  float *ptr = &tmp[0];
  float *dst_ptr = dst;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      dst_ptr[w] = ptr[w];
    }
    dst_ptr += width;
    ptr += width;
  }
}

int main(int argc, char *argv[]) {
  const std::string img_name = "images/jian20_small_salt_pepper_noise.jpg";

  cv::Mat img         = cv::imread(img_name);
  cv::Mat dst_mean    = cv::Mat::zeros(img.size(), img.type());
  cv::Mat dst_median  = cv::Mat::zeros(img.size(), img.type());
  cv::Mat dst_side    = cv::Mat::zeros(img.size(), img.type());

  int height      = img.rows;
  int width       = img.cols;
  int channel     = img.channels();
  int radius      = 3;
  int iteration   = 10;

  std::vector<std::vector<float>> channels;
  std::vector<std::vector<float>> results;

  channels.resize(channel);
  results.resize(channel);
  for (int i = 0; i < channel; ++i) {
    channels[i].resize(height * width);
    results[i].resize(height * width);
  }

  splitChannels(img.data, height, width, channel, channels);

  init(height, width, radius);
  for (int c = 0; c < channel; ++c) {
    side_window_filter(channels[c].data(), height, width, results[c].data());
  }
  for (int i = 0; i < iteration; ++i) {
    for (int c = 0; c < channel; ++c) {
      side_window_filter(results[c].data(), height, width, results[c].data());
    }
  }

  mergeChannels(results, height, width, channel, dst_side.data);

  for (int c = 0; c < channel; ++c) {
    mean_filter(channels[c].data(), height, width, radius, results[c].data());
  }
  for (int i = 0; i < iteration; ++i) {
      for (int c = 0; c < channel; ++c) {
          mean_filter(results[c].data(), height, width, radius, results[c].data());
      }
  }
  mergeChannels(results, height, width, channel, dst_mean.data);

  for (int c = 0; c < channel; ++c) {
    median_filter(channels[c].data(), height, width, radius, results[c].data());
  }
  for (int i = 0; i < iteration; ++i) {
      for (int c = 0; c < channel; ++c) {
          median_filter(results[c].data(), height, width, radius, results[c].data());
      }
  }
  mergeChannels(results, height, width, channel, dst_median.data);

  cv::imshow("original", img);
  cv::imshow("side-window", dst_side);
  cv::imshow("mean", dst_mean);
  cv::imshow("median", dst_median);
  cv::waitKey(0);
  return 0;
}