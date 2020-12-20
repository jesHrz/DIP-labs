#include "box_filter.h"

#include <vector>

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