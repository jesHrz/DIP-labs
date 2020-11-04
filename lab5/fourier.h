#ifndef FOURIER_H_
#define FOURIER_H_

#include "complex.h"
#include <vector>

#include <opencv2/core.hpp>

struct Fourier {
  std::vector<std::vector<std::vector<complex>>> coef;
  cv::Mat mat;
};

int DFT(const Fourier &src, Fourier &dst);

int IDFT(const Fourier &src, cv::Mat &dst);

int shift(cv::Mat &img);

int ILPF(Fourier src, Fourier &dst, int d);

int IHPF(Fourier src, Fourier &dst, int d);

int BLPF(Fourier src, Fourier &dst, int d, int n);

int BHPF(Fourier src, Fourier &dst, int d, int n);

#endif // FOURIER_H_