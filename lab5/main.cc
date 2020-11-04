#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "complex.h"
#include "fourier.h"

using namespace cv;

int main() {
  // const std::string img_name = "images/fig1.png";
  // const std::string img_name = "images/whales.png";
  const std::string img_name = "images/rgbsmall.png";
  // const std::string img_name = "images/black.png";
  // const std::string img_name = "images/fig5.png";
  //  const std::string img_name = "images/pepper.png";
  Fourier src, dft, pf;
  src.mat = imread(img_name, IMREAD_COLOR);

  cv::imshow("src", src.mat);
  if (!DFT(src, dft)) {
    cv::imshow("DFT", dft.mat);
    if (!shift(dft.mat)) {
      cv::imshow("DFT-shift", dft.mat);
    }
  }

  if (!ILPF(dft, pf, 10) && !IDFT(pf, pf.mat)) {
    cv::imshow("ILPF", pf.mat);
  }
  if (!IHPF(dft, pf, 10) && !IDFT(pf, pf.mat)) {
    imshow("IHPF", pf.mat);
  }
  if (!BLPF(dft, pf, 10, 5) && !IDFT(pf, pf.mat)) {
    imshow("BLPF", pf.mat);
  }
  if (!BHPF(dft, pf, 10, 5) && !IDFT(pf, pf.mat)) {
    imshow("BHPF", pf.mat);
  }

  cv::waitKey();
  cv::destroyAllWindows();
  return 0;
}