#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "side_window_filter.h"
#include "box_filter.h"

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

void imshowMany(const std::string& _winName, const std::vector<cv::Mat>& ployImages, const std::string& filename="")
{
	int nImg = (int)ployImages.size();//获取在同一画布中显示多图的数目  

	cv::Mat dispImg;

	if (nImg <= 0)
	{
		printf("Number of arguments too small....\n");
		return;
	}

  int w = ployImages[0].cols;
  int h = ployImages[0].rows;
  dispImg.create(cv::Size(w * nImg, h), ployImages[0].type());
  for (int i = 0; i < nImg; ++i) {
		int x = ployImages[i].cols; //第(i+1)张子图像的宽度(列数)  
		int y = ployImages[i].rows;//第(i+1)张子图像的高度（行数）  
		cv::Mat imgROI = dispImg(cv::Rect(w * i, 0, x, y)); //在画布dispImage中划分ROI区域  
		cv::resize(ployImages[i], imgROI, cv::Size(x, y)); //将要显示的图像设置为ROI区域大小  
  }

	cv::imshow(_winName, dispImg);
  if (filename != "") {
    cv::imwrite(filename, dispImg);
  }
}

int main(int argc, char *argv[]) {
  // const std::string img_name = "images/jian20_small_salt_pepper_noise.jpg";
  const std::string img_name = "images/rgbsmall.png";

  cv::Mat img         = cv::imread(img_name);
  cv::Mat dst_mean    = cv::Mat::zeros(img.size(), img.type());
  cv::Mat dst_median  = cv::Mat::zeros(img.size(), img.type());
  cv::Mat dst_side    = cv::Mat::zeros(img.size(), img.type());

  int height      = img.rows;
  int width       = img.cols;
  int channel     = img.channels();
  int radius      = 3;
  int iteration   = 3;

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
  std::vector<cv::Mat> imgs(iteration + 1);
  imgs[0] = img;
  for (int c = 0; c < channel; ++c) {
    side_window_filter(channels[c].data(), height, width, results[c].data());
  }
  mergeChannels(results, height, width, channel, dst_side.data);
  imgs[1] = dst_side;
  for (int i = 2; i <= iteration; ++i) {
    for (int c = 0; c < channel; ++c) {
      side_window_filter(results[c].data(), height, width, results[c].data());
    }
    mergeChannels(results, height, width, channel, dst_side.data);
    imgs[i] = dst_side;
  }
  imshowMany("side", imgs, "sidewindow/results/side.png");

  for (int c = 0; c < channel; ++c) {
    mean_filter(channels[c].data(), height, width, radius, results[c].data());
  }
  mergeChannels(results, height, width, channel, dst_mean.data);
  imgs[1] = dst_mean;
  for (int i = 2; i <= iteration; ++i) {
      for (int c = 0; c < channel; ++c) {
          mean_filter(results[c].data(), height, width, radius, results[c].data());
      }
      mergeChannels(results, height, width, channel, dst_mean.data);
      imgs[i] = dst_mean;
  }
  imshowMany("mean", imgs, "sidewindow/results/mean.png");

  for (int c = 0; c < channel; ++c) {
    median_filter(channels[c].data(), height, width, radius, results[c].data());
  }
  mergeChannels(results, height, width, channel, dst_median.data);
  imgs[1] = dst_median;
  for (int i = 2; i <= iteration; ++i) {
      for (int c = 0; c < channel; ++c) {
          median_filter(results[c].data(), height, width, radius, results[c].data());
      }
      mergeChannels(results, height, width, channel, dst_median.data);
      imgs[i] = dst_median;
  }
  imshowMany("median", imgs, "sidewindow/results/median.png");

  cv::waitKey(0);
  return 0;
}