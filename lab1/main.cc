#include <unistd.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "libs/freeimage/FreeImage.h"

using namespace cv;

// Intensity: Log
#define LOG_C 35
static float log_trans(float r) {
  return std::min(LOG_C * log(1 + r), (float)255);
}

// Intensity: Power
#define POWER_C 1
#define POWER_GAMMA 3.0
static float power_trans(float r) {
  return std::min((float)(POWER_C * r * POWER_GAMMA), (float)255);
}

// Contrast: increase
#define CONTRAST_K_UPPER (float)1.5
#define CONTRAST_B_UPPER -72
static float contrast_trans_upper(float r) {
  return std::min(std::max(CONTRAST_K_UPPER * r + CONTRAST_B_UPPER, (float)0),
                  (float)255);
}

// Contrast: decrease
#define CONTRAST_K_LOWER (float)0.556
#define CONTRAST_B_LOWER 0
static float contrast_trans_lower(float r) {
  return std::min(std::max(CONTRAST_K_LOWER * r + CONTRAST_B_LOWER, (float)0),
                  (float)255);
}

int transformation(const Mat &src, Mat &dst, float (*transform)(float)) {
  if (src.empty()) {
    return 1;
  }
  int rows = src.rows;
  int cols = src.cols;
  dst = Mat::zeros(src.size(), src.type());

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      switch (src.channels()) {
        case 3:
          // 3 channels: do trasformation for each channel
          dst.at<Vec3b>(row, col)[0] = transform(src.at<Vec3b>(row, col)[0]);
          dst.at<Vec3b>(row, col)[1] = transform(src.at<Vec3b>(row, col)[1]);
          dst.at<Vec3b>(row, col)[2] = transform(src.at<Vec3b>(row, col)[2]);
          break;
        case 1:
          // 1 channel: only do trasformation for GREY
          dst.at<uchar>(row, col) = transform(src.at<uchar>(row, col));
          break;
        default:
          break;
      }
    }
  }
  // change dst to 8bit
  convertScaleAbs(dst, dst);
  return 0;
}

void *show_gif(void *gif_path) {
  FreeImage_Initialise();

  // get width and height of GIF
  FIBITMAP *bitmap = FreeImage_Load(FIF_GIF, (char *)gif_path, GIF_DEFAULT);
  int width = FreeImage_GetWidth(bitmap);
  int height = FreeImage_GetHeight(bitmap);

  // get frame number of GIF
  FIMULTIBITMAP *gif = FreeImage_OpenMultiBitmap(FIF_GIF, (char *)gif_path, 0,
                                                 1, 0, GIF_PLAYBACK);
  int frame_count = FreeImage_GetPageCount(gif);

  RGBQUAD *palette = new RGBQUAD;
  // create a new Image
  IplImage *ipl_image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
  ipl_image->origin = 1;
  while (1) {
    for (int cur_frame = 0; cur_frame < frame_count; ++cur_frame) {
      // read per frame
      FIBITMAP *frame = FreeImage_LockPage(gif, cur_frame);
      for (int i = 0; i < height; ++i) {
        char *img_data_per_line =
            ipl_image->imageData + i * ipl_image->widthStep;
        for (int j = 0; j < width; ++j) {
          // get color of each pixel
          FreeImage_GetPixelColor(frame, j, height - i, palette);
          img_data_per_line[3 * j] = palette->rgbBlue;
          img_data_per_line[3 * j + 1] = palette->rgbGreen;
          img_data_per_line[3 * j + 2] = palette->rgbRed;
        }
      }
      // show the image
      cvShowImage("gif", ipl_image);
      if (cvWaitKey(80) == ' ') {
        FreeImage_UnlockPage(gif, frame, 1);
        goto end;
      }
      FreeImage_UnlockPage(gif, frame, 1);
    }
  }
end:
  // release resource
  delete palette;
  FreeImage_Unload(bitmap);
  FreeImage_DeInitialise();
  cvReleaseImage(&ipl_image);
  return 0;
}

int main() {
  const std::string gif_name = "images/fig1.gif";
  const std::string img_name = "images/fig1.png";
  const Mat img = imread(img_name, IMREAD_GRAYSCALE);
  Mat dst_log, dst_power, dst_contrast_upper, dst_contrast_lower;

  int pid = fork();
  if (pid < 0) {
    perror("fork err");
    exit(-1);
  } else if (pid == 0) {
    show_gif((void *)gif_name.c_str());
    return 0;
  }

  imshow("src", img);

  if (!transformation(img, dst_log, log_trans)) imshow("log", dst_log);

  if (!transformation(img, dst_power, power_trans)) imshow("power", dst_power);

  if (!transformation(img, dst_contrast_upper, contrast_trans_upper))
    imshow("contrast_upper", dst_contrast_upper);

  if (!transformation(img, dst_contrast_lower, contrast_trans_lower))
    imshow("contrast_lower", dst_contrast_lower);

  waitKey();
  return 0;
}
