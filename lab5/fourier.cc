#include "complex.h"
#include "fourier.h"

#include <cstdio>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

const double PI = acos(-1);

int DFT(const Fourier &src, Fourier &dst) {
  if (src.mat.empty()) {
    return 1;
  }

  int N = src.mat.rows;
  int M = src.mat.cols;
  int channels = src.mat.channels();

  dst.mat = cv::Mat::zeros(src.mat.size(), src.mat.type());
  dst.coef = std::vector<std::vector<std::vector<complex>>>(channels, std::vector<std::vector<complex>>(N, std::vector<complex>(M)));

  double theta;
  std::vector<std::vector<complex>> vecX(N, std::vector<complex>(N));
  for (int u = 0; u < N; ++u) {
    for (int x = 0; x < N; ++x) {
      theta = -2.0 * PI * (double)u * (double)x / (double)N;
      vecX[u][x] = complex(cos(theta), sin(theta));
    }
  }
  std::vector<std::vector<complex>> vecY(M, std::vector<complex>(M));
  for (int v = 0; v < M; ++v) {
    for (int y = 0; y < M; ++y) {
      theta = -2.0 * PI * (double)v * (double)y / (double)M;
      vecY[v][y] = complex(cos(theta), sin(theta));
    }
  }

  complex c;
  double A = sqrt(N * M);
  for (int ch = 0; ch < channels; ++ch) {
    std::vector<std::vector<complex>> g(M, std::vector<complex>(N));
    for (int v = 0; v < M; ++v) {
      for (int x = 0; x < N; ++x) {
        c.real = c.imag = 0;
        for (int y = 0; y < M; ++y) {
          c += vecY[v][y] * (double)src.mat.at<cv::Vec3b>(x, y)[ch];
        }
        g[v][x] = c;
      }
    }

    for (int u = 0; u < N; ++u) {
      for (int v = 0; v < M; ++v) {
        c.real = c.imag = 0;
        for (int x = 0; x < N; ++x) {
          c += vecX[u][x] * g[v][x];
        }
        dst.coef[ch][u][v] = c / A;
        dst.mat.at<cv::Vec3b>(u, v)[ch] = fmin(fmax(dst.coef[ch][u][v].abs(), 0), 255);
      }
    }
  }

  return 0;
}

int IDFT(const Fourier &src, cv::Mat &dst) {
  int channels = src.coef.size();
  if (channels < 1) {
    return 1;
  }
  int N = src.coef[0].size();
  if (N < 1) {
    return 1;
  }
  int M = src.coef[0][0].size();
  if (M < 1) {
    return 1;
  }
  int type[] = {CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4};
  dst = cv::Mat::zeros(cvSize(M, N), type[channels - 1]);

  double theta;
  std::vector<std::vector<complex>> vecX(N, std::vector<complex>(N));
  for (int u = 0; u < N; ++u) {
    for (int x = 0; x < N; ++x) {
      theta = 2.0 * PI * (double)u * (double)x / (double)N;
      vecX[u][x] = complex(cos(theta), sin(theta));
    }
  }
  std::vector<std::vector<complex>> vecY(M, std::vector<complex>(M));
  for (int v = 0; v < M; ++v) {
    for (int y = 0; y < M; ++y) {
      theta = 2.0 * PI * (double)v * (double)y / (double)M;
      vecY[v][y] = complex(cos(theta), sin(theta));
    }
  }
  complex c;
  double A = sqrt(N * M);
  for (int ch = 0; ch < channels; ++ch) {
    std::vector<std::vector<complex>> g(M, std::vector<complex>(N));
    for (int v = 0; v < M; ++v) {
      for (int x = 0; x < N; ++x) {
        c.real = c.imag = 0;
        for (int y = 0; y < M; ++y) {
          c += vecY[v][y] * src.coef[ch][x][y];
        }
        g[v][x] = c;
      }
    }

    for (int u = 0; u < N; ++u) {
      for (int v = 0; v < M; ++v) {
        c.real = c.imag = 0;
        for (int x = 0; x < N; ++x) {
          c += vecX[u][x] * g[v][x];
        }
        c /= A;
        dst.at<cv::Vec3b>(u, v)[ch] = (uchar)fmin(fmax(c.abs(), 0), 255);
      }
    }
  }

  return 0; 
}

int shift(cv::Mat &img) {
    if (img.empty()) {
        return 1;
    }

    int N = img.rows;
    int M = img.cols;
    
    cv::Mat tmp;
    cv::Mat q0(img, cv::Rect(0, 0, M / 2, N / 2));
    cv::Mat q1(img, cv::Rect(M / 2, 0, M / 2, N/ 2));
    cv::Mat q2(img, cv::Rect(0, N / 2, M / 2, N / 2));
    cv::Mat q3(img, cv::Rect(M / 2, N / 2, M / 2, N / 2));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    return 0;
}


int ILPF(Fourier src, Fourier &dst, int d) {
  int channels = src.coef.size();
  if (channels < 1) {
    return 1;
  }
  int N = src.coef[0].size();
  if (N < 1) {
    return 1;
  }
  int M = src.coef[0][0].size();
  if (M < 1) {
    return 1;
  }

  dst.coef = std::vector<std::vector<std::vector<complex>>>(channels, std::vector<std::vector<complex>>(N, std::vector<complex>(M)));

  double r = d * d;
  int u = N / 2;
  int v = M / 2;

  cv::Mat tmp = cv::Mat::zeros(cvSize(M, N), CV_8UC3);

  for (int ch = 0, tx, ty; ch < channels; ++ch) {
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < M; ++y) {
        if (x < u) {
          tx = x + u;
        } else {
          tx = x - u;
        }
        if (y < v) {
          ty = y + v;
        } else {
          ty = y - v;
        }
        dst.coef[ch][tx][ty] = ((x - u) * (x - u) + (y - v) * (y - v)) <= r ? src.coef[ch][tx][ty] : 0;
        tmp.at<cv::Vec3b>(tx, ty)[ch] = fmin(fmax(dst.coef[ch][tx][ty].abs(), 0), 255);
      }
    }
  }
  shift(tmp);
  cv::imshow("ILPF-", tmp);

  return 0; //IDFT(dst, dst.mat);
}

int IHPF(Fourier src, Fourier &dst, int d) {
  int channels = src.coef.size();
  if (channels < 1) {
    return 1;
  }
  int N = src.coef[0].size();
  if (N < 1) {
    return 1;
  }
  int M = src.coef[0][0].size();
  if (M < 1) {
    return 1;
  }

  dst.coef = std::vector<std::vector<std::vector<complex>>>(channels, std::vector<std::vector<complex>>(N, std::vector<complex>(M)));

  double r = d * d;
  int u = N / 2;
  int v = M / 2;

  cv::Mat tmp = cv::Mat::zeros(cvSize(M, N), CV_8UC3);

  for (int ch = 0, tx, ty; ch < channels; ++ch) {
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < M; ++y) {
        if (x < u) {
          tx = x + u;
        } else {
          tx = x - u;
        }
        if (y < v) {
          ty = y + v;
        } else {
          ty = y - v;
        }
        dst.coef[ch][tx][ty] = ((x - u) * (x - u) + (y - v) * (y - v)) > r ? src.coef[ch][tx][ty] : 0;
        tmp.at<cv::Vec3b>(tx, ty)[ch] = fmin(fmax(dst.coef[ch][tx][ty].abs(), 0), 255);
      }
    }
  }
  shift(tmp);
  cv::imshow("IHPF-", tmp);

  return 0;// IDFT(dst, dst.mat);
}

int BLPF(Fourier src, Fourier &dst, int d, int n) {
  int channels = src.coef.size();
  if (channels < 1) {
    return 1;
  }
  int N = src.coef[0].size();
  if (N < 1) {
    return 1;
  }
  int M = src.coef[0][0].size();
  if (M < 1) {
    return 1;
  }

  dst.coef = std::vector<std::vector<std::vector<complex>>>(channels, std::vector<std::vector<complex>>(N, std::vector<complex>(M)));

  double r = d * d;
  int u = N / 2;
  int v = M / 2;

  cv::Mat tmp = cv::Mat::zeros(cvSize(M, N), CV_8UC3);
  double h;
  for (int ch = 0, tx, ty; ch < channels; ++ch) {
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < M; ++y) {
        if (x < u) {
          tx = x + u;
        } else {
          tx = x - u;
        }
        if (y < v) {
          ty = y + v;
        } else {
          ty = y - v;
        }
        d = (x - u) * (x - u) + (y - v) * (y - v);
        h = (double)1 / (pow(d / r, n) + 1);
        dst.coef[ch][tx][ty] = src.coef[ch][tx][ty] * h;
        tmp.at<cv::Vec3b>(tx, ty)[ch] = fmin(fmax(dst.coef[ch][tx][ty].abs(), 0), 255);
      }
    }
  }

  shift(tmp);
  cv::imshow("BLPF-", tmp);

  return 0;// IDFT(dst, dst.mat);
}

int BHPF(Fourier src, Fourier &dst, int d, int n) {
    int channels = src.coef.size();
  if (channels < 1) {
    return 1;
  }
  int N = src.coef[0].size();
  if (N < 1) {
    return 1;
  }
  int M = src.coef[0][0].size();
  if (M < 1) {
    return 1;
  }

  dst.coef = std::vector<std::vector<std::vector<complex>>>(channels, std::vector<std::vector<complex>>(N, std::vector<complex>(M)));

  double r = d * d;
  int u = N / 2;
  int v = M / 2;

  cv::Mat tmp = cv::Mat::zeros(cvSize(M, N), CV_8UC3);
  double h;
  for (int ch = 0, tx, ty; ch < channels; ++ch) {
    for (int x = 0; x < N; ++x) {
      for (int y = 0; y < M; ++y) {
        if (x < u) {
          tx = x + u;
        } else {
          tx = x - u;
        }
        if (y < v) {
          ty = y + v;
        } else {
          ty = y - v;
        }
        d = (x - u) * (x - u) + (y - v) * (y - v);
        h = (double)1 / (pow(d / r, n) + 1);
        h = (double)1 - h ;
        dst.coef[ch][tx][ty] = src.coef[ch][tx][ty] * h;
        tmp.at<cv::Vec3b>(tx, ty)[ch] = fmin(fmax(dst.coef[ch][tx][ty].abs(), 0), 255);
      }
    }
  }

  shift(tmp);
  cv::imshow("BHPF-", tmp);

  return 0;// IDFT(dst, dst.mat);
}
