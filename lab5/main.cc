#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

const double PI = acos(-1);

#define EPS 1e-6
#define EQUAL(x, y) (fabs((x) - (y)) < EPS)

struct complex {
  double real, imag;

  complex(double real = 0, double imag = 0): real(real), imag(imag) {}

  complex operator + (const complex &t) const { return complex(real + t.real, imag + t.imag); }
  complex operator - (const complex &t) const { return complex(real - t.real, imag - t.imag); }
  complex operator * (const complex &t) const { return complex(real * t.real - imag * t.imag, real * t.imag + imag * t.real); }
  complex operator * (double k)   const { return complex(real * k, imag * k); }
  complex operator / (const complex &t) const { return (*this * t.conjugate()) / t.abs2(); }
  complex operator / (double k) const { return complex(real / k, imag / k); }

  complex& operator += (const complex &t) { return *this = *this + t; }
  complex& operator -= (const complex &t) { return *this = *this - t; }
  complex& operator *= (const complex &t) { return *this = *this * t; }
  complex& operator *= (double k)   { return *this = *this * k; }
  complex& operator /= (const complex &t) { return *this = *this / t; }
  complex& operator /= (double k)   { return *this = *this / k; }

  bool operator == (const complex &t) const { return EQUAL(real, t.real) && EQUAL(imag, t.imag); }
  complex& operator = (const complex &t) { 
    if (this != &t) {
      this->real = t.real;
      this->imag = t.imag;
    }
    return *this;
  }

  complex conjugate() const { return complex(real, -imag); }
  double abs2() const { return real * real + imag * imag; }
  double abs()  const { return sqrt(abs2()); }
};

struct Fourier {
  std::vector<std::vector<std::vector<complex>>> coef;
  Mat mat;
};

int DFT(const Mat &src, Fourier &dst, bool shift) {
  if (src.empty()) {
    return 1;
  }

  int N = src.rows;
  int M = src.cols;
  int channels = src.channels();

  dst.mat = Mat::zeros(src.size(), src.type());
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
          c += vecY[v][y] * (double)src.at<Vec3b>(x, y)[ch];
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
        dst.mat.at<Vec3b>(u, v)[ch] = fmin(fmax(dst.coef[ch][u][v].abs(), 0), 255);
      }
    }
  }

  if (shift) {
    Mat tmp;
    Mat q0(dst.mat, Rect(0, 0, M / 2, N / 2));
    Mat q1(dst.mat, Rect(M / 2, 0, M / 2, N/ 2));
    Mat q2(dst.mat, Rect(0, N / 2, M / 2, N / 2));
    Mat q3(dst.mat, Rect(M / 2, N / 2, M / 2, N / 2));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
  }

  return 0;
}

int IDFT(const Fourier &src, Mat &dst) {
  if (src.mat.empty()) {
    return 1;
  }

  int N = src.mat.rows;
  int M = src.mat.cols;
  int channels = src.mat.channels();

  dst = Mat::zeros(src.mat.size(), src.mat.type());

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
        dst.at<Vec3b>(u, v)[ch] = (uchar)fmin(fmax(c.abs(), 0), 255);
      }
    }
  }

  return 0; 
}

int main() {
  // const std::string img_name = "images/fig1.png";
  // const std::string img_name = "images/whales.png";
  // const std::string img_name = "images/rgbsmall.png";
  const std::string img_name = "images/black.png";
  // const std::string img_name = "images/fig5.png";
  Mat src = imread(img_name, IMREAD_COLOR);
  Fourier mDFT;
  Mat mIDFT;
  // Mat dst_DFT, dst_IDFT;

  imshow("src", src);
  if (!DFT(src, mDFT, true)) {
    imshow("DFT-shift", mDFT.mat);
  }
  if (!IDFT(mDFT, mIDFT)) {
    imshow("IDFT", mIDFT);
  }
  waitKey();
  destroyAllWindows();
  return 0;
}