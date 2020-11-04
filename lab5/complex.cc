#include "complex.h"
#include <cmath>

#define EPS 1e-6
#define EQUAL(x, y) (fabs((x) - (y)) < EPS)

complex::complex(double real, double imag): real(real), imag(imag) {}

complex complex::operator + (const complex &t) const { return complex(real + t.real, imag + t.imag); }
complex complex::operator - (const complex &t) const { return complex(real - t.real, imag - t.imag); }
complex complex::operator * (const complex &t) const { return complex(real * t.real - imag * t.imag, real * t.imag + imag * t.real); }
complex complex::operator * (double k)   const { return complex(real * k, imag * k); }
complex complex::operator / (const complex &t) const { return (*this * t.conjugate()) / t.abs2(); }
complex complex::operator / (double k) const { return complex(real / k, imag / k); }

complex& complex::operator += (const complex &t) { return *this = *this + t; }
complex& complex::operator -= (const complex &t) { return *this = *this - t; }
complex& complex::operator *= (const complex &t) { return *this = *this * t; }
complex& complex::operator *= (double k)   { return *this = *this * k; }
complex& complex::operator /= (const complex &t) { return *this = *this / t; }
complex& complex::operator /= (double k)   { return *this = *this / k; }

bool complex::operator == (const complex &t) const { return EQUAL(real, t.real) && EQUAL(imag, t.imag); }
complex& complex::operator = (const complex &t) { 
    if (this != &t) {
        this->real = t.real;
        this->imag = t.imag;
    }
    return *this;
}

complex& complex::operator = (double k) {
    this->real = k;
    this->imag = 0;
    return *this;
}

complex complex::conjugate() const { return complex(real, -imag); }
double complex::abs2() const { return real * real + imag * imag; }
double complex::abs()  const { return sqrt(abs2()); }
