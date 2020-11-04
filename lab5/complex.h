#ifndef COMPLEX_H_
#define COMPLEX_H_

class complex {
public:
  double real, imag;

  complex(double real = 0, double imag = 0);

  complex operator + (const complex &t) const;
  complex operator - (const complex &t) const;
  complex operator * (const complex &t) const;
  complex operator * (double k)   const;
  complex operator / (const complex &t) const;
  complex operator / (double k) const;

  complex& operator += (const complex &t);
  complex& operator -= (const complex &t);
  complex& operator *= (const complex &t);
  complex& operator *= (double k);
  complex& operator /= (const complex &t);
  complex& operator /= (double k);

  bool operator == (const complex &t) const;
  complex& operator = (const complex &t);
  complex& operator = (double k);

  complex conjugate() const;
  double abs2() const;
  double abs()  const;
};
#endif // COMPLEX_H_