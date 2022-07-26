#include <bits/stdc++.h>
using namespace std;
typedef complex<double> comp;
const double PI=acos(-1.0);

void fft(vector<comp> &a, int invert=0) {    // Compute the FFT of the polynomial
  int n=a.size(), i, j, len; comp w, u, v;   // whose coefficients are given by
  for(i=1, j=0;i<n;i++) {                    // the elements of a.
    int bit = n/2; for(; j >= bit; bit /= 2) j-=bit;
    j += bit; if(i < j) swap(a[i], a[j]);
  }
  for(len=2;len<=n;len<<=1) {
    double ang=2*PI/len*(invert?-1:1); comp wlen = polar(1.0, ang);
    for(i=0; i<n; i+=len) for(j=0, w=1; j < len/2; j++)
      u=a[i+j], v=a[i+j+len/2]*w, a[i+j]=u+v, a[i+j+len/2]=u-v, w*=wlen;
  }
  if(invert) for(i=0;i<n;i++) a[i]/=n;
}

// Compute the convolution a * b
template<typename T> vector<T> multiply(const vector<T>& a, const vector<T>& b) {
  int i, n;  vector<comp> fa(a.begin(), a.end()), fb(b.begin(), b.end());
  for(n=1;n<2*(int)max(a.size(), b.size());n*=2);
  fa.resize(n), fb.resize(n), fft(fa), fft(fb);
  for(i=0;i<n;i++) fa[i]*=fb[i];
  fft(fa, 1); vector<T> res(n);   // Remove rounding below if T is non-integral
  for(i=0;i<n;i++) res[i]=(T)(fa[i].real()+0.5);
  return res;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    
    system("pause");
    return 0;
}
