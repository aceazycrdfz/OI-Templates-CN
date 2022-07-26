#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<int> vi;
typedef ll big;  // use __int128 when required and available

ll gcd(ll a, ll b, ll& x, ll& y) {
	if (b == 0) { y = 0; x = (a < 0) ? -1 : 1; return (a < 0) ? -a : a; }
	else { ll g = gcd(b, a%b, y, x); y -= a/b*x; return g; }
}

ll inv(ll a, ll m) { ll x, y; gcd(m,a,x,y); return ((y % m) + m) % m; }

// Integer convolution mod m using number theoretic transform.
// m = modulo, r = a primitive root, ord = order of the root
// (Must be a power of two). The length of the given input
// vectors must not exceed n = ord.   Complexity: O(n log(n))
//
// Usable coefficients::
//   m                    | r           | ord       | __int128 required
//------------------------|-------------|-----------|--------------------
//  7340033               | 5           | 1 << 20   | No
//  469762049             | 13          | 1 << 25   | No
//  998244353             | 31          | 1 << 23   | No
//  1107296257            | 8           | 1 << 24   | No
//  10000093151233        | 366508      | 1 << 26   | Yes
//  1000000523862017      | 2127080     | 1 << 26   | Yes
//  1000000000949747713   | 465958852   | 1 << 26   | Yes
//
// In general, you may use mod = c * 2^k + 1 which has a primitive
// root of order 2^k, then use number theory to find a generator.
template<typename T> struct convolution {
	const T m, r, ord;
	T mult(T x, T y) { return big(x) * y % m; }
	void ntt(vector<T> & a, int invert = 0) {
		int n = (int)a.size(); T ninv = inv(n, m), rinv = inv(r, m);  // Modular inverses
		for (int i=1, j=0; i<n; ++i) {
			int bit = n >> 1;	for (; j>=bit; bit>>=1)	j -= bit;
			j += bit;	if (i < j) swap (a[i], a[j]);
		}
		for (int len=2; len<=n; len<<=1) {
			T wlen = invert ? rinv : r;
			for (int i=len; i<ord; i<<=1) wlen = mult(wlen, wlen);
			for (int i=0; i<n; i+=len) {
				T w = 1;
				for (int j=0; j<len/2; ++j) {
					T u = a[i+j],  v = mult(a[i+j+len/2], w);
					a[i+j] = u + v < m ? u + v : u + v - m;
					a[i+j+len/2] = u - v >= 0 ? u - v : u - v + m;
					w = mult(w, wlen);
				}
			}
		}
		if (invert) for (int i=0; i<n; ++i)	a[i] = mult(a[i], ninv);
	}
	// Compute the convolution a * b -- Complexity: O(n log(n))
	vector<T> multiply(vector<T>& a, vector<T>& b) {
		vector<T> fa(a.begin(), a.end()), fb(b.begin(), b.end());
		int n = 1;  while (n < 2 * (int)max(a.size(), b.size())) n*=2;
		fa.resize(n), fb.resize(n);	ntt(fa), ntt(fb);
		for(int i=0;i<n;i++) fa[i] = mult(fa[i], fb[i]);
		ntt(fa, 1);	fa.resize(n);
		return fa;
	}
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    
    system("pause");
    return 0;
}
