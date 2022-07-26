#include <bits/stdc++.h>
using namespace std;
bool isPrime(long long x){
    if(x<=1) return false;
    long long i;
    for(i=2;i*i<=x;i++) if(x%i==0) return false;
    return true;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //朴素方法判断x是否为质数，复杂度O(sqrt x) 
    //bool isPrime(long long x);
    
    
    
    system("pause");
    return 0;
}
