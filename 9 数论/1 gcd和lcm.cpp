#include <bits/stdc++.h>
using namespace std;
long long gcd(long long x,long long y){
    if(x%y==0) return y;
    return gcd(y,x%y);
}
long long lcm(long long x,long long y){
    return x*y/gcm(x,y);
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��x��y�����Լ��
    //long long gcd(long long x,long long y);
    
    //��x��y����С������
    //long long lcm(long long x,long long y);
    
    
    
    system("pause");
    return 0;
}
