#include <bits/stdc++.h>
using namespace std;
long long gcd(long long x,long long y){
    long long z=x%y;
    if(z==0) return y;
    return gcd(y,z);
}
long long lcm(long long x,long long y){
    return x*y/gcd(x,y);
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��x��y�����Լ��
    //long long gcd(long long x,long long y);
    //Ҳ���Ե�������� 
    //long long __gcd<long long>(long long x,long long y);
    
    //��x��y����С������
    //long long lcm(long long x,long long y);
    
    
    system("pause");
    return 0;
}
