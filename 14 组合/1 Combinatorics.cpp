#include <bits/stdc++.h>
using namespace std;
class Combo{
private:
    long long p;
    vector<long long> f,invf,subf;
    long long invfact(long long x){
        if(x>=invf.size()) invf.resize(x+1,-1);
        if(invf[x]!=-1) return invf[x];
        return invf[x]=inv(fact(x));
    }
    
public:
    Combo(long long P):
        p(P),f({1,1}),invf({1,1}),subf({1,0}){}
    long long exp(long long x,long long y){
        long long res=1;
        x%=p;
        y%=p-1;
        while(y>0){
            if(y%2==1) res=res*x%p;
            x=x*x%p;
            y/=2;
        }
        return res;
    }
    long long inv(long long x){
        return exp(x,p-2);
    }
    long long fact(long long x){
        if(x<f.size()) return f[x];
        long long i,oldsize=f.size();
        f.resize(x+1);
        for(i=oldsize;i<=x;i++) f[i]=f[i-1]*i%p;
        return f[x];
    }
    long long binom(long long n,long long k){
        return (fact(n)*invfact(n-k)%p)*invfact(k)%p;
    }
    long long subfact(long long x){
        if(x<subf.size()) return subf[x];
        long long i,oldsize=subf.size();
        subf.resize(x+1);
        for(i=oldsize;i<=x;i++) subf[i]=(i-1)*((subf[i-1]+subf[i-2])%p)%p;
        return subf[x];
    }
    long long Catalan(long long x){
        return binom(2*x,x)*inv(x+1)%p;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����һ��Combo object
    //�����������м��㶼��mod P��P����������
    //Combo c(long long P);
    
    //����x^y
    //long long c.exp(long long x,long long y);
    
    //����x�ĳ˷���Ԫmultiplicative inverse
    //long long c.inv(long long x);
    
    //����x�Ľ׳�: x!
    //long long c.fact(long long x);
    
    //�������ʽϵ��n choose k
    //long long c.binom(long long n,long long k);
    
    //����x��subfactorial: !x
    //!x��ʾ�ж��ٸ�1��x������ʹ��ÿ�����������Լ���ԭλ��
    //long long c.subfact(long long x);
    
    //�����x����������
    //��x������������ʾx���������ųɶ��ٸ���ͬ�ĺϷ���������
    //����x=2������: (())��()()
    //long long c.Catalan(long long x);
    
    
    
    system("pause");
    return 0;
}
