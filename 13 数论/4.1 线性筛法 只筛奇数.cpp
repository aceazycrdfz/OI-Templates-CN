#include <bits/stdc++.h>
using namespace std;
class Sieve{
private:
    int N;
    vector<int> minpf,plist;
    inline int NumToID(int x){
        return x/2-1;
    }

public:
    Sieve(int Nsize):
        N(Nsize),minpf(NumToID(N)+1),plist({0,2})
    {
        int i,j,id;
        for(i=3;i<=N;i+=2){
            id=NumToID(i);
            if(minpf[id]==0){
                minpf[id]=i;
                plist.push_back(i);
            }
            for(j=2;1ll*plist[j]*i<=N;j++){
                minpf[NumToID(1ll*plist[j]*i)]=plist[j];
                if(plist[j]==minpf[id]) break;
            }
        }
    }
    bool IsPrime(int x){
        if(x<2) return false;
        if(x==2) return true;
        if(x%2==0) return false;
        return minpf[NumToID(x)]==x;
    }
    int MinPrimeFactor(int x){
        if(x<2) return 1;
        if(x%2==0) return 2;
        return minpf[NumToID(x)];
    }
    int PrimeCount(){
        return plist.size()-1;
    }
    int KthPrime(int k){
        return plist[k];
    }
    void PrintB(int upb,string outfile){
        ofstream fout(outfile);
        int i;
        fout<<"vector<bool> isP={";
        for(i=0;i<=upb;i++){
            if(i>0) fout<<',';
            fout<<IsPrime(i);
        }
        fout<<"};"<<endl;
    }
    void PrintL(int upb,string outfile){
        ofstream fout(outfile);
        int i;
        fout<<"vector<int> plist={";
        for(i=0;i<plist.size();i++){
            if(plist[i]>upb) break;
            if(i>0) fout<<',';
            fout<<KthPrime(i);
        }
        fout<<"};"<<endl;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义一个Sieve object
    //用线性筛法判质数直到N(包含)
    //Sieve s(int N);
    
    //询问x是否为质数，x小于等于N
    //bool s.IsPrime(int x);
    
    //询问x的最小质因子，可反复调用用于质因数分解
    //int s.MinPrimeFactor(int x);
    
    //询问N(包含)以下共有多少质数
    //int s.PrimeCount();
    
    //询问从小到大第k个质数，k不能超过PrimeCount
    //int s.KthPrime(int k);
    
    //打isP表到文件outfile，直到upb(包含)
    //isP数组是bool，表示每个数是否是质数
    //outfile若已存在则会覆盖
    //void s.PrintB(int upb,string outfile);
    
    //打plist表到文件outfile，直到upb(包含)
    //plist数组从下标1开始从小到大列出质数
    //outfile若已存在则会覆盖
    //void s.PrintL(int upb,string outfile);
    
    
    
    system("pause");
    return 0;
}
