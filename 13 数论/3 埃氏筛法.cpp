#include <bits/stdc++.h>
using namespace std;
class Sieve{
private:
    int N;
    vector<bool> isP;
    vector<int> plist;

public:
    Sieve(int Nsize):
        N(Nsize),isP(max(N+1,2),true),plist(1)
    {
        int i,j;
        isP[0]=false;
        isP[1]=false;
        for(i=2;i<=N;i++){
            if(!isP[i]) continue;
            plist.push_back(i);
            for(j=i;1ll*i*j<=N;j++) isP[1ll*i*j]=false;
        }
    }
    bool IsPrime(int x){
        return isP[x]; 
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
    //用埃氏筛法判质数直到N(包含)
    //Sieve s(int N);
    
    //询问x是否为质数，x小于等于N
    //bool s.IsPrime(int x);
    
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
