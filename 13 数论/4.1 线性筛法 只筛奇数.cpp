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
    
    //����һ��Sieve object
    //������ɸ��������ֱ��N(����)
    //Sieve s(int N);
    
    //ѯ��x�Ƿ�Ϊ������xС�ڵ���N
    //bool s.IsPrime(int x);
    
    //ѯ��x����С�����ӣ��ɷ������������������ֽ�
    //int s.MinPrimeFactor(int x);
    
    //ѯ��N(����)���¹��ж�������
    //int s.PrimeCount();
    
    //ѯ�ʴ�С�����k��������k���ܳ���PrimeCount
    //int s.KthPrime(int k);
    
    //��isP���ļ�outfile��ֱ��upb(����)
    //isP������bool����ʾÿ�����Ƿ�������
    //outfile���Ѵ�����Ḳ��
    //void s.PrintB(int upb,string outfile);
    
    //��plist���ļ�outfile��ֱ��upb(����)
    //plist������±�1��ʼ��С�����г�����
    //outfile���Ѵ�����Ḳ��
    //void s.PrintL(int upb,string outfile);
    
    
    
    system("pause");
    return 0;
}
