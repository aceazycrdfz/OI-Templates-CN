#include <bits/stdc++.h>
using namespace std;
class Sieve{
private:
    int N;
    vector<bool> isP;
    vector<int> plist;
    inline int NumToID(int x){
        return x/2-1;
    }

public:
    Sieve(int Nsize):
        N(Nsize),isP(NumToID(N)+1,true),plist({0,2})
    {
        int i,j;
        for(i=3;i<=N;i+=2){
            if(!isP[NumToID(i)]) continue;
            plist.push_back(i);
            for(j=i;1ll*i*j<=N;j+=2) isP[NumToID(1ll*i*j)]=false;
        }
    }
    bool IsPrime(int x){
        if(x<2) return false;
        if(x==2) return true;
        if(x%2==0) return false;
        return isP[NumToID(x)]; 
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
    //�ð���ɸ��������ֱ��N(����)
    //Sieve s(int N);
    
    //ѯ��x�Ƿ�Ϊ������xС�ڵ���N
    //bool s.IsPrime(int x);
    
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
