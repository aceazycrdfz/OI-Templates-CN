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
