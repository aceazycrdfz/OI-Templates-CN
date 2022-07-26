#include <bits/stdc++.h>
using namespace std;
ofstream fout("prime.out");
vector<bool> isP;
vector<int> plist;
void Sieve(int upb){
    int i,j;
    isP.resize(max(upb+1,2),true);
    isP[0]=false;
    isP[1]=false;
    for(i=2;i*i<=upb;i++){
        if(!isP[i]) continue;
        for(j=2;i*j<=upb;j++) isP[i*j]=false;
    }
}
void toList(int upb){
    int i;
    plist.clear();
    for(i=2;i<=upb;i++) if(isP[i]) plist.push_back(i);
}
void printB(int upb){
    int i;
    fout<<"vector<int> isP={";
    for(i=0;i<=upb;i++){
        if(i>0) fout<<',';
        fout<<isP[i];
    }
    fout<<"};"<<endl;
}
void printL(int upb){
    int i;
    fout<<"vector<bool> plist={";
    for(i=0;i<plist.size();i++){
        if(plist[i]>upb) break;
        if(i>0) fout<<',';
        fout<<plist[i];
    }
    fout<<"};"<<endl;
}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��ɸ����������ֱ��upb(����)
    //����bool����isP�true��ʾ������
    //void Sieve(int upb);
    
    //��isP��������г�����ֱ��upb(����)
    //����int����plist��±��0��ʼ
    //void toList(int upb);
    
    //��isP���������ļ�prime.out��ֱ��upb(����)
    //void printB(int upb);
    
    //��plist���������ļ�prime.out��ֱ��upb(����)
    //void printL(int upb);
    
    
    
    system("pause");
    return 0;
}
