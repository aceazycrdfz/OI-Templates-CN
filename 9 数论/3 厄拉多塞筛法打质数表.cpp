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
    
    //用筛法判质数，直到upb(包含)
    //存在bool数组isP里，true表示是质数
    //void Sieve(int upb);
    
    //把isP存的质数列出来，直到upb(包含)
    //存在int数组plist里，下标从0开始
    //void toList(int upb);
    
    //把isP表打出来到文件prime.out，直到upb(包含)
    //void printB(int upb);
    
    //把plist表打出来到文件prime.out，直到upb(包含)
    //void printL(int upb);
    
    
    
    system("pause");
    return 0;
}
