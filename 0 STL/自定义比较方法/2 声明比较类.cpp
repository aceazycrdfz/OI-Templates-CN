#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y;
};
//дһ��struct���ڱ���"<"��
struct MYSTRcmp{
    bool operator() (MYSTR a,MYSTR b) const{
        //���a<b������true
        if(a.x<b.x) return true;
        if(a.x>b.x) return false;
        return a.y<b.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����MYSTRcmp���struct����"<"���Ƚ�
    set<MYSTR,MYSTRcmp> test1;
    priority_queue<MYSTR,vector<MYSTR>,MYSTRcmp> test2;
    
    
    
    system("pause");
    return 0;
}
