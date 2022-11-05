#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y;
};
//写一个struct用于表定义"<"号
struct MYSTRcmp{
    bool operator() (MYSTR a,MYSTR b) const{
        //如果a<b，返回true
        if(a.x<b.x) return true;
        if(a.x>b.x) return false;
        return a.y<b.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //改用MYSTRcmp这个struct当作"<"来比较
    set<MYSTR,MYSTRcmp> test1;
    priority_queue<MYSTR,vector<MYSTR>,MYSTRcmp> test2;
    
    
    
    system("pause");
    return 0;
}
