#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y;
    //重载<运算符
    bool operator<(const MYSTR& a) const{
        //如果a应比我优先，返回true
        if(x<a.x) return true;
        if(x>a.x) return false;
        return y<a.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //用MYSTR已重载运算符<来排
    priority_queue<MYSTR> test1;
    
    
    MYSTR e1,e2,e3;
    e1.x=1;
    e1.y=1;
    e2.x=2;
    e2.y=2;
    e3.x=2;
    e3.y=1;
    test1.push(e1);
    test1.push(e2);
    test1.push(e3);
    
    
    while(!test1.empty()){
        MYSTR tp=test1.top();
        cout<<tp.x<<' '<<tp.y<<endl;
        test1.pop();
    }
    
    
    system("pause");
    return 0;
}
