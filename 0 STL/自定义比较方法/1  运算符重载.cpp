#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y,z;
    //重载<运算符
    bool operator<(const MYSTR& a) const{
        //如果我比a小，返回true
        if(x<a.x) return true;
        if(x>a.x) return false;
        return y<a.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //重载<运算符后可以在代码里直接用"<"比较
    //min和max也可以直接用了
    //还不能用">","==","<="等其它运算符，要另外重载
    
    
    
    system("pause");
    return 0;
}
