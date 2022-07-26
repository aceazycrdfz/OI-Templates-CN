#include <bits/stdc++.h>
using namespace std;
bool cmp(int x,int y){
    //如果x应该排在y前面，返回true 
    return x<y;
}
struct MYSTR{
    int v1,v2;
    //重载<运算符
    bool operator<(const MYSTR& x) const{
        //如果x应该排在我前面，返回true
        if(v1<x.v1) return true;
        if(v1>x.v1) return false;
        return v2<x.v2;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //给a数组下标1到N排序(默认从小到大) 
    //sort(a+1,a+N+1);
    
    //这样从大到小(如果a是int数组) 
    //sort(a+1,a+N+1,greater<int>());
    
    //用自定义比较函数cmp排序
    //sort(a+1,a+N+1,cmp);
    
    //假如a是某个struct的数组，可以重载<运算符 
    
    
    
    system("pause");
    return 0;
}
