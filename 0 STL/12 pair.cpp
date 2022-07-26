#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //pair的类型像这样
    //pair<T1,T2>
    
    //直接定义并初始化这个变量
    //pair<int,double> a(1,2.0);
    //如果不自己初始化，pair变量会自动初始化各部分为0等默认值 
    
    //也可以这样用make_pair弄一个pair值
    //pair<int,double> a=make_pair(1,2.0); 
    
    //也可以这样先定义再赋值
    //pair<int,double> a;
    //用first和second可以访问或修改
    //a.first=1;
    //a.second=2.0; 
    
    //比较运算符<,>,<=,>=默认优先比较第一个值，相等则比较第二个
    
    //同类型可以直接这么赋值
    //a1=a2;
    
    //也可以swap
    //swap(a1,a2);
    //a1.swap(a2);
    
    
    
    system("pause");
    return 0;
}
