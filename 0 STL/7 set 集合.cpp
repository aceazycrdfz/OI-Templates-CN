#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义T set
    //set<T> a;
    
    //返回a内有多少元素，无视重复元素
    //int a.size();
    
    //返回a是否为空(size为0)
    //bool a.empty();
    
    //插入元素x，如果其已经在集合内则无事发生
    //void a.insert(T x);
    
    //删除元素x，如果其不在集合内则无事发生
    //返回被删除元素个数(0或1) 
    //int a.erase(T x);
    
    //删除迭代器it指向的元素，要求迭代器合法
    //void a.erase(set<T>::iterator it);
    
    //删除[it1,it2)区间的元素，要求迭代器合法
    //void a.erase(set<T>::iterator it1,set<T>::iterator it2);
    
    //返回元素x在集合中的个数(0或1)
    //int a.count(T x);
    
    //清空集合
    //void a.clear(); 
    
    //返回指向大于等于x的最小元素的迭代器，不存在则返回a.end()
    //set<T>::iterator a.lower_bound(x);
    
    //返回指向大于x的最小元素的迭代器，不存在则返回a.end()
    //set<T>::iterator a.upper_bound(x);
    
    //对于迭代器it，这样访问它指向的元素
    //*it
    
    //排在it下一个的迭代器(更大的)
    //next(it,1);
    //让迭代器it指向下一个元素(更大的)
    //it++;
    
    //排在it上一个的迭代器(更小的)
    //prev(it,1);
    //让迭代器it指向上一个元素(更小的)
    //it--;
    
    
    //默认等于这样，从小到大
    //set<double,less<double> > a;
    //反过来，从大到小
    //set<double,greater<double> > a;
    //换言之，lower_bound和upper_bound是找大于(等于)x中最小的
    
    //也可以自定义比较方法，参考自定义比较方法目录
    //互不小于的元素会被判定为相同
    
    
    
    system("pause");
    return 0;
}
