#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义T multiset
    //multiset<T> a;
    
    //返回a内有多少元素，重复元素会被重复算
    //int a.size();
    
    //返回a是否为空(size为0)
    //bool a.empty();
    
    //插入元素x
    //void a.insert(T x);
    
    //删除元素x，如果其不在集合内则无事发生，有多个则全部删除
    //返回被删除元素个数
    //int a.erase(T x);
    
    //删除迭代器it指向的元素，要求迭代器合法
    //只删除一个元素，相同值的元素不会被删
    //void a.erase(multiset<T>::iterator it);
    
    //删除[it1,it2)区间的元素，要求迭代器合法
    //void a.erase(multiset<T>::iterator it1,multiset<T>::iterator it2);
    
    //返回元素x在集合中的个数
    //int a.count(T x);
    
    //清空集合
    //void a.clear();
    
    //返回指向大于等于x的最小元素的迭代器，不存在则返回a.end()
    //multiset<T>::iterator a.lower_bound(x);
    
    //返回指向大于x的最小元素的迭代器，不存在则返回a.end()
    //multiset<T>::iterator a.upper_bound(x);
    
    //对于迭代器it，这样访问它指向的元素
    //*it
    
    //让迭代器it指向下一个元素(更大的)，重复元素算不同的
    //it++;
    
    //让迭代器it指向上一个元素(更小的)，重复元素算不同的
    //it--;
    
    
    
    //也可以像priority_queue一样定义顺序
    //默认等于这样(大数优先)
    //set<double,less<double> > a;
    //反过来(小数优先)
    //set<double,greater<double> > a;
    //换言之，lower_bound和upper_bound是找优先级大于(等于)x中优先度最低的
    
    
    
    system("pause");
    return 0;
}
