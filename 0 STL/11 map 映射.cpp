#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义map
    //map<Key,T> a;
    
    //直接这样插入、查询、或修改键k对应的值
    //a[k]=x;
    //这样初次访问即使不赋值，也会视为插入且赋值为默认值0 
    
    //返回a内有多少元素
    //int a.size();
    
    //返回a是否为空(size为0)
    //bool a.empty();
    
    //用一个pair插入，如果该键已经存在则无事发生
    //void a.insert(pair<Key,T>);
    
    //删除键k的元素，如果其不在集合内则无事发生
    //返回被删除元素个数(0或1) 
    //int a.erase(Key k);
    
    //删除迭代器it指向的元素，要求迭代器合法
    //void a.erase(set<T>::iterator it);
    
    //删除[it1,it2)区间的元素，要求迭代器合法
    //void a.erase(set<T>::iterator it1,set<T>::iterator it2);
    
    //返回键k在map中的个数(0或1)
    //int a.count(Key k);
    
    //清空map
    //void a.clear(); 
    
    //返回指向键大于等于k的最小元素的迭代器，不存在则返回a.end()
    //map<Key,T>::iterator a.lower_bound(k);
    
    //返回指向键大于k的最小元素的迭代器，不存在则返回a.end()
    //map<Key,T>::iterator a.upper_bound(k);
    
    //对于迭代器it，这样访问它指向的元素，它是个pair
    //*it
    //这样访问它的键和值
    //it->first
    //it->second
    
    //排在it下一个的迭代器(更大的)
    //next(it,1);
    //让迭代器it指向下一个元素(更大的)
    //it++;
    
    //排在it上一个的迭代器(更小的)
    //prev(it,1);
    //让迭代器it指向上一个元素(更小的)
    //it--;
    
    
    
    //也可以像priority_queue一样定义顺序，比的是键的顺序 
    //默认等于这样(大数优先)
    //map<double,int,greater<double> > a;
    //反过来(小数优先)
    //map<double,int,greater<double> > a;
    //换言之，lower_bound和upper_bound是找优先级大于(等于)x中优先度最低的
    
    
    //和set类似，也有multimap,unordered_map,unordered_multimap
    //如果是multi就不能如上访问键对应的值了
    //multi用键值erase会删掉所有有这个键的映射
    //而用迭代器erase只会删那一个
    
    
    system("pause");
    return 0;
}
