#include <bits/stdc++.h>
using namespace std;
//这个hash用时间作为随机，难以被hack 
struct my_hash{
    static uint64_t splitmix64(uint64_t x){
        x+=0x9e3779b97f4a7c15;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9;
        x=(x^(x>>27))*0x94d049bb133111eb;
        return x^(x>>31);
    }
    
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM=
            chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x+FIXED_RANDOM);
    }
    
    //针对std::pair<int,int>作为主键类型的哈希函数
    size_t operator()(pair<uint64_t,uint64_t> x) const {
        static const uint64_t FIXED_RANDOM=
            chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x.first+FIXED_RANDOM)^
               (splitmix64(x.second+FIXED_RANDOM)>>1);
    }
    
    //或者自己写需要hash的类型
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义T unordered_multiset
    //unordered_multiset<T> a;
    
    //返回a内有多少元素，重复元素会被重复算
    //int a.size();
    
    //返回a是否为空(size为0)
    //bool a.empty();
    
    //插入元素x
    //void a.insert(T x);
    
    //删除元素x，如果其不在集合内则无事发生，有多个则全部删除
    //返回被删除元素个数
    //int a.erase(T x);
    
    //返回元素x在集合中的个数
    //int a.count(T x);
    
    //清空集合
    //void a.clear();
    
    //对于迭代器it，这样访问它指向的元素
    //*it
    
    
    
    //unordered本质是哈希，不能lower_bound等
    //但是可以这样自定义哈希函数
    //unordered_multiset<T,my_hash> a;
    
    
    
    system("pause");
    return 0;
}
