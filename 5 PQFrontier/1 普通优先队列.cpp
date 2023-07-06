#include <bits/stdc++.h>
using namespace std;
template<typename T> class PQFrontier{
private:
    struct Entry{
        int id;
        T val;
        bool operator<(const Entry& a) const{
            return val>a.val;
        }
    };
    int N;
    vector<T> vals;
    priority_queue<Entry> PQ;
    void ClearPopped(){
        while(!PQ.empty()){
            if(vals[PQ.top().id]==PQ.top().val) break;
            PQ.pop();
        }
    }
    
public:
    PQFrontier(int Nsize):
        N(Nsize),vals(N+1,-1){}
    bool Empty(){
        ClearPopped();
        return PQ.empty();
    }
    int MinID(){
        ClearPopped();
        return PQ.top().id;
    }
    T ValueOf(int x){
        return vals[x];
    }
    void DeleteMin(){
        ClearPopped();
        PQ.pop();
    }
    bool DecreaseVal(int x,T v){
        if(vals[x]!=-1&&vals[x]<=v) return false;
        vals[x]=v;
        PQ.push({x,v});
        return true;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //此为Dijkstra和Prim特供的Frontier模板
    //初始化N个点，T是int、double等(边长的类)
    //PQFrontier<T> F(int N);
    
    
    //以下点的编号都是1,...,N
    
    //询问F是否为空
    //bool F.Empty();
    
    //询问此时对应值最小的点(可能是之一)，要求F非空
    //int F.MinID();
    
    //询问点x对应的值，要求x在F里
    //T F.ValueOf(int x);
    
    //删除对应值最小的点(MinID那个)，要求F非空
    //void F.DeleteMin();
    
    //如果x目前在F内且值大于v，则会被降为v，返回true
    //如果x目前在F内且值小于等于v，则无事发生，返回false
    //如果x目前不在F内，则会加入F且值设为v，返回true
    //bool DecreaseVal(int x,T v);
    
    
    
    system("pause");
    return 0;
}
