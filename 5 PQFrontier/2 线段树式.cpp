#include <bits/stdc++.h>
using namespace std;
template<typename T> class PQFrontier{
private:
    struct Entry{
        int id;
        T val;
        bool exist;
        bool operator<(const Entry& a) const{
            if(exist&&!a.exist) return true;
            if(!exist&&a.exist) return false;
            return val<a.val;
        }
    };
    int N,len;
    vector<Entry> tree;
    void UpdateUp(int x){
        x/=2;
        while(x>0){
            tree[x]=min(tree[2*x],tree[2*x+1]);
            x/=2;
        }
    }
    
public:
    PQFrontier(int Nsize):
        N(Nsize)
    {
        int i;
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        for(i=len;i<=2*len-1;i++) tree[i]={i-len+1,0,false};
        for(i=len-1;i>=1;i--) tree[i]=min(tree[i*2],tree[i*2+1]);
    }
    bool Empty(){
        return !tree[1].exist;
    }
    int MinID(){
        return tree[1].id;
    }
    T ValueOf(int x){
        if(tree[x+len-1].exist) return tree[x+len-1].val;
        return -1;
    }
    void DeleteMin(){
        int x=tree[1].id+len-1;
        tree[x].exist=false;
        UpdateUp(x);
    }
    bool DecreaseVal(int x,T v){
        x+=len-1;
        if(tree[x].exist&&tree[x].val<=v) return false;
        tree[x]={x-len+1,v,true};
        UpdateUp(x);
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
