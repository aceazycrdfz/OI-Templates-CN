#include <bits/stdc++.h>
using namespace std;
template<typename T> class PQFrontier{
private:
    struct Entry{
        int id;
        T val;
    };
    int N,tTop;
    vector<Entry> heap;
    vector<int> loc;
    void Swap(int x,int y){
        swap(loc[heap[x].id],loc[heap[y].id]);
        swap(heap[x],heap[y]);
    }
    void UpdateUp(int x){
        while(x>1){
            if(heap[x].val>=heap[x/2].val) break;
            Swap(x,x/2);
            x/=2;
        }
    }
    void UpdateDown(int x){
        while(2*x<=tTop){
            if(2*x+1>tTop||heap[2*x].val<=heap[2*x+1].val){
                if(heap[2*x].val>=heap[x].val) break;
                Swap(x,x*2);
                x*=2;
            }
            else{
                if(heap[2*x+1].val>=heap[x].val) break;
                Swap(x,x*2+1);
                x=x*2+1;
            }
        }
    }
    
public:
    PQFrontier(int Nsize):
        N(Nsize),tTop(0),
        heap(N+1),loc(N+1,-1){}
    bool Empty(){
        return tTop==0;
    }
    int MinID(){
        return heap[1].id;
    }
    T ValueOf(int x){
        if(loc[x]>0) return heap[loc[x]].val; 
        return -1;
    }
    void DeleteMin(){
        if(tTop==1){
            loc[heap[1].id]=-1;
            tTop=0;
        }
        else{
            Swap(1,tTop);
            loc[heap[tTop].id]=-1;
            tTop--;
            UpdateDown(1);
        }
    }
    bool DecreaseVal(int x,T v){
        if(loc[x]>0){
            if(heap[loc[x]].val<=v) return false;
            heap[loc[x]].val=v;
            UpdateUp(loc[x]);
        }
        else{
            tTop++;
            heap[tTop]={x,v};
            loc[x]=tTop;
            UpdateUp(loc[x]);
        }
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
