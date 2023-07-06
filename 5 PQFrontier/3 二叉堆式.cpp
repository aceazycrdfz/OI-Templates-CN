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
    
    //��ΪDijkstra��Prim�ع���Frontierģ��
    //��ʼ��N���㣬T��int��double��(�߳�����)
    //PQFrontier<T> F(int N);
    
    
    //���µ�ı�Ŷ���1,...,N
    
    //ѯ��F�Ƿ�Ϊ��
    //bool F.Empty();
    
    //ѯ�ʴ�ʱ��Ӧֵ��С�ĵ�(������֮һ)��Ҫ��F�ǿ�
    //int F.MinID();
    
    //ѯ�ʵ�x��Ӧ��ֵ��Ҫ��x��F��
    //T F.ValueOf(int x);
    
    //ɾ����Ӧֵ��С�ĵ�(MinID�Ǹ�)��Ҫ��F�ǿ�
    //void F.DeleteMin();
    
    //���xĿǰ��F����ֵ����v����ᱻ��Ϊv������true
    //���xĿǰ��F����ֵС�ڵ���v�������·���������false
    //���xĿǰ����F�ڣ�������F��ֵ��Ϊv������true
    //bool DecreaseVal(int x,T v);
    
    
    
    system("pause");
    return 0;
}
