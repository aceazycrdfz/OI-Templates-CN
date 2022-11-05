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
