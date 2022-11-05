#include <bits/stdc++.h>
using namespace std;
template<typename T> class Dijkstra{
private:
    struct EDGE{
        int v;
        T len;
        EDGE *nex;
    };
    struct NODE{
        EDGE *fir;
    };
    int N,M,s;
    vector<NODE> n;
    vector<EDGE> e;
    vector<T> dis;
    vector<int> prev;
    PQFrontier<T> F;
    
public:
    Dijkstra(int Nsize,int Esize):
        N(Nsize),M(0),n(Nsize+1),e(Esize+1),
        dis(N+1,-1),prev(N+1),
        F(N){}
    void AddEdge(int u,int v,T len){
        M++;
        e[M]={v,len,n[u].fir};
        n[u].fir=&e[M];
    }
    void RunSSSP(int start){
        int x;
        s=start;
        dis[s]=0;
        F.DecreaseVal(s,0);
        while(!F.Empty()){
            x=F.MinID();
            dis[x]=F.ValueOf(x);
            F.DeleteMin();
            EDGE *ei=n[x].fir;
            while(ei!=NULL){
                if(dis[ei->v]==-1&&F.DecreaseVal(ei->v,dis[x]+ei->len))
                    prev[ei->v]=x;
                ei=ei->nex;
            }
        }
    }
    bool Reachable(int id){
        return dis[id]>=0;
    }
    T DisTo(int id){
        return dis[id];
    }
    vector<int> PathTo(int id){
        int x=id;
        vector<int> path;
        path.push_back(x);
        while(x!=s){
            x=prev[x];
            path.push_back(x);
        }
        reverse(path.begin(),path.end());
        return path;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��ʼ��һ����N�����M���ߵ�ͼ��T��int��double��(�߳�����)
    //M���Թ��󣬲��ܹ�С
    //Dijkstra<T> G(int N,int M);
    
    
    //���µ�ı�Ŷ���1,...,N
    
    //����һ����uָ��v������Ϊlen������ߣ�len����Ǹ�
    //void G.AddEdge(int u,int v,T len);
    
    //��startΪ�����㵥Դ���·��
    //void G.RunSSSP(int start);
    
    //ѯ������Ƿ��ܵ���id�ŵ�
    //bool G.Reachable(id);
    
    //ѯ����㵽id�ŵ�����·��
    //Ҫ������ܵ���id�ŵ�
    //T G.DisTo(int id);
     
    //����һ������㵽id�ŵ�����·��
    //Ҫ������ܵ���id�ŵ�
    //vector<int> G.PathTo(int id);
    //���������һ����(�±�0)��start�����һ����id
    
    
    
    system("pause");
    return 0;
}
