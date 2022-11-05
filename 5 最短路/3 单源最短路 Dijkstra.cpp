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
    
    //初始化一个有N个点和M条边的图，T是int、double等(边长的类)
    //M可以过大，不能过小
    //Dijkstra<T> G(int N,int M);
    
    
    //以下点的编号都是1,...,N
    
    //增加一条从u指向v，长度为len的有向边，len必须非负
    //void G.AddEdge(int u,int v,T len);
    
    //以start为起点计算单源最短路径
    //void G.RunSSSP(int start);
    
    //询问起点是否能到达id号点
    //bool G.Reachable(id);
    
    //询问起点到id号点的最短路长
    //要求起点能到达id号点
    //T G.DisTo(int id);
     
    //返回一个从起点到id号点的最短路径
    //要求起点能到达id号点
    //vector<int> G.PathTo(int id);
    //返回数组第一个数(下标0)是start，最后一个是id
    
    
    
    system("pause");
    return 0;
}
