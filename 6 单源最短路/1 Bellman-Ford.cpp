#include <bits/stdc++.h>
using namespace std;
template<typename T> class BellmanFord{
private:
    struct EDGE{
        int u,v;
        T len;
    };
    int N,M,s;
    bool negcyc;
    vector<EDGE> e;
    vector<T> dis;
    vector<bool> reached;
    vector<int> prev;

public:
    BellmanFord(int Nsize,int Esize):
        N(Nsize),M(0),e(Esize+1),
        dis(N+1),reached(N+1),prev(N+1){}
    void AddEdge(int u,int v,T len){
        M++;
        e[M]={u,v,len};
    }
    void RunSSSP(int start){
        int i,j;
        bool updated;
        s=start;
        reached[s]=true;
        dis[s]=0;
        negcyc=false;
        for(i=1;i<=N-1;i++){
            updated=false;
            for(j=1;j<=M;j++){
                if(!reached[e[j].u]) continue;
                if(!reached[e[j].v]||dis[e[j].u]+e[j].len<dis[e[j].v]){
                    reached[e[j].v]=true;
                    dis[e[j].v]=dis[e[j].u]+e[j].len;
                    prev[e[j].v]=e[j].u;
                    updated=true;
                }
            }
            if(!updated) break;
        }
        for(j=1;j<=M;j++){
            if(!reached[e[j].u]) continue;
            if(!reached[e[j].v]||dis[e[j].u]+e[j].len<dis[e[j].v]){
                negcyc=true;
                break;
            }
        }
    }
    bool NegCycle(){
        return negcyc;
    }
    bool Reachable(int id){
        return reached[id];
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
    //BellmanFord<T> G(int N,int M);
    
    
    //以下点的编号都是1,...,N
    
    //增加一条从u指向v，长度为len的有向边
    //void G.AddEdge(int u,int v,T len);
    
    //以start为起点计算单源最短路径
    //void G.RunSSSP(int start); 
    
    //询问起点是否能进入负权回路
    //只有此结果为false以下其它询问结果才有意义
    //bool G.NegCycle();
    
    //询问起点是否能到达id号点
    //bool G.Reachable(int id);
    
    //询问起点到id号点的最短路长
    //要求起点不能进入负权回路且能到达id号点
    //T G.DisTo(int id);
     
    //返回一个从起点到id号点的最短路径
    //要求起点不能进入负权回路且能到达id号点
    //vector<int> G.PathTo(int id);
    //返回数组第一个数(下标0)是start，最后一个是id
    
    
    
    system("pause");
    return 0;
}
