#include <bits/stdc++.h>
using namespace std;
template<typename T> class Floyd{
private:
    int N;
    bool negcyc;
    vector<vector<T> > dis;
    vector<vector<bool> > reached;
    vector<vector<int> > prev;
    
public:
    Floyd(int Nsize):
        N(Nsize),
        dis(N+1,vector<T> (N+1)),
        reached(N+1,vector<bool> (N+1)),
        prev(N+1,vector<int> (N+1))
    {
        int i;
        for(i=1;i<=N;i++) reached[i][i]=true;
    }
    void AddEdge(int u,int v,T len){
        dis[u][v]=min(dis[u][v],len);
        reached[u][v]=true;
        prev[u][v]=u;
    }
    void RunAPSP(){
        int i,j,k;
        for(k=1;k<=N;k++){
            for(i=1;i<=N;i++){
                for(j=1;j<=N;j++){
                    if(!reached[i][k]||!reached[k][j]) continue;
                    if(dis[i][k]+dis[k][j]<dis[i][j]){
                        dis[i][j]=dis[i][k]+dis[k][j];
                        reached[i][j]=true;
                        prev[i][j]=prev[k][j];
                    }
                }
            }
        }
        for(i=1;i<=N;i++){
            if(dis[i][i]<0){
                negcyc=true;
                return ;
            }
        }
    }
    bool NegCycle(){
        return negcyc;
    }
    bool Reachable(int u,int v){
        return reached[u][v];
    }
    T DisTo(int u,int v){
        return dis[u][v];
    }
    vector<int> PathTo(int u,int v){
        int x=v;
        vector<int> path;
        path.push_back(x);
        while(x!=u){
            x=prev[u][x];
            path.push_back(x);
        }
        reverse(path.begin(),path.end());
        return path;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //初始化一个有N个点的图，T是int、double等(边长的类)
    //Floyd<T> G(int N);
    
    
    //以下点的编号都是1,...,N
    
    //增加一条从u指向v，长度为len的有向边
    //void G.AddEdge(int u,int v,T len);
    
    //计算多源最短路径
    //void G.RunAPSP();
    
    //询问是否存在负权回路
    //只有此结果为false以下其它询问结果才有意义
    //bool G.NegCycle();
    
    //询问从点u是否能到达点v
    //bool G.Reachable(int u,int v);
    
    //询问点u到点v的最短路长
    //要求点u能到达点v
    //T G.DisTo(int u,int v);
     
    //返回一个从点u到点v的最短路径
    //要求点u能到达点v
    //vector<int> G.PathTo(int u,int v);
    //返回数组第一个数(下标0)是u，最后一个是v
    
    
    
    system("pause");
    return 0;
}
