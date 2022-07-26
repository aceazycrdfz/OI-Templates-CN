#include <bits/stdc++.h>
using namespace std;
template<typename T> class SPFA{
    private:
        struct EDGE{
            int u,v;
            T len;
            EDGE *nex;
        };
        struct NODE{
            EDGE *fir;
        };
        int N,M,s;
        bool negcyc;
        vector<NODE> n;
        vector<EDGE> e;
        vector<T> dis;
        vector<bool> reached,inQ;
        vector<int> prev,inQcnt;
        list<int> Q;
        void QPush(int id){
            if(Q.empty()) Q.push_back(id);
            else{
                if(dis[Q.front()]<dis[Q.back()]) Q.push_front(id);
                else Q.push_back(id);
            }
        }
        int QPop(){
            int ret;
            if(dis[Q.front()]<=dis[Q.back()]){
                ret=Q.front();
                Q.pop_front();
            }
            else{
                ret=Q.back();
                Q.pop_back();
            }
            return ret;
        }
    
    public:
        SPFA(int Nsize,int Esize){
            N=Nsize;
            M=0;
            n.resize(Nsize+1);
            e.resize(Esize+1);
            dis.resize(N+1);
            reached.resize(N+1);
            prev.resize(N+1);
            inQ.resize(N+1);
            inQcnt.resize(N+1);
        }
        void AddEdge(int u,int v,T len){
            M++;
            e[M].u=u;
            e[M].v=v;
            e[M].len=len;
            e[M].nex=n[u].fir;
            n[u].fir=&e[M];
        }
        void RunSSSP(int start){
            int x;
            s=start;
            reached[s]=true;
            dis[s]=0;
            QPush(s);
            inQ[s]=true;
            negcyc=false;
            while(!Q.empty()){
                x=QPop();
                inQ[x]=false;
                EDGE *ei=n[x].fir;
                while(ei!=NULL){
                    if(!reached[ei->v]||dis[x]+ei->len<dis[ei->v]){
                        reached[ei->v]=true;
                        dis[ei->v]=dis[x]+ei->len;
                        prev[ei->v]=x;
                        if(!inQ[ei->v]){
                            QPush(ei->v);
                            inQ[ei->v]=true;
                            inQcnt[ei->v]++;
                            if(inQcnt[ei->v]==N){
                                negcyc=true;
                                return ;
                            }
                        }
                    }
                    ei=ei->nex;
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
            path.push_back(0);
            reverse(path.begin(),path.end());
            return path;
        }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //初始化一个有N个点和M条边的图，T是int、double等(边长的类)
    //M可以过大，不能过小
    //SPFA<T> a(int N,int M);
    
    //以下点的编号视为1,...,N
    
    //增加一条从u指向v，长度为len的有向边
    //void a.AddEdge(int u,int v,T len);
    
    //以start为起点计算单源最短路径
    //void a.RunSSSP(int start); 
    
    //询问起点是否能进入负权回路
    //bool a.NegCycle();
    
    //询问起点是否能到达id号点 
    //bool a.Reachable(id);
    
    //询问起点到id号点的最短路长
    //要求起点不能进入负权回路且能到达id号点
    //T a.DisTo(int id);
     
    //返回一个从起点到id号点的最短路径 
    //要求起点不能进入负权回路且能到达id号点
    //vector<int> a.PathTo(int id);
    //返回数组第一个数(下标1)是start，最后一个是id 
    
    
    
    system("pause");
    return 0;
}
