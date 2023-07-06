#include <bits/stdc++.h>
using namespace std;
class TopoSort{
private:
    struct EDGE{
        int v;
        EDGE *nex;
    };
    struct NODE{
        int mark;
        EDGE *fir;
    };
    int N,M;
    bool DAG;
    vector<NODE> n;
    vector<EDGE> e;
    vector<int> ans;
    void DFS(int id){
        if(n[id].mark==-1){
            DAG=false;
            return ;
        }
        if(n[id].mark==1) return ;
        n[id].mark=-1;
        EDGE *ei=n[id].fir;
        while(ei!=NULL){
            DFS(ei->v);
            ei=ei->nex;
        }
        ans.push_back(id);
        n[id].mark=1;
    }
    
public:
    TopoSort(int Nsize,int Esize):
        N(Nsize),M(0),n(N+1),e(Esize+1){}
    void AddEdge(int u,int v){
        M++;
        e[M]={v,n[u].fir};
        n[u].fir=&e[M];
    }
    void RunTopo(){
        int i;
        DAG=true;
        for(i=1;i<=N;i++) n[i].mark=0;
        ans.clear();
        for(i=1;i<=N;i++) DFS(i);
        reverse(ans.begin(),ans.end());
    }
    bool isDAG(){
        return DAG;
    }
    vector<int> Result(){
        return ans;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //初始化一个有N个点和M条边的图
    //M可以过大，不能过小
    //TopoSort a(int N,int M);
    
    //以下点的编号视为1,...,N
    
    //增加一条从u指向v的有向边
    //void a.AddEdge(int u,int v);
    
    //计算拓扑排序
    //void a.RunTopo();
    
    //询问是否为有向无环图 
    //bool a.isDAG();
    
    //返回一个拓扑排序
    //要求是有向无环图
    //vector<int> a.Result();
    //返回数组下标从0开始 
    
    
    
    system("pause");
    return 0;
}
