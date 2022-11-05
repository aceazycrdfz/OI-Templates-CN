#include <bits/stdc++.h>
using namespace std;
class TopoSort{
    private:
        struct EDGE{
            int u,v;
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
            e[M]={u,v,n[u].fir};
            n[u].fir=&e[M];
        }
        void RunTopo(){
            int i;
            DAG=true;
            for(i=1;i<=N;i++) n[i].mark=0;
            ans.clear();
            for(i=1;i<=N;i++) DFS(i);
            ans.push_back(0);
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
    
    int N,M,i,u,v;
    cin>>N>>M;
    TopoSort G(N,M);
    for(i=1;i<=M;i++){
        cin>>u>>v;
        G.AddEdge(u,v);
    }
    G.RunTopo();
    if(G.isDAG()){
        cout<<"是有向无环图！"<<endl;
        vector<int> ans=G.Result();
        for(i=1;i<=N;i++) cout<<ans[i]<<' ';
        cout<<endl;
    }
    else cout<<"不是有向无环图！"<<endl;
    
    
    
    
    system("pause");
    return 0;
}
