#include <bits/stdc++.h>
using namespace std;

template<typename T> class DijFrontier{
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
        DijFrontier(int Nsize):
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

template<typename T> class Dijkstra{
    private:
        struct EDGE{
            T len;
            EDGE *nex;
            int u,v;//try this x2
        };
        
        struct NODE{
            EDGE *fir;
        };
        int N,M,s;
        vector<NODE> n;
        vector<EDGE> e;
        vector<T> dis;
        vector<int> prev;
        DijFrontier<T> F;
    
    public:
        Dijkstra(int Nsize,int Esize):
            N(Nsize),M(0),n(Nsize+1),e(Esize+1),
            dis(N+1,-1),prev(N+1),
            F(N){}
        void AddEdge(int u,int v,T len){
            M++;
            e[M]={len,n[u].fir,u,v};
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


long long N,M;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    long long i,a,b,w;
    cin>>N>>M;
    Dijkstra<long long> G(N,2*M);
    for(i=1;i<=M;i++){
        cin>>a>>b>>w;
        G.AddEdge(a,b,w);
        G.AddEdge(b,a,w);
    }
    G.RunSSSP(1);
    if(!G.Reachable(N)) cout<<-1<<endl;
    else{
        vector<int> p=G.PathTo(N);
        for(i=0;i<p.size();i++){
            cout<<p[i]<<' ';
        }
        cout<<endl;
    }
    
    system("pause");
    return 0;
}
