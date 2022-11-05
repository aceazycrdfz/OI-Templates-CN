#include <bits/stdc++.h>
using namespace std;

template<typename T> class DijFrontier{
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
        DijFrontier(int Nsize):
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

template<typename T> class Dijkstra{
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
            e[M]={u,v,len,n[u].fir};
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
    
    int N,M,i,j,u,v,len;
    cin>>N>>M;
    Dijkstra<int> G(N,M);
    for(i=1;i<=M;i++){
        cin>>u>>v>>len;
        G.AddEdge(u,v,len);
    }
    G.RunSSSP(1);
    if(false) cout<<"起点能进入负权回路！"<<endl;
    else{
        for(i=1;i<=N;i++){
            if(!G.Reachable(i)) printf("起点走不到点%d！\n\n",i);
            else{
                printf("起点到点%d的最短路长度是%d\n",i,G.DisTo(i));
                printf("起点到点%d的最短路是",i);
                vector<int> pat=G.PathTo(i);
                for(j=0;j<=pat.size()-1;j++) cout<<pat[j]<<' ';
                cout<<endl;
            }
            cout<<endl;
        }
    }
    
    
    
    
    system("pause");
    return 0;
}
