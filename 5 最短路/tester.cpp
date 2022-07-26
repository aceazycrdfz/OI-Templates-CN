#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    int N,M,i,j,u,v,len;
    cin>>N>>M;
    BellmanFord<int> G(N,M);
    for(i=1;i<=M;i++){
        cin>>u>>v>>len;
        G.AddEdge(u,v,len);
    }
    G.RunSSSP(1);
    if(G.NegCycle()) cout<<"起点能进入负权回路！"<<endl;
    else{
        for(i=1;i<=N;i++){
            if(!G.Reachable(i)) printf("起点走不到点%d！\n\n",i);
            else{
                printf("起点到点%d的最短路长度是%d\n",i,G.DisTo(i));
                printf("起点到点%d的最短路是",i);
                vector<int> pat=G.PathTo(i);
                for(j=1;j<=pat.size()-1;j++) cout<<pat[j]<<' ';
                cout<<endl;
            }
            cout<<endl;
        }
    }
    
    
    
    
    system("pause");
    return 0;
}
