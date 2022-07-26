#include <bits/stdc++.h>
using namespace std;
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
        cout<<"�������޻�ͼ��"<<endl;
        vector<int> ans=G.Result();
        for(i=1;i<=N;i++) cout<<ans[i]<<' ';
        cout<<endl;
    }
    else cout<<"���������޻�ͼ��"<<endl;
    
    
    
    
    system("pause");
    return 0;
}
