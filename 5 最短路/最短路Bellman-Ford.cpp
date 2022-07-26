#include <iostream>
#include <climits>
using namespace std;
struct EDGE{
    int u,v,len;
}e[100001];
int N,M,dis[100001],prev[100001],path[100001];
void printpath(int id){
    int x,i,ptop=1;
    x=id;
    path[ptop]=x;
    ptop++;
    while(x!=1){
        x=prev[x];
        path[ptop]=x;
        ptop++;
    }
    for(i=ptop-1;i>=1;i--) cout<<path[i]<<' ';
    cout<<endl;
}
int main(){
    int i,j;
    bool negcycle=false,updated=false;
    cin>>N>>M;
    for(i=1;i<=M;i++) cin>>e[i].u>>e[i].v>>e[i].len;
    dis[1]=0;
    for(i=2;i<=N;i++) dis[i]=INT_MAX;
    for(i=1;i<=N-1;i++){
        updated=false;
        for(j=1;j<=M;j++){
            if(dis[e[j].u]==INT_MAX) continue;
            if(dis[e[j].u]+e[j].len<dis[e[j].v]){
                dis[e[j].v]=dis[e[j].u]+e[j].len;
                prev[e[j].v]=e[j].u;
                updated=true;
            }
        }
        if(!updated) break;
    }
    for(j=1;j<=M;j++){
        if(dis[e[j].u]==INT_MAX) continue;
        if(dis[e[j].u]+e[j].len<dis[e[j].v]){
            //dis[e[j].v]=dis[e[j].u]+e[j].len;
            negcycle=true;
            break;
        }
    }
    if(negcycle) cout<<"起点能进入负权回路！"<<endl;
    else{
        for(i=1;i<=N;i++){
            if(dis[i]==INT_MAX) printf("起点走不到点%d！\n\n",i);
            else{
                printf("起点到点%d的最短路长度是%d\n",i,dis[i]);
                printf("起点到点%d的最短路是",i);
                printpath(i);
                cout<<endl;
            }
        }
    }
    system("pause");
    return 0;
}
