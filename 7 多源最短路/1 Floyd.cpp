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
    
    //��ʼ��һ����N�����ͼ��T��int��double��(�߳�����)
    //Floyd<T> G(int N);
    
    
    //���µ�ı�Ŷ���1,...,N
    
    //����һ����uָ��v������Ϊlen�������
    //void G.AddEdge(int u,int v,T len);
    
    //�����Դ���·��
    //void G.RunAPSP();
    
    //ѯ���Ƿ���ڸ�Ȩ��·
    //ֻ�д˽��Ϊfalse��������ѯ�ʽ����������
    //bool G.NegCycle();
    
    //ѯ�ʴӵ�u�Ƿ��ܵ����v
    //bool G.Reachable(int u,int v);
    
    //ѯ�ʵ�u����v�����·��
    //Ҫ���u�ܵ����v
    //T G.DisTo(int u,int v);
     
    //����һ���ӵ�u����v�����·��
    //Ҫ���u�ܵ����v
    //vector<int> G.PathTo(int u,int v);
    //���������һ����(�±�0)��u�����һ����v
    
    
    
    system("pause");
    return 0;
}
