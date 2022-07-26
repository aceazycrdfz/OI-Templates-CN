#include <bits/stdc++.h>
using namespace std;
template<typename T> class BellmanFord{
    private:
        struct EDGE{
            int u,v;
            T len;
        };
        int N,M,s;
        bool negcyc;
        vector<EDGE> e;
        vector<T> dis;
        vector<bool> reached;
        vector<int> prev;
    
    public:
        BellmanFord(int Nsize,int Esize){
            N=Nsize;
            M=0;
            e.resize(Esize+1);
            dis.resize(N+1);
            reached.resize(N+1);
            prev.resize(N+1);
        }
        void AddEdge(int u,int v,T len){
            M++;
            e[M].u=u;
            e[M].v=v;
            e[M].len=len;
        }
        void RunSSSP(int start){
            int i,j;
            bool updated;
            s=start;
            reached[s]=true;
            dis[s]=0;
            negcyc=false;
            for(i=1;i<=N-1;i++){
                updated=false;
                for(j=1;j<=M;j++){
                    if(!reached[e[j].u]) continue;
                    if(!reached[e[j].v]||dis[e[j].u]+e[j].len<dis[e[j].v]){
                        reached[e[j].v]=true;
                        dis[e[j].v]=dis[e[j].u]+e[j].len;
                        prev[e[j].v]=e[j].u;
                        updated=true;
                    }
                }
                if(!updated) break;
            }
            for(j=1;j<=M;j++){
                if(!reached[e[j].u]) continue;
                if(!reached[e[j].v]||dis[e[j].u]+e[j].len<dis[e[j].v]){
                    negcyc=true;
                    break;
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
            reverse(path.begin(),path.end());
            return path;
        }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��ʼ��һ����N�����M���ߵ�ͼ��T��int��double��(�߳�����)
    //M���Թ��󣬲��ܹ�С
    //BellmanFord<T> a(int N,int M);
    
    //���µ�ı����Ϊ1,...,N
    
    //����һ����uָ��v������Ϊlen�������
    //void a.AddEdge(int u,int v,T len);
    
    //��startΪ�����㵥Դ���·��
    //void a.RunSSSP(int start); 
    
    //ѯ������Ƿ��ܽ��븺Ȩ��·
    //bool a.NegCycle();
    
    //ѯ������Ƿ��ܵ���id�ŵ� 
    //bool a.Reachable(id);
    
    //ѯ����㵽id�ŵ�����·��
    //Ҫ����㲻�ܽ��븺Ȩ��·���ܵ���id�ŵ�
    //T a.DisTo(int id);
     
    //����һ������㵽id�ŵ�����·�� 
    //Ҫ����㲻�ܽ��븺Ȩ��·���ܵ���id�ŵ�
    //vector<int> a.PathTo(int id);
    //���������һ����(�±�0)��start�����һ����id 
    
    
    
    system("pause");
    return 0;
}
