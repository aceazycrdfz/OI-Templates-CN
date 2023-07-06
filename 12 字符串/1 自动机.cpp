#include <bits/stdc++.h>
using namespace std;
template<typename T> class DFAutomaton{
private:
    struct STATE{
        int def;
        bool acc;
        unordered_map<T,int> e;
    };
    int N,pos;
    vector<STATE> s;
    
public:
    DFAutomaton(int Nsize,int def):
        N(Nsize),s(N+1,{def,false,{}})
    {
        int i;
        if(def==0) for(i=1;i<=N;i++) s[i].def=i;
    }
    void SetAcc(int id,bool acc){
        s[id].acc=acc;
    }
    void AddTran(int u,T c,int v){
        if(v==0) s[u].e.erase(c);
        else s[u].e[c]=v;
    }
    void CopyTran(int from,int to){
        s[to].e=s[from].e;
    }
    void SetDef(int id,int def){
        s[id].def=def;
    }
    void SetPos(int p){
        pos=p;
    }
    void Transition(T c){
        if(s[pos].e[c]==0){
            s[pos].e.erase(c);
            pos=s[pos].def;
        }
        else pos=s[pos].e[c];
    }
    bool Accept(){
        return s[pos].acc;
    }
    int GetPos(){
        return pos;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    
    
    system("pause");
    return 0;
}
