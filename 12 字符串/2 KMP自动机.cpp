#include <bits/stdc++.h>
using namespace std;
template<typename T> class KMPAutomaton{
private:
    int N;
    DFAutomaton<T> dfa;
    
public:
    KMPAutomaton(vector<T> str):
        N(str.size()-1),dfa(N+1,1)
    {
        int i;
        dfa.SetAcc(N+1,true);
        dfa.SetPos(1);
        for(i=1;i<=N;i++){
            dfa.AddTran(i,str[i],i+1);
            if(i>1) dfa.Transition(str[i]);
            dfa.CopyTran(dfa.GetPos(),i+1);
        }
        dfa.SetPos(1);
    }
    void Reset(){
        dfa.SetPos(1);
    }
    void Transition(T c){
        dfa.Transition(c); 
    }
    bool Accept(){
        return dfa.Accept();
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    vector<char> pat={' ','a','a','b'};
    KMPAutomaton<char> kmp(pat);
    while(true){
        char inp;
        cin>>inp;
        kmp.Transition(inp);
        if(kmp.Accept()) cout<<"Accept"<<endl;
        else cout<<"Reject"<<endl;
    }
    
    
    system("pause");
    return 0;
}
