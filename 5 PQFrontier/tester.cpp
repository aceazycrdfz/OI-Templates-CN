#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    int i,N=10;
    DijFrontier<long long> t(N);
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    t.DecreaseVal(1,4);
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    t.DecreaseVal(2,3);
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    t.DecreaseVal(1,1);
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    t.DeleteMin();
    
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    t.DecreaseVal(2,2);
    
    if(t.Empty()) cout<<"Empty"<<endl<<endl;
    else{
        cout<<"Min ID: "<<t.MinID()<<endl;
        cout<<"Min Val: "<<t.ValueOf(t.MinID());
        cout<<endl<<endl;
    }
    
    
    
    system("pause");
    return 0;
}
