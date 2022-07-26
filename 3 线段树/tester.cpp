#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    vector<long long> vec({-999,1,2,3,4,5,6,7,8,9,10});
    BITree<long long> bit(vec);
    int x,y,v;
    char op;
    while(true){
        cin>>op;
        if(op=='a'){
            cin>>x>>y>>v;
            bit.RgAdd(x,y,v);
        }
        if(op=='b'){
            cin>>x>>y;
            cout<<bit.RgSum(x,y)<<endl;
        }
    }
    
    
    
    
    system("pause");
    return 0;
}
