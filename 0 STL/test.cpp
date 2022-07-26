#include <bits/stdc++.h>
using namespace std;
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    unordered_multimap<long long,double> m;
    m.insert(make_pair(1,2.1));
    m.insert(make_pair(2,2.2));
    m.insert(make_pair(2,2.2));
    m.insert(make_pair(2,2.2));
    m.insert(make_pair(2,2.3));
    m.insert(make_pair(3,2.3));
    m.insert(make_pair(4,2.5));
    
    auto iter=next(m.begin(),1);
    m.erase(2);
    for(auto it=m.begin();it!=m.end();it++){
        cout<<it->second<<endl;
    }
    cout<<endl;
    
    system("pause");
    return 0;
}
