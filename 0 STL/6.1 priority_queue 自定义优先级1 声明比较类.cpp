#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y;
};
//дһ��struct���ڱȽ�
struct MYSTRcmp{
    bool operator() (MYSTR a,MYSTR b) const{
        //���bӦ��a���ȣ�����true
        if(a.x<b.x) return true;
        if(a.x>b.x) return false;
        return a.y<b.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��MYSTRcmp���struct���Ƚ� 
    priority_queue<MYSTR,vector<MYSTR>,MYSTRcmp> test1;
    
    
    MYSTR e1,e2,e3;
    e1.x=1;
    e1.y=1;
    e2.x=2;
    e2.y=2;
    e3.x=2;
    e3.y=1;
    test1.push(e1);
    test1.push(e2);
    test1.push(e3);
    
    
    while(!test1.empty()){
        MYSTR tp=test1.top();
        cout<<tp.x<<' '<<tp.y<<endl;
        test1.pop();
    }
    
    
    system("pause");
    return 0;
}
