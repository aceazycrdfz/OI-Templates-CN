#include <bits/stdc++.h>
using namespace std;
bool cmp(int x,int y){
    //���xӦ������yǰ�棬����true 
    return x<y;
}
struct MYSTR{
    int v1,v2;
    //����<�����
    bool operator<(const MYSTR& x) const{
        //���xӦ��������ǰ�棬����true
        if(v1<x.v1) return true;
        if(v1>x.v1) return false;
        return v2<x.v2;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��a�����±�1��N����(Ĭ�ϴ�С����) 
    //sort(a+1,a+N+1);
    
    //�����Ӵ�С(���a��int����) 
    //sort(a+1,a+N+1,greater<int>());
    
    //���Զ���ȽϺ���cmp����
    //sort(a+1,a+N+1,cmp);
    
    //����a��ĳ��struct�����飬��������<����� 
    
    
    
    system("pause");
    return 0;
}
