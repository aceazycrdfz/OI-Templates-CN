#include <bits/stdc++.h>
using namespace std;
struct MYSTR{
    int x,y,z;
    //����<�����
    bool operator<(const MYSTR& a) const{
        //����ұ�aС������true
        if(x<a.x) return true;
        if(x>a.x) return false;
        return y<a.y;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����<�����������ڴ�����ֱ����"<"�Ƚ�
    //min��maxҲ����ֱ������
    //��������">","==","<="�������������Ҫ��������
    
    
    
    system("pause");
    return 0;
}
