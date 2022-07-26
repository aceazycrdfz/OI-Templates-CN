#include <bits/stdc++.h>
using namespace std;
class UFSet{
    private:
        int N,setcnt;
        vector<int> ufs;
        int Find(int x){
            if(ufs[x]<0) return x;
            return ufs[x]=Find(ufs[x]);
        }
        
    public:
        UFSet(int size){
            N=size;
            setcnt=N;
            ufs.resize(N+1,-1);
        }
        void Union(int x,int y){
            x=Find(x);
            y=Find(y);
            if(x==y) return ;
            setcnt--;
            if(-ufs[x]>=-ufs[y]){
                ufs[x]+=ufs[y];
                ufs[y]=x;
            }
            else{
                ufs[y]+=ufs[x];
                ufs[x]=y;
            }
        }
        bool SameSet(int x,int y){
            return Find(x)==Find(y);
        }
        int GetSize(int x){
            return -ufs[Find(x)];
        }
        int NSet(){
            return setcnt;
        }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����һ�����鼯object
    //������ʼ��N���㣬��������ֻ�����Լ�
    //UFSet a(int N);
    
    
    //����indexing����1,...,N
    
    //�ϲ�x��y���ڵļ��ϣ���ͬһ���������·���
    //void a.Union(int x,int y);
    
    //ѯ��x��y�Ƿ�Ϊͬһ����
    //bool a.SameSet(int x,int y);
    
    //ѯ��x���ڵļ��ϵĴ�С
    //int a.GetSize(int x);
    
    //ѯ���ܹ��ж��ٸ�����
    //int a.NSet();
    
    
    
    system("pause");
    return 0;
}
