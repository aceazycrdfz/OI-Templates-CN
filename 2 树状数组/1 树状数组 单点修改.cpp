#include <bits/stdc++.h>
using namespace std;
template<typename T> class BITree{
private:
    int N;
    vector<T> tree;
    inline int lowbit(int x){
        return x&-x;
    }
    T PfSum(int x){
        T ret=0;
        while(x>=1){
            ret+=tree[x];
            x-=lowbit(x);
        }
        return ret;
    }
    
public:
    BITree(int size):
        N(size),tree(N+1){}
    BITree(vector<T>& init):
        N(init.size()-1),tree(N+1)
    {
        int i;
        vector<T> pre(N+1);
        for(i=1;i<=N;i++){
            pre[i]=pre[i-1]+init[i];
            tree[i]=pre[i]-pre[i-lowbit(i)];
        }
    }
    void Add(int x,T val){
        while(x<=N){
            tree[x]+=val;
            x+=lowbit(x);
        }
    }
    T RgSum(int lef,int rig){
        return PfSum(rig)-PfSum(lef-1);
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����һ����״����object��T��int��double��
    //������ʼ������N�������ʼȫ0
    //BITree<T> a(int N);
    //������ʼ������Ϊinit
    //BITree<T> a(vector<T> init);
    
    
    //�����±궼��1,...,N
    
    //����i������val
    //void a.Add(int x,T val);
    
    //��lef��rig����ͣ�����lef��rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
