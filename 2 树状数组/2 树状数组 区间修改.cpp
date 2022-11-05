#include <bits/stdc++.h>
using namespace std;
template<typename T> class BITree{
private:
    int N,offset;
    vector<T> tree1,tree2;
    inline int lowbit(int x){
        return x&-x;
    }
    T PfSum(int x){
        int i=x;
        T s=0;
        while(i>=1){
            s+=(x+offset+1)*tree1[i]-tree2[i];
            i-=lowbit(i);
        }
        return s;
    }
    
public:
    BITree(int size):
        N(size),offset(-N/2-1),tree1(N+1),tree2(N+1){}
    BITree(vector<T>& init):
        N(init.size()-1),offset(-N/2-1),tree1(N+1),tree2(N+1)
    {
        int i;
        T d;
        vector<T> pre1(N+1),pre2(N+1);
        for(i=1;i<=N;i++){
            if(i==1) d=init[1];
            else d=init[i]-init[i-1];
            pre1[i]=pre1[i-1]+d;
            tree1[i]=pre1[i]-pre1[i-lowbit(i)];
            pre2[i]=pre2[i-1]+(i+offset)*d;
            tree2[i]=pre2[i]-pre2[i-lowbit(i)];
        }
    }
    void RgAdd(int lef,int rig,T val){
        int i=lef;
        while(i<=N){
            tree1[i]+=val;
            tree2[i]+=(lef+offset)*val;
            i+=lowbit(i);
        }
        if(rig<N){
            i=rig+1;
            while(i<=N){
                tree1[i]-=val;
                tree2[i]-=(rig+1+offset)*val;
                i+=lowbit(i);
            }
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
    
    //��lef��rig�����val������lef��rig
    //void a.RgAdd(int lef,int rig,T val);
    
    //��lef��rig����ͣ�����lef��rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
