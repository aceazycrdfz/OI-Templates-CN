#include <bits/stdc++.h>
using namespace std;
template<typename T> class SegTree{
private:
    struct NODE{
        T sum,lazy;
        int lef,rig;
    };
    int N,len;
    vector<NODE> tree;
    void BuildTree(vector<T>& init){
        int i;
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        for(i=len;i<=len+N-1;i++)
            tree[i]={init[i-len+1],0,i-len+1,i-len+1};
        for(i=len+N;i<=2*len-1;i++)
            tree[i]={0,0,i-len+1,i-len+1};
        for(i=len-1;i>=1;i--)
            tree[i]={tree[2*i].sum+tree[2*i+1].sum,0,
                     tree[2*i].lef,tree[2*i+1].rig};
    }
    void PushLazy(int id){
        if(tree[id].lazy==0) return ;
        tree[id].sum+=(tree[id].rig-tree[id].lef+1)*tree[id].lazy;
        if(tree[id].lef!=tree[id].rig){
            tree[2*id].lazy+=tree[id].lazy;
            tree[2*id+1].lazy+=tree[id].lazy;
        }
        tree[id].lazy=0;
    }
    void RgAddRec(int id,int lef,int rig,T val){
        PushLazy(id);
        if(tree[id].lef==lef&&tree[id].rig==rig){
            tree[id].lazy+=val;
            return ;
        }
        if(rig<=tree[2*id].rig) RgAddRec(2*id,lef,rig,val);
        else if(tree[2*id+1].lef<=lef) RgAddRec(2*id+1,lef,rig,val);
        else{
            RgAddRec(2*id,lef,tree[2*id].rig,val);
            RgAddRec(2*id+1,tree[2*id+1].lef,rig,val);
        }
        PushLazy(2*id);
        PushLazy(2*id+1);
        tree[id].sum=tree[2*id].sum+tree[2*id+1].sum;
    }
    T RgSumRec(int id,int lef,int rig){
        PushLazy(id);
        if(tree[id].lef==lef&&tree[id].rig==rig) return tree[id].sum;
        if(rig<=tree[2*id].rig) return RgSumRec(2*id,lef,rig);
        if(tree[2*id+1].lef<=lef) return RgSumRec(2*id+1,lef,rig);
        return RgSumRec(2*id,lef,tree[2*id].rig)+
               RgSumRec(2*id+1,tree[2*id+1].lef,rig);
    }
    
public:
    SegTree(int size):
        N(size)
    {
        vector<T> init(N+1);
        BuildTree(init);
    }
    SegTree(vector<T>& init):
        N(init.size()-1)
    {
        BuildTree(init);
    }
    void RgAdd(int lef,int rig,T val){
        RgAddRec(1,lef,rig,val);
    }
    T RgSum(int lef,int rig){
        return RgSumRec(1,lef,rig);
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义一个线段树object，T是int、double等
    //这样初始化长度N，数组初始全0
    //SegTree<T> a(int N);
    //这样初始化数组为init
    //SegTree<T> a(vector<T> init);
    
    
    //以下下标都是1,...,N
    
    //给lef到rig区间加val，包含lef和rig
    //void a.RgAdd(int lef,int rig,T val);
    
    //求lef到rig区间和，包含lef和rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
