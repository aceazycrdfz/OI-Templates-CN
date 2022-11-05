#include <bits/stdc++.h>
using namespace std;
template<typename T> class SegTree{
private:
    struct NODE{
        T sum,lazy;
        NODE *lson,*rson;
    };
    int N,len,tTop;
    vector<NODE> tree;
    NODE *Root;
    NODE *BuildTree(int lef,int rig,vector<T>& init){
        NODE *ret=&tree[tTop];
        tTop++;
        ret->lazy=0;
        if(lef==rig){
            ret->sum=init[lef];
            ret->lson=NULL;
            ret->rson=NULL;
        }
        else{
            int mid=lef+(rig-lef)/2;
            ret->lson=BuildTree(lef,mid,init);
            ret->rson=BuildTree(mid+1,rig,init);
            ret->sum=ret->lson->sum+ret->rson->sum;
        }
        return ret;
    }
    void PushLazy(NODE *nod,int nl,int nr){
        if(nod->lazy==0) return ;
        nod->sum+=(nr-nl+1)*nod->lazy;
        if(nod->lson!=NULL) nod->lson->lazy+=nod->lazy;
        if(nod->rson!=NULL) nod->rson->lazy+=nod->lazy;
        nod->lazy=0;
    }
    void RgAddRec(NODE *nod,int nl,int nr,int lef,int rig,T val){
        PushLazy(nod,nl,nr);
        if(nl==lef&&nr==rig){
            nod->lazy+=val;
            return ;
        }
        int LsRb=nl+(nr-nl)/2,RsLb=nl+(nr-nl)/2+1;
        if(rig<=LsRb) RgAddRec(nod->lson,nl,LsRb,lef,rig,val);
        else if(RsLb<=lef) RgAddRec(nod->rson,RsLb,nr,lef,rig,val);
        else{
            RgAddRec(nod->lson,nl,LsRb,lef,LsRb,val);
            RgAddRec(nod->rson,RsLb,nr,RsLb,rig,val);
        }
        PushLazy(nod->lson,nl,LsRb);
        PushLazy(nod->rson,RsLb,nr);
        nod->sum=nod->lson->sum+nod->rson->sum;
    }
    T RgSumRec(NODE *nod,int nl,int nr,int lef,int rig){
        PushLazy(nod,nl,nr);
        if(nl==lef&&nr==rig) return nod->sum;
        int LsRb=nl+(nr-nl)/2,RsLb=nl+(nr-nl)/2+1;
        if(rig<=LsRb) return RgSumRec(nod->lson,nl,LsRb,lef,rig);
        if(RsLb<=lef) return RgSumRec(nod->rson,RsLb,nr,lef,rig);
        return RgSumRec(nod->lson,nl,LsRb,lef,LsRb)+
               RgSumRec(nod->rson,RsLb,nr,RsLb,rig);
    }
    
public:
    SegTree(int size):
        N(size)
    {
        vector<T> init(N+1);
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        tTop=0;
        Root=BuildTree(1,N,init);
    }
    SegTree(vector<T>& init):
        N(init.size()-1)
    {
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        tTop=0;
        Root=BuildTree(1,N,init);
    }
    void RgAdd(int lef,int rig,T val){
        RgAddRec(Root,1,N,lef,rig,val);
    }
    T RgSum(int lef,int rig){
        return RgSumRec(Root,1,N,lef,rig);
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
