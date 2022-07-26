#include <bits/stdc++.h>
using namespace std;
template<typename T> class SegTree{
    private:
        struct NODE{
            T sum,lazy;
            int lef,rig;
            NODE *lson,*rson;
        };
        int N,len,tTop;
        vector<NODE> tree;
        NODE *Root;
        NODE *BuildTree(int lef,int rig,vector<T>& init){
            NODE *ret=&tree[tTop];
            tTop++;
            ret->lazy=0;
            ret->lef=lef;
            ret->rig=rig;
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
        void PushLazy(NODE *nod){
            if(nod->lazy==0) return ;
            nod->sum+=(nod->rig-nod->lef+1)*nod->lazy;
            if(nod->lson!=NULL) nod->lson->lazy+=nod->lazy;
            if(nod->rson!=NULL) nod->rson->lazy+=nod->lazy;
            nod->lazy=0;
        }
        void RgAddRec(NODE *nod,int lef,int rig,T val){
            PushLazy(nod);
            if(nod->lef==lef&&nod->rig==rig){
                nod->lazy+=val;
                return ;
            }
            if(rig<=nod->lson->rig) RgAddRec(nod->lson,lef,rig,val);
            else if(nod->rson->lef<=lef) RgAddRec(nod->rson,lef,rig,val);
            else{
                RgAddRec(nod->lson,lef,nod->lson->rig,val);
                RgAddRec(nod->rson,nod->rson->lef,rig,val);
            }
            PushLazy(nod->lson);
            PushLazy(nod->rson);
            nod->sum=nod->lson->sum+nod->rson->sum;
        }
        T RgSumRec(NODE *nod,int lef,int rig){
            PushLazy(nod);
            if(nod->lef==lef&&nod->rig==rig) return nod->sum;
            if(rig<=nod->lson->rig) return RgSumRec(nod->lson,lef,rig);
            if(nod->rson->lef<=lef) return RgSumRec(nod->rson,lef,rig);
            return RgSumRec(nod->lson,lef,nod->lson->rig)+
                   RgSumRec(nod->rson,nod->rson->lef,rig);
        }
        
    public:
        SegTree(int size){
            N=size;
            vector<T> init(N+1);
            len=1;
            while(len<N) len*=2;
            tree.resize(len*2);
            tTop=0;
            Root=BuildTree(1,N,init);
        }
        SegTree(vector<T>& init){
            N=init.size()-1;
            len=1;
            while(len<N) len*=2;
            tree.resize(len*2);
            tTop=0;
            Root=BuildTree(1,N,init);
        }
        void RgAdd(int lef,int rig,T val){
            RgAddRec(Root,lef,rig,val);
        }
        T RgSum(int lef,int rig){
            return RgSumRec(Root,lef,rig);
        }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����һ���߶���object��T��int��double��
    //������ʼ������N�������ʼȫ0
    //SegTree<T> a(int N);
    //������ʼ������Ϊinit
    //SegTree<T> a(vector<T> init);
    
    
    //����indexing����1,...,N
    
    //��lef��rig�����val������lef��rig
    //void a.RgAdd(int lef,int rig,T val);
    
    //��lef��rig����ͣ�����lef��rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
