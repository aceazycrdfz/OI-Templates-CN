#include <bits/stdc++.h>
using namespace std;
template<typename T> class SegTree{
    private:
        struct NODE{
            T sum;
            queue<T> lazy;
        };
        int N,len;
        vector<NODE> tree;
        void BuildTree(vector<T>& init){
            int i;
            len=1;
            while(len<N) len*=2;
            tree.resize(len*2);
            for(i=len;i<=len+N-1;i++) tree[i].sum=init[i-len+1];
            for(i=len+N;i<=2*len-1;i++) tree[i].sum=0;
            for(i=len-1;i>=1;i--) tree[i].sum=tree[2*i].sum+tree[2*i+1].sum;
        }
        void PushLazy(int id,int il,int ir){
            while(!tree[id].lazy.empty()){
                tree[id].sum+=(ir-il+1)*tree[id].lazy.front();
                if(il!=ir){
                    tree[2*id].lazy.push(tree[id].lazy.front());
                    tree[2*id+1].lazy.push(tree[id].lazy.front());
                }
                tree[id].lazy.pop();
            }
        }
        void RgAddRec(int id,int il,int ir,int lef,int rig,T val){
            PushLazy(id,il,ir);
            if(il==lef&&ir==rig){
                tree[id].lazy.push(val);
                return ;
            }
            int LsRb=il+(ir-il)/2,RsLb=il+(ir-il)/2+1;
            if(rig<=LsRb) RgAddRec(2*id,il,LsRb,lef,rig,val);
            else if(RsLb<=lef) RgAddRec(2*id+1,RsLb,ir,lef,rig,val);
            else{
                RgAddRec(2*id,il,LsRb,lef,LsRb,val);
                RgAddRec(2*id+1,RsLb,ir,RsLb,rig,val);
            }
            PushLazy(2*id,il,LsRb);
            PushLazy(2*id+1,RsLb,ir);
            tree[id].sum=tree[2*id].sum+tree[2*id+1].sum;
        }
        T RgSumRec(int id,int il,int ir,int lef,int rig){
            PushLazy(id,il,ir);
            if(il==lef&&ir==rig) return tree[id].sum;
            int LsRb=il+(ir-il)/2,RsLb=il+(ir-il)/2+1;
            if(rig<=LsRb) return RgSumRec(2*id,il,LsRb,lef,rig);
            if(RsLb<=lef) return RgSumRec(2*id+1,RsLb,ir,lef,rig);
            return RgSumRec(2*id,il,LsRb,lef,LsRb)+
                   RgSumRec(2*id+1,RsLb,ir,RsLb,rig);
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
            RgAddRec(1,1,len,lef,rig,val);
        }
        T RgSum(int lef,int rig){
            return RgSumRec(1,1,len,lef,rig);
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
