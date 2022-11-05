#include <bits/stdc++.h>
using namespace std;
template<typename T> class zkwSegTree{
private:
    int N,len;
    vector<T> tree;
    void BuildTree(vector<T>& init){
        int i;
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        for(i=len;i<=len+N-1;i++) tree[i]=init[i-len+1];
        for(i=len+N;i<=2*len-1;i++) tree[i]=0;
        for(i=len-1;i>=1;i--) tree[i]=tree[2*i]+tree[2*i+1];
    }
    
public:
    zkwSegTree(int size):
        N(size)
    {
        vector<T> init(N+1);
        BuildTree(init);
    }
    zkwSegTree(vector<T>& init):
        N(init.size()-1)
    {
        BuildTree(init);
    }
    void Add(int x,T val){
        int i=x+len-1;
        while(i>=1){
            tree[i]+=val;
            i/=2;
        }
    }
    T RgSum(int lef,int rig){
        int i,l=lef+len-1,r=rig+len-1;
        T s=0;
        while(l<=r){
            if(l%2==1){
                s+=tree[l];
                l++;
            }
            if(r%2==0){
                s+=tree[r];
                r--;
            }
            l/=2;
            r/=2;
        }
        return s;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //定义一个线段树object，T是int、double等
    //这样初始化长度N，数组初始全0
    //zkwSegTree<T> a(int N);
    //这样初始化数组为init
    //zkwSegTree<T> a(vector<T> init);
    
    
    //以下下标都是1,...,N
    
    //给第i个数加val
    //void a.Add(int x,T val);
    
    //求lef到rig区间和，包含lef和rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
