#include <bits/stdc++.h>
using namespace std;
template<typename T> class zkwSegTree{
private:
    int N,len,offset;
    vector<T> tree1,tree2;
    void BuildTree(vector<T>& init){
        int i;
        len=1;
        while(len<N) len*=2;
        tree1.resize(len*2);
        tree1[len]=init[1];
        for(i=len+1;i<=len+N-1;i++) tree1[i]=init[i-len+1]-init[i-len];
        for(i=len+N;i<=2*len-1;i++) tree1[i]=0;
        for(i=len-1;i>=1;i--) tree1[i]=tree1[2*i]+tree1[2*i+1];
        tree2.resize(len*2);
        tree2[len]=(1+offset)*init[1];
        for(i=len+1;i<=len+N-1;i++) tree2[i]=(i-len+1+offset)*(init[i-len+1]-init[i-len]);
        for(i=len+N;i<=2*len-1;i++) tree2[i]=0;
        for(i=len-1;i>=1;i--) tree2[i]=tree2[2*i]+tree2[2*i+1];
    }
    
public:
    zkwSegTree(int size):
        N(size),offset(-N/2-1)
    {
        vector<T> init(N+1);
        BuildTree(init);
    }
    zkwSegTree(vector<T>& init):
        N(init.size()-1),offset(-N/2-1)
    {
        BuildTree(init);
    }
    void RgAdd(int lef,int rig,T val){
        int i=lef+len-1;
        while(i>=1){
            tree1[i]+=val;
            tree2[i]+=(lef+offset)*val;
            i/=2;
        }
        if(rig<N){
            i=rig+1+len-1;
            while(i>=1){
                tree1[i]-=val;
                tree2[i]-=(rig+1+offset)*val;
                i/=2;
            }
        }
    }
    T RgSum(int lef,int rig){
        int i,l=lef+len-1,r=rig+len-1;
        T s=0;
        if(lef>1){
            i=lef-1+len-1;
            while(i>=1){
                if(i%2==0){
                    s+=(rig-lef+1)*tree1[i];
                    i--;
                }
                i/=2;
            }
        }
        while(l<=r){
            if(l%2==1){
                s+=(rig+offset+1)*tree1[l]-tree2[l];
                l++;
            }
            if(r%2==0){
                s+=(rig+offset+1)*tree1[r]-tree2[r];
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
    
    //给lef到rig区间加val，包含lef和rig
    //void a.RgAdd(int lef,int rig,T val);
    
    //求lef到rig区间和，包含lef和rig
    //T a.RgSum(int lef,int rig);
    
    
    
    system("pause");
    return 0;
}
