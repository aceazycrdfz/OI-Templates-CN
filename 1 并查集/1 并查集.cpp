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
    
    //定义一个并查集object
    //这样初始化N个点，各个集合只包含自己
    //UFSet a(int N);
    
    
    //以下indexing都是1,...,N
    
    //合并x和y所在的集合，在同一集合则无事发生
    //void a.Union(int x,int y);
    
    //询问x和y是否为同一集合
    //bool a.SameSet(int x,int y);
    
    //询问x所在的集合的大小
    //int a.GetSize(int x);
    
    //询问总共有多少个集合
    //int a.NSet();
    
    
    
    system("pause");
    return 0;
}
