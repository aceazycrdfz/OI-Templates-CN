#include <bits/stdc++.h>
using namespace std;
template<typename T> class PQFrontier{
private:
    struct Entry{
        int id;
        T val;
        bool exist;
        bool operator<(const Entry& a) const{
            if(exist&&!a.exist) return true;
            if(!exist&&a.exist) return false;
            return val<a.val;
        }
    };
    int N,len;
    vector<Entry> tree;
    void UpdateUp(int x){
        x/=2;
        while(x>0){
            tree[x]=min(tree[2*x],tree[2*x+1]);
            x/=2;
        }
    }
    
public:
    PQFrontier(int Nsize):
        N(Nsize)
    {
        int i;
        len=1;
        while(len<N) len*=2;
        tree.resize(len*2);
        for(i=len;i<=2*len-1;i++) tree[i]={i-len+1,0,false};
        for(i=len-1;i>=1;i--) tree[i]=min(tree[i*2],tree[i*2+1]);
    }
    bool Empty(){
        return !tree[1].exist;
    }
    int MinID(){
        return tree[1].id;
    }
    T ValueOf(int x){
        if(tree[x+len-1].exist) return tree[x+len-1].val;
        return -1;
    }
    void DeleteMin(){
        int x=tree[1].id+len-1;
        tree[x].exist=false;
        UpdateUp(x);
    }
    bool DecreaseVal(int x,T v){
        x+=len-1;
        if(tree[x].exist&&tree[x].val<=v) return false;
        tree[x]={x-len+1,v,true};
        UpdateUp(x);
        return true;
    }
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //��ΪDijkstra��Prim�ع���Frontierģ��
    //��ʼ��N���㣬T��int��double��(�߳�����)
    //PQFrontier<T> F(int N);
    
    
    //���µ�ı�Ŷ���1,...,N
    
    //ѯ��F�Ƿ�Ϊ��
    //bool F.Empty();
    
    //ѯ�ʴ�ʱ��Ӧֵ��С�ĵ�(������֮һ)��Ҫ��F�ǿ�
    //int F.MinID();
    
    //ѯ�ʵ�x��Ӧ��ֵ��Ҫ��x��F��
    //T F.ValueOf(int x);
    
    //ɾ����Ӧֵ��С�ĵ�(MinID�Ǹ�)��Ҫ��F�ǿ�
    //void F.DeleteMin();
    
    //���xĿǰ��F����ֵ����v����ᱻ��Ϊv������true
    //���xĿǰ��F����ֵС�ڵ���v�������·���������false
    //���xĿǰ����F�ڣ�������F��ֵ��Ϊv������true
    //bool DecreaseVal(int x,T v);
    
    
    
    system("pause");
    return 0;
}
