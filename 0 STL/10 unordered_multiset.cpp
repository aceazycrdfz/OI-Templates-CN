#include <bits/stdc++.h>
using namespace std;
//���hash��ʱ����Ϊ��������Ա�hack 
struct my_hash{
    static uint64_t splitmix64(uint64_t x){
        x+=0x9e3779b97f4a7c15;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9;
        x=(x^(x>>27))*0x94d049bb133111eb;
        return x^(x>>31);
    }
    
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM=
            chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x+FIXED_RANDOM);
    }
    
    //���std::pair<int,int>��Ϊ�������͵Ĺ�ϣ����
    size_t operator()(pair<uint64_t,uint64_t> x) const {
        static const uint64_t FIXED_RANDOM=
            chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x.first+FIXED_RANDOM)^
               (splitmix64(x.second+FIXED_RANDOM)>>1);
    }
    
    //�����Լ�д��Ҫhash������
};
int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    
    //����T unordered_multiset
    //unordered_multiset<T> a;
    
    //����a���ж���Ԫ�أ��ظ�Ԫ�ػᱻ�ظ���
    //int a.size();
    
    //����a�Ƿ�Ϊ��(sizeΪ0)
    //bool a.empty();
    
    //����Ԫ��x
    //void a.insert(T x);
    
    //ɾ��Ԫ��x������䲻�ڼ����������·������ж����ȫ��ɾ��
    //���ر�ɾ��Ԫ�ظ���
    //int a.erase(T x);
    
    //����Ԫ��x�ڼ����еĸ���
    //int a.count(T x);
    
    //��ռ���
    //void a.clear();
    
    //���ڵ�����it������������ָ���Ԫ��
    //*it
    
    
    
    //unordered�����ǹ�ϣ������lower_bound��
    //���ǿ��������Զ����ϣ����
    //unordered_multiset<T,my_hash> a;
    
    
    
    system("pause");
    return 0;
}
