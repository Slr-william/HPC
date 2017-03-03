#include<iostream>
#include<vector>

#define N 100
using namespace std;

void print(vector<int> v) {
  for (int i = 0; i < v.size(); i++) {
    cout << v[i] <<" ";
    if (i%10 == 0) {
      cout<<endl;
    }
  }
}

vector<int> sumVector(vector<int> A, vector<int> B){
  vector<int> C;
  for (int i = 0; i < N; i++) {
    C.push_back(A[i] + B[i]);
  }
  return C;
}

int main(int argc, char const *argv[]) {
  vector<int> A(N,3);
  vector<int> B(N,3);
  vector<int> C;

  C = sumVector(A, B);
  print(C);
  return 0;
}
