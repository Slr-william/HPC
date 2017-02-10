#include<iostream>
#include<vector>
#include <stdlib.h>
using namespace std;

#define N 1000

void print(vector<vector<int>> &c){
  for (int i = 0; i < c.size(); i++) {
    for (int j = 0; j < c.size(); j++) {
      cout << c[i][j] <<" ";
    }
    cout<<endl;
  }
}

void randNumbers(vector<vector<int> > &v) {
  for (int i = 0; i < v.size(); i++) {
    for (int j = 0; j < v.size(); j++) {
      v[i][j] = rand() % 10 + 1;
    }
  }
}

void matrixMult(vector<vector<int>> &a, vector<vector<int>> &b, vector<vector<int>> &c){
  for (int i = 0; i<a.size(); i++){
    for (int j = 0; j<b.size(); j++){
        for (int k = 0; k<N; k++){
          c[i][j] += a[i][k] * b[k][j];
         }
      }
   }
}

int main(int argc, char const *argv[]) {
  vector<vector<int> > a(N,vector<int>(N,2));
  vector<vector<int> > b(N,vector<int>(N,3));
  vector<vector<int> > c(N,vector<int>(N,0));

  randNumbers(a);
  randNumbers(b);
  clock_t begin = clock();
  matrixMult( a, b, c);
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  cout<<"Time : "<<time_spent<<endl;
  //print(c);
  return 0;
}
