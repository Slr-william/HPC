#include<iostream>
#include<vector>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

#define CHUNKSIZE 20
#define N 2000

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
  int i,j,k;
  int nthreads, tid,chunk;
  #pragma omp parallel shared(nthreads,chunk) private(i,j,k,tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0){nthreads = omp_get_num_threads();}

    chunk = CHUNKSIZE;

    #pragma omp for schedule(static,chunk)
    for ( i = 0; i<a.size(); i++){
      for ( j = 0; j<b.size(); j++){
          for ( k = 0; k<b.size(); k++){
            c[i][j] += a[i][k] * b[k][j];
           }
        }
     }
  }
}

int main(int argc, char const *argv[]) {

  vector<vector<int> > a(N,vector<int>(N,1));
  vector<vector<int> > b(N,vector<int>(N,1));
  vector<vector<int> > c(N,vector<int>(N,0));
  double begin, end;
  randNumbers(a);
  randNumbers(b);
  begin = omp_get_wtime();
  matrixMult( a, b, c);
  end = omp_get_wtime();
  cout<<"Time : "<<end - begin<<endl;
  cout<<"Begin: "<<begin<<" End : "<<end<<endl;
  //print(c);
  return 0;
}
