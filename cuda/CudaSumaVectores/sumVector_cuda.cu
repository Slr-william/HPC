#include<cuda.h>
#include<vector>
#include<stdio.h>

void print(float *v, int n ) {
  for (int i = 0; i < n ; i++) {
   printf("%f ",v[i]);
    if (i%10 == 0) {
      printf("\n");
    }
  }
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void){

    int numElements = 500;
    int size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1;//rand()/(float)RAND_MAX;
        h_B[i] = 1;//rand()/(float)RAND_MAX;
    }

    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = 2;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
	
	print(h_C, numElements);	
	
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
