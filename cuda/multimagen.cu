#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <cuda.h>

using namespace cv;
using namespace std;

__global__ void PictureKernell(unsigned char * d_Pin, unsigned char * d_Pout, int n, int m ){
	int Row = blockIdx.y*blockDim.y + threadIdx.y;

	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((Row < m)&&(Col < n)){
		d_Pout[Row*n + Col] = 2*d_Pin[Row*n+Col];
	}
}


void makeImage(unsigned char *h_img, unsigned char *result_img, int width, int height) {
  int size = width * height * sizeof(unsigned char);
  unsigned char *d_img, *d_result_img;
  
  cudaMalloc((void**) &d_img, size);
  cudaMalloc((void**) &d_result_img, size);
  cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice);
  
  int block_size = 64;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);
  PictureKernell<<<dim_grid, dim_block>>>(d_img, d_result_img, width, height);
  cudaMemcpy(result_img, d_result_img, size, cudaMemcpyDeviceToHost);
  
  cudaFree(d_img);
  cudaFree(d_result_img);
}

int main(int argc, char *argv[]){

	string name = "grayRock.jpg";

	if (argc > 1){
		name = string(argv[1]);
	}
	
	cout<<"Name: "<<name<<endl;

	Mat image = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	int height = image.rows;
	int width = image.cols;
	int size = width * height * sizeof(unsigned char);
  
	unsigned char *result_img = (unsigned char*) malloc(size);
	unsigned char *org_img = (unsigned char*) image.data;

	makeImage(org_img,result_img,width,height);
	Mat final_image(height, width, CV_8UC1, (void*) result_img);

	namedWindow( "Display window", WINDOW_NORMAL ); // Create a window for display.
    imshow( "Display window", final_image);        // Show our image inside it.
    waitKey(0);	
    free(result_img);
	return 0;
	
}
