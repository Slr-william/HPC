#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <cuda.h>

using namespace std;

__global__ void PictureKernell(float * d_Pin, float * d_Pout, int n, int m ){
	int Row = blockIdx.y*blockDim.y + threadIdx.y;

	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((Row < m)&&(Col < n)){
		d_Pout[Row*n + Col] = 2*d_Pin[Row*n+Col];
	}
}

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P,int width){
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ((Row < width)&&(Col < width)){
		float Pvalue = 0;
		for (int i = 0; i < width; ++i){
			Pvalue += d_M[Row*width+k]*d_N[k*width+Col];
		}
		d_P[Row*width + Col] = Pvalue;
	}
}

int main(int argc, char *argv[]){

	string name = "grayRock.jpg";

	if (argc > 1){
		name = string(argv[1]);
	}

	Mat image;
	image = imread(, CV_LOAD_IMAGE_COLOR);
	namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0);
	
	return 0;
	
}
