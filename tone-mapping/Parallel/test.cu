#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <string>

#define BLUE 0
#define GREEN 1
#define RED 2

using namespace cv;


__device__ float maxLum = 0;

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

void checkError(cudaError_t err) {
	if(err!=cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}


__device__ float logarithmic_mapping(float k, float q, float val_pixel){
	return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}

__global__ void find_maximum_kernel(float *array, int *mutex, unsigned int n, int blockSize){
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	extern	__shared__ float cache[];

	float temp = -1.0;
	while(index + offset < n){
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		maxLum = fmaxf(maxLum, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void tonemap_logarithmic(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float q, float k){	
	//printf("maxLum : %f\n", maxLum);
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	if(Row < height && Col < width) {
		imageOut[(Row*width+Col)*3+BLUE] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+BLUE]);
		imageOut[(Row*width+Col)*3+GREEN] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+GREEN]);
		imageOut[(Row*width+Col)*3+RED] = logarithmic_mapping(k, q, imageIn[(Row*width+Col)*3+RED]);
	}
}

 void showImage(Mat &image, const char *window) {
	namedWindow(window, CV_WINDOW_NORMAL);
	imshow(window, image);
 }

__device__ float gamma_correction(float f_stop, float gamma, float val)
{
	return powf((val*powf(2,f_stop)),(1.0/gamma));
}

__global__ void tonemap_gamma(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float f_stop,
						float gamma)
{
	int Row = blockDim.y * blockIdx.y + threadIdx.y;
	int Col = blockDim.x * blockIdx.x + threadIdx.x;

	if(Row < height && Col < width) {
		imageOut[(Row*width+Col)*3+BLUE] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+BLUE]);
		imageOut[(Row*width+Col)*3+GREEN] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+GREEN]);
		imageOut[(Row*width+Col)*3+RED] = gamma_correction(f_stop, gamma, imageIn[(Row*width+Col)*3+RED]);
	}
}

int main(int argc, char** argv){
	char* image_name = argv[1];
    char* image_out_name = argv[5];
	float *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut;
	char* option;
	Mat hdr, ldr;
	Size imageSize;
	int width, height, channels, sizeImage;
	float q=0.0, k=0.0;
	int show_flag, N;
	int *d_mutex;
//	std::vector<Mat>images;

//	printf("%s\n", image_name);
	hdr = imread(image_name, -1);
	if(argc !=7 || !hdr.data) {
		printf("No image Data \n");
		printf("Usage: ./test <file_path> <q|f-stop> <k|gamma> <show_flag> <output_file_path> [L|G]");
		return -1;
	}

	q = atof(argv[2]);
	k = atof(argv[3]);
	show_flag = atoi(argv[4]);
	option = argv[7];

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}
	imageSize = hdr.size();
	width = imageSize.width;
	height = imageSize.height;
	channels = hdr.channels();
	N = width*height*channels;
	sizeImage = sizeof(float)*width*height*channels;

	//printf("Width: %d\nHeight: %d\n", width, height);
	std::string ty =  type2str( hdr.type() );
//	printf("Image: %s %dx%d \n", ty.c_str(), hdr.cols, hdr.rows );

	//printf("Channels: %d\nDepth: %d\n", hdr.channels(), hdr.depth());

	h_ImageData = (float *) malloc (sizeImage);
	h_ImageData = (float *)hdr.data;
	h_ImageOut = (float *) malloc (sizeImage);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;

	checkError(cudaMalloc((void**)&d_mutex, sizeof(int)));
	checkError(cudaMalloc((void **)&d_ImageData, sizeImage));
	checkError(cudaMalloc((void **)&d_ImageOut, sizeImage));
	checkError(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

	cudaMemset(d_mutex, 0, sizeof(int));

	int blockSize = 32;	
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
	switch(option){
		case "l":
		case "L": {
			printf("Logarithmic_mapping\n");
			cudaEventRecord(start);
			find_maximum_kernel<<< dimGrid, dimBlock, sizeof(float)*blockSize >>>(d_ImageData, d_mutex, N, blockSize);
			cudaDeviceSynchronize();
			tonemap_logarithmic<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, 32, q, k);
			cudaEventRecord(stop);
			break;
		}
		case "g":
		case "G": {
			printf("Gamma_correction\n");
			cudaEventRecord(start);
			tonemap_gamma<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, 32, q, k);
			cudaEventRecord(stop);
			break;
		}
		default:{
			printf("Wrong choice\n");
			return -1;
		}

	}

	cudaDeviceSynchronize();
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%s|%.10f\n", image_name, milliseconds/1000.0);

	checkError(cudaMemcpy(h_ImageOut, d_ImageOut, sizeImage, cudaMemcpyDeviceToHost));

	ldr.create(height, width, CV_32FC3);
	ldr.data = (unsigned char *)h_ImageOut;
	ldr.convertTo(ldr, CV_8UC3, 255);
	imwrite(image_out_name, ldr);

    ty =  type2str( ldr.type() );
//    printf("Image result: %s %dx%d \n", ty.c_str(), ldr.cols, ldr.rows );

	if(show_flag) {
		showImage(ldr, "Image out LDR");
		waitKey(0);
	}

	free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut); cudaFree(d_mutex);

	return 0;
}
