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
__device__ float pLum = 0;
__device__ float lum = 0;

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


__device__ float logarithmic_mapping(float k, float q, float val_pixel)
{
	return (log10(1 + q * val_pixel))/(log10(1 + k * maxLum));
}

__device__ float findLum(float * imageInput, int width, int height){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        lum = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 + imageInput[(row*width+col)*3+BLUE]*0.114;
        if (lum > pLum)
        {
        	pLum = lum;
        }
    }

    __syncthreads();

    maxLum = pLum;

}

__global__ void tonemap(float* imageIn, float* imageOut, int width, int height, int channels, int depth, float q, float k)
{	
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

int main(int argc, char** argv)
{
	char* image_name = argv[1];
    char* image_out_name = argv[5];
	float *h_ImageData, *d_ImageData, *d_ImageOut, *h_ImageOut;
	Mat hdr, ldr;
	Size imageSize;
	int width, height, channels, sizeImage;
	float q=0.0, k=0.0;
	int show_flag;
//	std::vector<Mat>images;

//	printf("%s\n", image_name);
	hdr = imread(image_name, -1);
	if(argc !=6 || !hdr.data) {
		printf("No image Data \n");
		printf("Usage: ./test <file_path> <q> <k> <show_flag> <output_file_path>");
		return -1;
	}

	q = atof(argv[2]);
	k = atof(argv[3]);
	show_flag = atoi(argv[4]);

	if(hdr.empty()) {
		printf("Couldn't find or open the image...\n");
		return -1;
	}
	imageSize = hdr.size();
	width = imageSize.width;
	height = imageSize.height;
	channels = hdr.channels();
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

	checkError(cudaMalloc((void **)&d_ImageData, sizeImage));
	checkError(cudaMalloc((void **)&d_ImageOut, sizeImage));
	checkError(cudaMemcpy(d_ImageData, h_ImageData, sizeImage, cudaMemcpyHostToDevice));

	int blockSize = 32;
	dim3 dimBlock(blockSize, blockSize, 1);
	dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);
	cudaEventRecord(start);
	findLum<<<dimGrid, dimBlock>>>(d_ImageData, width, height);
	cudaDeviceSynchronize();
	tonemap<<<dimGrid, dimBlock>>>(d_ImageData, d_ImageOut, width, height, channels, 32, q, k);
	cudaEventRecord(stop);
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

	free(h_ImageOut); cudaFree(d_ImageData); cudaFree(d_ImageOut);

	return 0;
}
