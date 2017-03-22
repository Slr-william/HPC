#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <cuda.h>
#include <stdio.h>
#include <fstream>

using namespace cv;
using namespace std;

#define RED 2
#define GREEN 1
#define BLUE 0

__global__ void PictureKernell(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \ 
        + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

int main(int argc, char *argv[]){

	string name = "crash.jpg";

	if (argc > 1){
		name = string(argv[1]);
	}
	
	cout<<"Name: "<<name<<endl;

	Mat image = imread(name, 1);
	int height = image.rows;
	int width = image.cols;
	int size = width * height * sizeof(unsigned char)*image.channels();
	int sizeGray = sizeof(unsigned char)*width*height;
	clock_t start, end, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
  unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
	
	dataRawImage = (unsigned char*)malloc(size);
	cudaMalloc((void**)&d_dataRawImage,size);

	h_imageOutput = (unsigned char *)malloc(sizeGray);
	cudaMalloc((void**)&d_imageOutput,sizeGray);

	dataRawImage = image.data;

	startGPU = clock();

	cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);

	int blockSize = 32;
  dim3 dimBlock(blockSize,blockSize,1);
  dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
  PictureKernell<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);

  endGPU = clock();

  Mat gray_image;
  gray_image.create(height,width,CV_8UC1);
  gray_image.data = h_imageOutput;

 	start = clock();
  Mat gray_image_opencv;
  cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
  end = clock();

  //imwrite("./Gray_Image.jpg",gray_image);

  //namedWindow(name, WINDOW_NORMAL);
  //namedWindow("Gray Image CUDA", WINDOW_NORMAL);
  //namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

  //imshow(name,image);
  //imshow("Gray Image CUDA", gray_image);
  //imshow("Gray Image OpenCV",gray_image_opencv);

  //waitKey(0);

  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
  double aceleration = cpu_time_used/gpu_time_used;

  printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
  printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
  printf("La aceleración obtenida es de %.10fX\n",aceleration);

  ofstream outfile("Times",ios::binary | ios::app);
  outfile << gpu_time_used<<" "<< cpu_time_used <<" "<< aceleration << "\n";
  outfile.close();

  cudaFree(d_dataRawImage);
  cudaFree(d_imageOutput);
	return 0;
	
}
