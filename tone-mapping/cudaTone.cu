#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <fstream>
#include <string>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;
using namespace std;

__constant__ float c_const[2];

__device__ unsigned char setNumber(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void exposure(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
         
        imageOutput[(row*width+col)*3+RED] = setnumber(imageInput[(row*width+col)*3+RED]*c_const[0] + c_const[1]);
        imageOutput[(row*width+col)*3+GREEN] = setnumber(imageInput[(row*width+col)*3+GREEN]*c_const[0] + c_const[1]);
        imageOutput[(row*width+col)*3+BLUE] = setnumber(imageInput[(row*width+col)*3+BLUE]*c_const[0] + c_const[1]);
    }
}

int main(int argc, char *argv[]){

    string name = "crash.jpg";

    if (argc != 4){
        printf("./cudaTone image alpha beta\n");
        return -1;
    }
    name = string(argv[1]);
    float var[] = {(float)atoi(argv[2]),(float)atoi(argv[3])}; 

    Mat image = imread(name, CV_LOAD_IMAGE_COLOR);
    int height = image.rows;
    int width = image.cols;
    int size = width * height * sizeof(unsigned char)*image.channels();
    int sizeImage = size;
    //clock_t start, end, startGPU, endGPU;
    //double cpu_time_used, gpu_time_used;
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput;
    
 // string text  = name+"Times";
  for (int i = 0; i < 1; i++){
    
    dataRawImage = (unsigned char*)malloc(size);
    cudaMalloc((void**)&d_dataRawImage,size);

    h_imageOutput = (unsigned char *)malloc(sizeImage);
    cudaMalloc((void**)&d_imageOutput,sizeImage);

    cudaMemcpyToSymbol(c_const, var, 2*sizeof(float));

    dataRawImage = image.data;

    //startGPU = clock();

    cudaMemcpy(d_dataRawImage,dataRawImage,size, cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    exposure<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeImage,cudaMemcpyDeviceToHost);

    //endGPU = clock();

    Mat exposure_image;
    exposure_image.create(height,width, CV_8UC3);
    exposure_image.data = h_imageOutput;

    //start = clock();
    //Mat exposure_image_opencv;
    //cvtColor(image, exposure_image_opencv, CV_BGR2GRAY);
    //end = clock();


    imwrite("./exposure_Image.jpg",exposure_image);
    imshow("imagen",exposure_image);
    waitKey(0);

    //namedWindow(name, WINDOW_NORMAL);
    //namedWindow("Gray Image CUDA", WINDOW_NORMAL);
    //namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    //imshow(name,image);
    //imshow("Gray Image CUDA", exposure_image);
    //imshow("Gray Image OpenCV",exposure_image_opencv);

    //waitKey(0);

    //gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    //cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    //double aceleration = cpu_time_used/gpu_time_used;

    //printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
    //printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
    //printf("La aceleración obtenida es de %.10fX\n",aceleration);
    
    //ofstream outfile(text.c_str(),ios::binary | ios::app);
    //outfile << gpu_time_used<<" "<< cpu_time_used <<" "<< aceleration << "\n";
    //outfile.close();

    cudaFree(d_dataRawImage);
    cudaFree(d_imageOutput);
  }
    return 0;
    
}
