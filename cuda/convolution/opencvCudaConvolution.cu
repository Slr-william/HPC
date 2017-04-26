#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <opencv2/gpu/gpu.hpp>
#include <fstream>
#include <string>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	if(argc !=3 && argc != 4){
        printf("Enter the image's name and to repeat \n");
        return -1;
    }
    int times = 1;
    bool writeImage = false;
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float milliseconds;

    if (argc == 4){
        writeImage = true;
    }

    char* imageName = argv[1];
    times = atoi(argv[2]);

    Mat src = imread(imageName, 0);
    if (!src.data) exit(1);

    string text  = string(imageName)+"opencvCudaTimes";

    for (int i = 0; i < times; i++){
    	gpu::GpuMat d_src(src);
    	gpu::GpuMat d_dst;
        cudaEventRecord(startGPU);
    	gpu::Sobel(d_src, d_dst, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        cudaEventRecord(stopGPU);
    	Mat dst(d_dst);
        cudaEventSynchronize(stopGPU);
        cudaEventElapsedTime(&milliseconds, startGPU, stopGPU);

        if (writeImage)
        {
            imwrite("opencvCudaSobel.jpg", dst);
            writeImage = false;
        }

        printf("Time in GPU: %.10f\n", milliseconds);

        ofstream outfile(text.c_str(),ios::binary | ios::app);
        outfile << milliseconds << "\n";
        outfile.close();
    }
	return 0;
}