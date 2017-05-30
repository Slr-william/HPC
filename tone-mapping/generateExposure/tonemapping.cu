#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
//#include <opencv2/gpu/gpu.hpp>
#include <fstream>
#include <string>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    int times = 1;
    bool writeImage = false;
    //cudaEvent_t startGPU, stopGPU;
    //cudaEventCreate(&startGPU);
    //cudaEventCreate(&stopGPU);
    //float milliseconds;

    if(argc !=3 && argc != 4){
        printf("Enter the image's name and to repeat \n");
        return -1;
    }
    if (argc == 4){writeImage = true;}

    char* imageName = argv[1];
    times = atoi(argv[2]);

    Mat img = imread(imageName, CV_LOAD_IMAGE_COLOR);
    if (!img.data) exit(1);

    namedWindow("Image", CV_WINDOW_AUTOSIZE);
    //imshow("Image", img);
    //waitKey(0);

    string text  = string(imageName)+"opencvCudaTimes";

    for (int i = 0; i < times; i++){
        Mat imgInc;
        img.convertTo(imgInc, -1, 1, 25);  //increase by 25 units    

        Mat imgDec;
        img.convertTo(imgDec, -1, 1, -25);  //decrease by 25 units

        Mat imgInc2;
        img.convertTo(imgInc2, -1, 1, 50);  //decrease by 25 units

        Mat imgDec2;
        img.convertTo(imgDec2, -1, 1, -10);  //decrease by 25 units

        if (writeImage)
        {
            imwrite("opencvTonemappingInc.jpg", imgInc);
            imwrite("opencvTonemappingInc2.jpg", imgInc2);
            imwrite("opencvTonemappingDec.jpg", imgDec);
            imwrite("opencvTonemappingDec2.jpg", imgDec2);
            writeImage = false;
        }

        //imshow("Inc Brightness", imgInc);
        //imshow("Dec Brightness", imgDec);

        //waitKey(0);
        //imshow("Inc", imgInc);
        //imshow("Dec ", imgDec);

       // waitKey(0);
        //printf("Time in GPU: %.10f\n", milliseconds);

        //ofstream outfile(text.c_str(),ios::binary | ios::app);
        //outfile << milliseconds << "\n";
        //outfile.close();
    }
  return 0;
}
