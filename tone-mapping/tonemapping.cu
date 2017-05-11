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

Mat exposureTonemap (Mat m, float gamma = 2.2, float exposure = 1) {
  // Exposure tone mapping
  Mat exp;
  cv::exp( (-m) * exposure, exp );
  Mat mapped = 1.0f - exp;

  // Gamma correction
  cv::pow(mapped, 1.0f / gamma, mapped);

  return mapped;
}

Mat hsvExposureTonemap(Mat &a) {
  Mat hsvComb;
  cvtColor(a, hsvComb, COLOR_RGB2HSV);

  Mat hsv[3];
  split(hsvComb, hsv);

  hsv[2] = exposureTonemap(hsv[2], 2.2, 10);

  merge(hsv, 3, hsvComb);

  Mat rgb;
  cvtColor(hsvComb, rgb, COLOR_HSV2RGB);

  return rgb;
}

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
    imshow("Image", img);
    WaitKey(0);

    string text  = string(imageName)+"opencvCudaTimes";

    for (int i = 0; i < times; i++){
        Mat imgInc;
        img.convertTo(imgInc, -1, 1, 25);  //increase by 25 units    

        Mat imgDec;
        img.convertTo(imgDec, -1, 1, -25);  //decrease by 25 units

        if (writeImage)
        {
            imwrite("opencvTonemappingInc.jpg", imgInc);
            imwrite("opencvTonemappingDec.jpg", imgDec);
            writeImage = false;
        }

        imshow("Inc Brightness", imgInc);
        imshow("Dec Brightness", imgDec);

        waitKey(0);

        //printf("Time in GPU: %.10f\n", milliseconds);

        //ofstream outfile(text.c_str(),ios::binary | ios::app);
        //outfile << milliseconds << "\n";
        //outfile.close();
    }
	return 0;
}