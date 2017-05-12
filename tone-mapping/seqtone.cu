#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
//#include <opencv2/gpu/gpu.hpp>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("../cuda/images/bikes.jpg",CV_LOAD_IMAGE_COLOR);
    imshow("original", img);
    waitKey(0);
    unsigned char *input = (unsigned char*)(img.data);
    int i,j,r,g,b;
    for(int i = 0;i < img.rows;i++)
    {
    	for(int j = 0;j < img.cols;j++){
            b = input[img.step * j + i ] ;
            g = input[img.step * j + i + 1];
            r = input[img.step * j + i + 2];
        }
    }
    img.data = input;
    imshow("Expo", img);
    waitKey(0);
    return 0;
}