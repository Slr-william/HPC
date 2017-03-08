#include<iostream>
#include <string>
#include <opencv2\opencv.hpp>

using namespace std;

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