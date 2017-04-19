#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // import no include errors 
#include <cuda.h>
#include <stdio.h>

void processUsingOpenCvGpu(std::string input_file, std::string output_file) {
	//Read input image from the disk
	Mat input = imread(input_file, CV_LOAD_IMAGE_COLOR);
	Mat output;
	if(input.empty())
	{
		std::cout<<"Image Not Found: "<< input_file << std::endl;
		return;
	}

	GpuTimer timer;
	timer.Start();

	// copy the input image from CPU to GPU memory
	cuda::GpuMat gpuInput = cuda::GpuMat(input);

	// blur the input image to remove the noise
	Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(gpuInput.type(), gpuInput.type(), Size(3,3), 0);
	filter->apply(gpuInput, gpuInput);

	// convert it to grayscale (CV_8UC3 -> CV_8UC1)
	cv::cuda::GpuMat gpuInput_gray;
	cv::cuda::cvtColor( gpuInput, gpuInput_gray, COLOR_RGB2GRAY );

	// compute the gradients on both directions x and y
	cv::cuda::GpuMat gpuGrad_x, gpuGrad_y;
	cv::cuda::GpuMat abs_gpuGrad_x, abs_gpuGrad_y;
	int scale = 1;
	int ddepth = CV_16S; // use 16 bits unsigned to avoid overflow

	// gradient x direction
	filter = cv::cuda::createSobelFilter(gpuInput_gray.type(), ddepth, 1, 0, 3, scale, BORDER_DEFAULT);
	filter->apply(gpuInput_gray, gpuGrad_x);
	cv::cuda::abs(gpuGrad_x, gpuGrad_x);
	gpuGrad_x.convertTo(abs_gpuGrad_x, CV_8UC1); // CV_16S -> CV_8U

	// gradient y direction
	filter = cv::cuda::createSobelFilter(gpuInput_gray.type(), ddepth, 0, 1, 3, scale, BORDER_DEFAULT);
	filter->apply(gpuInput_gray, gpuGrad_y);
	cv::cuda::abs(gpuGrad_y, gpuGrad_y);
	gpuGrad_y.convertTo(abs_gpuGrad_y, CV_8UC1); // CV_16S -> CV_8U

	// create the output by adding the absolute gradient images of each x and y direction
	cv::cuda::GpuMat gpuOutput;
	cv::cuda::addWeighted( abs_gpuGrad_x, 0.5, abs_gpuGrad_y, 0.5, 0, gpuOutput );

	// copy the result gradient from GPU to CPU and release GPU memory
	gpuOutput.download(output);
	gpuOutput.release();
	gpuInput.release();
	gpuInput_gray.release();
	gpuGrad_x.release();
	gpuGrad_y.release();
	abs_gpuGrad_x.release();
	abs_gpuGrad_y.release();

	timer.Stop();
	printf("OpenCV GPU code ran in: %f msecs.\n", timer.Elapsed());

	show image
	imshow("Image", output);

	// wait until user press a key
	waitKey(0);

	imwrite(output_file, output);
}

int main(int argc, char const *argv[])
{
	processUsingOpenCvGpu("bikes.jpg","prueba.jpg");
	return 0;
}