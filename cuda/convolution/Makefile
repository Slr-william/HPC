CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv

all: convolution smconvolution constConvolution sharedMconvolution opencvCudaConvolution

convolution: convolution.cu
	$(CUDAC) $(CFLAGS) convolution.cu $(LIBS) -o convolution.out -O3

smconvolution: smconvolution.cu
	$(CUDAC) $(CFLAGS) smconvolution.cu $(LIBS) -o smconvolution.out -O3

constConvolution: constConvolution.cu
	$(CUDAC) $(CFLAGS) constConvolution.cu $(LIBS) -o constConvolution.out -O3

sharedMconvolution: sharedMconvolution.cu
	$(CUDAC) $(CFLAGS) sharedMconvolution.cu $(LIBS) -o sharedMconvolution.out -O3

opencvCudaConvolution: opencvCudaConvolution.cu
	$(CUDAC) $(CFLAGS) opencvCudaConvolution.cu $(LIBS) -o opencvCudaConvolution.out -O3
