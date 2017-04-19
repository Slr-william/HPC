CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

convolution: convolution.cu
	$(CUDAC) $(CFLAGS) convolution.cu $(LIBS) -o convolution.out -O3

smconvolution: smconvolution.cu
	$(CUDAC) $(CFLAGS) smconvolution.cu $(LIBS) -o smconvolution.out -O3