CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

grayScale: grayScale.cu
	$(CUDAC) $(CFLAGS) grayScale.cu $(LIBS) -o grayScale.out
