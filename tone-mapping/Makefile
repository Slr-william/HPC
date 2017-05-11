CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemapping: tonemapping.cu
	$(CUDAC) $(CFLAGS) tonemapping.cu $(LIBS) -o tonemapping.out -O3
