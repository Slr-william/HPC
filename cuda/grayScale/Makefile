CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

graySclae: graySclae.cu
	$(CUDAC) $(CFLAGS) graySclae.cu $(LIBS) -o graySclae.out
