CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

multimagen: multimagen.cu
	$(CUDAC) $(CFLAGS) multimagen.cu $(LIBS) -o multimagen.out
