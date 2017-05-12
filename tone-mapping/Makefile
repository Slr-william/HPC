CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

tonemapping: tonemapping.cu
	$(CUDAC) $(CFLAGS) tonemapping.cu $(LIBS) -o tonemapping.out -O3

seqtone: seqtone.cu
	$(CUDAC) $(CFLAGS) seqtone.cu $(LIBS) -o seqtone.out -O3
