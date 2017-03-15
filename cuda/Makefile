all: program

program: cudacode.o
	g++ -o program -L/usr/local/cuda/lib64 -lcuda -lcudart multimagen.cpp  cudacode.o 

cudacode.o:
	nvcc -c -arch=sm_20 multimagen.cu 

clean: rm -rf *o program
