all: seq parallel

seq: matrices.c
	g++ -std=c++11 matrices.c -o out1

parallel: parallelMatrix.c
	g++ -std=c++11 parallelMatrix.c -o out2 -fopenmp
