CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

# tone_mapping: tone_mapping.cu
# 	$(CUDAC) $(LIBS) -o tone_mapping tone_mapping.cu $(CFLAGS)

test: test.cu
	$(CUDAC) $(LIBS) -o test test.cu $(CFLAGS)	
clean:
	rm test
