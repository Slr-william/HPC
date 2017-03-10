cmake_minimum_required(VERSION 2.8)
project( multimagen )
find_package( OpenCV REQUIRED )
#find_package( CUDA REQUIRED )
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_52)
_add_executable()multimagen multimagen.cpp
#cuda_add_executable( multimagen_out multimagen.cpp )
target_link_libraries( multimagen ${OpenCV_LIBS} )