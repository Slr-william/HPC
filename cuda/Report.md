### Introducción

El fin de este documento es comprobar que hay una diferencia en rendimiento entre la computación paralela respecto a la secuencial, por qué sucede este aumento o disminución en la velocidad de procesamiento y que ventajas nos da usar memoria compartida en procesamiento paralelo, esto para el procesamiento de imágenes con OPENCV y la multiplicación de matrices. El lenguaje de programación usado para realizar estas pruebas fue c++/cuda.

### Comparación procesamiento de imágenes con opencv (CPU vs GPU)

En la siguiente imagen se muestra una gráfica en la que se puede observar el desempeño de la función ‘cvtColor’ de OPENCV que utiliza CPU y un algoritmo paralelo con GPU, estos algoritmos se probaron con las mismas imágenes, cada una de diferente tamaño.

|Imagen| dimensiones|tamaño|
|-----------|-----------|----------| 
| Crash     | 640x360   | 230400   | 
| natalie   | 2880x1800 | 5184000 | 
| moon     | 2600x2910 | 7566000 | 
| landscape | 5184x3456 | 17915904 |

![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%201.PNG)
 
(Como se puede observar el algoritmo paralelo que usa GPU tiene un mejor rendimiento).

![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%202.PNG)

Como se puede notar para imágenes pequeñas la aceleración respecto al algoritmo secuencial es variable, pero superior, para imágenes mayores esta comienza a aproximarse a 3 por lo que se puede decir que el algoritmo paralelo es tres veces más rápido que el secuencial. Sin embargo, su desempeño podría ser mejor.
El siguiente código es el kernel usado para el procesamiento de imágenes, pero este puede ser mejorado usando memoria compartida además de ajustar los bloques e hilos usados para obtener un mejor rendimiento.

```cpp
__global__ void PictureKernell(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	if((row < height) && (col < width)){
	imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + 	
	imageInput[(row*width+col)*3+GREEN]*0.587 + imageInput[(row*width+col)*3+BLUE]*0.114;
	}
}
```
### Comparación multiplicación de matrices

Para esta comparación se utilizaron matrices de tamaño 128, 512, 1024, 2048 y 4096, en algoritmos secuencial y paralelo.
 
![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%203.PNG)

Como se muestra en la figura anterior la curva que hacen los dos algoritmos son muy parecidas, pero a diferencia de gráfica de CPU, la de GPU tiene tiempos mucho menores, en la siguiente imagen se muestran las curvas juntas.

![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%204.PNG)

En la figuiente imagen se muestra la que eficiencia del algoritmo paralelo tiende a crecer de forma logaritmica respecto a la multiplicación de matrices secuencial.

![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%205.PNG)

Como se puede observar el desempeño del algoritmo paralelo en GPU para matrices con tamaños pequeños no tiene mucha diferencia con el secuencial, para tamaños grandes superiores a 2000 su diferencia comienza a ser importante.
Aunque el desempeño del algoritmo paralelo es muy superior al secuencial se puede mejorar más utilizando memoria compartida.


### Kernel sin memoria compartida:

```cpp
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P,int width){
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	if ((Row < width)&&(Col < width)){
		float Pvalue = 0;
		for (int i = 0; i < width; ++i){
		Pvalue += d_M[Row*width+i]*d_N[i*width+Col]; 
		}
			d_P[Row*width + Col] = Pvalue;
	}
}
```

### Kernel con memoria compartida:

En este kernel a diferencia del anterior se manda una parte(TILE_WIDTH) de las matrices a memoria compartida para hacer el acceso a los datos más rápido que con memoria global.
Funciona con matrices que son múltiplos del TILE_WIDTH, esto puede ser mejorado al cambiar la condición del primer for, implentando una función ‘ceil’ para aceptar matrices con cualquier tamaño.

```c++
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P,int width){
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int row = by * TILE_WIDTH + ty; int col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;
	for (int i = 0; i < width/TILE_WIDTH; ++i){
		Mds[ty][tx] = d_M[row*width + i*TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(i*TILE_WIDTH + ty)*width + col];
		__syncthreads();
		for (int j = 0; j < TILE_WIDTH; ++j){
			Pvalue += Mds[ty][j] * Nds[j][tx];
		}
		__syncthreads();
	}
	d_P[row*width + col] = Pvalue;
}
```

Al momento de implementar memoria compartida se debe tener en cuenta que los hilos sólo acceden a la memoria compartida de su bloque, además de que se debe tener en cuenta el tamaño total de la memoria compartida, ya que esto puede causar problemas o perdida de paralelismo al hacer TILE_WIDTH demasiado grande.
En las siguientes gráficas se muestra una comparativa de rendimiento entre el algoritmo de multiplicación de matrices con memoria compartida y sin memoria compartida.
 
![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%206.PNG)
![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%207.PNG)

Como se puede apreciar, el rendimiento del algoritmo con memoria compartida es mucho mejor a el que no la utiliza, esto como se dijo anteriormente se debe a que la memoria compartida es de mucho más rápido acceso que la memoria global.
 
![alt text](https://github.com/Slr-william/HPC/blob/master/cuda/report/imagen%208.PNG)
 
La aceleración respecto al algoritmo sin memoria compartida es poco más del doble después de tener una matriz de 1000 elementos.

### Conclusión 

* Como se pudo apreciar, el rendimiento general de la GPU para el procesamiento masivo de datos es muy superior al de la CPU.

* El utilizar memoria compartida en GPU aumenta considerablemente el rendimiento, pero se debe tener en cuenta el tamaño de la memoria compartida en la GPU.

* La memoria compartida sólo es accesible por los hilos de un mismo bloque.


### Anexo

#### Tiempos promedio de imagenes

| Nombre de imagen | Tiempo prom(GPU) | Tiempo prom(CPU) | Tiempo prom(aceleración) | 
|------------------|------------------|------------------|--------------------------| 
| Crash            | 0.00016625       | 0.0007245        | 4.2346495                | 
| natalie          | 0.00338045       | 0.00876105       | 2.599498                 | 
| moon             | 0.0047239        | 0.01453325       | 3.0823685                | 
| landscape        | 0.01052215       | 0.0319395        | 3.0397215                | 


#### Tiempos promedio de multiplicación de matrices

| N    | Tiempos prom(GPU) | Tiempos prom(CPU) | Aceleración | 
|------|-------------------|-------------------|-------------| 
| 128  | 0.0001116         | 0.0080233         | 71.82663    | 
| 512  | 0.0023784         | 0.53549595        | 225.2168    | 
| 1024 | 0.01379815        | 4.3116845         | 312.4832    | 
| 2048 | 0.10350995        | 60.22562          | 582.3944    | 
| 4096 | 0.711967          | 625.435           | 878.461     | 

#### Tiempos promedio de multiplicación de matrices con memoria compartida

| N    | GPU(sm)     | GPU        | Aceleración | 
|------|-------------|------------|-------------| 
| 128  | 0.000082050 | 0.0001116  | 1.360146252 | 
| 512  | 0.00110385  | 0.0023784  | 2.154640576 | 
| 1024 | 0.00534495  | 0.01379815 | 2.581530229 | 
| 2048 | 0.0394175   | 0.10350995 | 2.625989725 |

