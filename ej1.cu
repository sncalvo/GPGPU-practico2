#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include <locale.h>

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512

__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length)
	{
		d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
	}
}


__global__ void decrypt_kernel_first(int *d_message, int length)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < length; i += stride)
	{
		d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
	}
}

__global__ void count_occurences(int *d_message, int occurenses[M], int length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < length)
	{
		occurenses[modulo(d_message[i], M)]++;
		__syncthreads();
	}
}


int parte_a(int length, unsigned int size, int *message)
{
	int *d_message;
	cudaMalloc((void**)&d_message, length * sizeof(int));
	cudaMemcpy(d_message, message, length * sizeof(int), cudaMemcpyHostToDevice);

	decrypt_kernel_first<<<1, 256>>>(d_message, length);
	cudaDeviceSynchronize();

	cudaMemcpy(message, d_message, size, cudaMemcpyDeviceToHost);

	cudaFree(d_message);

	return 0;
}


int parte_b(int length, unsigned int size, int *message)
{
	int *d_message;
	cudaMalloc((void**)&d_message, length * sizeof(int));
	cudaMemcpy(d_message, message, length * sizeof(int), cudaMemcpyHostToDevice);

	decrypt_kernel_first<<<length / 256, 256>>>(d_message, length);
	cudaDeviceSynchronize();

	cudaMemcpy(message, d_message, size, cudaMemcpyDeviceToHost);

	cudaFree(d_message);

	return 0;
}

int parte_c(int length, unsigned int size, int *message)
{
	int *d_message;
	cudaMalloc((void**)&d_message, length * sizeof(int));
	cudaMemcpy(d_message, message, length * sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (length + blockSize - 1) / blockSize;

	decrypt_kernel<<<numBlocks, blockSize>>>(d_message, length);
	cudaDeviceSynchronize();

	cudaMemcpy(message, d_message, size, cudaMemcpyDeviceToHost);
	cudaFree(d_message);

	return 0;
}

int parte_2(int length, unsigned int size, int *message, int *occurenses)
{
	int *d_message;
	cudaMalloc((void**)&d_message, length * sizeof(int));
	cudaMemcpy(d_message, message, length * sizeof(int), cudaMemcpyHostToDevice);

	int *d_occurenses;
	cudaMalloc((void**)&d_occurenses, M * sizeof(int));
	cudaMemset(d_occurenses, 0, M * sizeof(int));

	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;

	decrypt_kernel<<<numBlocks, blockSize>>>(d_message, length);
	count_occurences<<<numBlocks, blockSize>>>(d_message, d_occurenses, length);

	cudaMemcpy(message, d_message, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(occurenses, d_occurenses, M * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_message);
	cudaFree(d_occurenses);

	return 0;
}

void print_occurences(int *occurenses)
{
	for (int i = 0; i < 256; i++)
	{
		printf("%d: %d\n", i, occurenses[i]);
	}
}
void print_message(int *message, int length)
{
	for (int i = 0; i < length; i++)
	{
		printf("%c", (char)message[i]);
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	int *h_message;
	// int *d_message;
	unsigned int size;

	const char *fname;
	const char *ejercicio;

	if (argc < 3) {
		printf("Debe ingresar el nombre del archivo y el ejercicio (1a, 1b, 1c, 2)\n");
	} else {
		fname = argv[1];
		ejercicio = argv[2];
	}

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	printf("Ejecutando ejercicio: %s\n", ejercicio);

	if (strcmp(ejercicio, "1a")) {
		parte_a(length, size, h_message);
		print_message(h_message, length);
	} else if (strcmp(ejercicio, "1b")) {
		parte_b(length, size, h_message);
		print_message(h_message, length);
	} else if (strcmp(ejercicio, "1c")) {
		parte_c(length, size, h_message);
		print_message(h_message, length);
	} else if (strcmp(ejercicio, "2")) {
		int *h_occurenses = (int *)malloc(M * sizeof(int));
		parte_2(length, size, h_message, h_occurenses);
		print_occurences(h_occurenses);
		free(h_occurenses);
	}
	free(h_message);

	return 0;
}

int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);
	fseek(f, 0, SEEK_END);
	size_t length = ftell(f);
	fseek(f, pos, SEEK_SET);

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c;
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
