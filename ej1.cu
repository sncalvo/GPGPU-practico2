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

	// int blockSize = 256;
	// int numBlocks = length / blockSize;

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

int main(int argc, char *argv[])
{
	int *h_message;
	// int *d_message;
	unsigned int size;

	const char * fname;

	if (argc < 2) {
		printf("Debe ingresar el nombre del archivo\n");
	} else {
		fname = argv[1];
	}

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	/* reservar memoria en la GPU */
	// CUDA_CHK(cudaMalloc((void **)&d_message, size));

	// /* copiar los datos de entrada a la GPU */
	// CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

	// /* Configurar la grilla y lanzar el kernel */
	// // dim3 dimBlock(16, 1);
	// // dim3 dimGrid(size / dimBlock.x, size / dimBlock.y);
	// int blockSize = 256;
	// int numBlocks = (size + blockSize - 1) / blockSize;

	// decrypt_kernel<<<numBlocks, blockSize>>>(d_message, length);

	// int* d_occurenses = (int *)malloc(M * sizeof(int));

	// CUDA_CHK(cudaMalloc((void **)&d_occurenses, M * sizeof(int)));
	// CUDA_CHK(cudaMemset(d_occurenses, 0, M * sizeof(int)));
	// count_occurences<<<numBlocks, blockSize>>>(d_message, d_occurenses, length);

	// cudaDeviceSynchronize();

	// /* Copiar los datos de salida a la CPU en h_message */
	// CUDA_CHK(cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost));
	// CUDA_CHK(cudaMemcpy(h_occurenses, d_occurenses, M * sizeof(int), cudaMemcpyDeviceToHost));
	// parte_a(length, size, h_message);
	// parte_b(length, size, h_message);
	// parte_c(length, size, h_message);
	int* h_occurenses = (int *)malloc(M * sizeof(int));
	parte_2(length, size, h_message, h_occurenses);

	// setlocale(LC_ALL, "en_US.UTF-8");
	// despliego el mensaje
	// FILE *res_file = fopen("test.txt", "w");
	// fwrite(h_message, sizeof(int), length, res_file);
	for (int i = 0; i < length; i++) {
		// if (h_message[i] < 32 || h_message[i] > 126) {
		// 	fprintf(res_file, "%c", (short)h_message[i]);
		// } else {
		// 	fprintf(res_file, "%c", (char)h_message[i]);
		// }
		// fwprintf(res_file, L"%c", (char)h_message[i]);
		printf("%c", (char)h_message[i]);
	}

	printf("\n");
	for (int i = 0; i < M; i++) {
		// if (h_message[i] < 32 || h_message[i] > 126) {
		// 	fprintf(res_file, "%c", (short)h_message[i]);
		// } else {
		// 	fprintf(res_file, "%c", (char)h_message[i]);
		// }
		// fwprintf(res_file, L"%c", (char)h_message[i]);
		printf("%d: %d\n" , i, h_occurenses[i]);
	}
	printf("\n");

	// fwprintf(res_file, L"\n");

	// libero la memoria en la GPU
	// CUDA_CHK(cudaFree(d_message));

	// fclose(res_file);
	// libero la memoria en la CPU
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
