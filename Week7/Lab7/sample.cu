#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1024

__global__ void CUDACount(char* A, unsigned int *d_count, int str_length) {
    int i = threadIdx.x;
    if (i < str_length) { 
        if (A[i] == 'a')  
            atomicAdd(d_count, 1);
    }
}

int main() {
    char A[N];
    char *d_A;
    unsigned int count = 0, *d_count, result = 0;  
    printf("Enter a string: ");
    fgets(A, N, stdin);  


    A[strcspn(A, "\n")] = 0;

    int str_length = strlen(A);

    if (str_length >= N) {
        printf("Error: Input string is too long!\n");
        return -1;
    }

    if (cudaMalloc((void**)&d_A, (str_length + 1) * sizeof(char)) != cudaSuccess ||
        cudaMalloc((void**)&d_count, sizeof(unsigned int)) != cudaSuccess) {
        printf("CUDA Memory Allocation Error\n");
        return -1;
    }

    cudaMemcpy(d_A, A, (str_length + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    CUDACount<<<1, str_length>>>(d_A, d_count, str_length);
    cudaDeviceSynchronize();  
    cudaMemcpy(&result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Total occurrences of 'a': %u\n", result);
    cudaFree(d_A);
    cudaFree(d_count);

    return 0;
}
