#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 

__global__ void matrixMulKernel(int *A, int *B, int *C, int width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    int Cvalue = 0;

    // Perform multiplication and accumulation for the element C[Row][Col]
    if (Row < width && Col < width) {
        for (int k = 0; k < width; ++k) {
            Cvalue += A[Row * width + k] * B[k * width + Col];
        }
        C[Row * width + Col] = Cvalue;
    }
}

void matrixMultiply(int *A, int *B, int *C, int width) {
    int size = width * width * sizeof(int);

    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (width + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int width;

    printf("Enter the dimension of the square matrices: ");
    scanf("%d", &width);

    int *A = (int *)malloc(width * width * sizeof(int));
    int *B = (int *)malloc(width * width * sizeof(int));
    int *C = (int *)malloc(width * width * sizeof(int));

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < width * width; i++) {
        scanf("%d", &A[i]);
    }

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < width * width; i++) {
        scanf("%d", &B[i]);
    }

    matrixMultiply(A, B, C, width);

    printf("\nResultant matrix C:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%6d ", C[i * width + j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
