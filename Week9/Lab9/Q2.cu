#include <iostream>
#include <cuda_runtime.h>

#define M 4  // Number of rows
#define N 3  // Number of columns

// CUDA Kernel to update each row of the matrix
__global__ void modifyMatrix(int *d_A, int cols) {
    int row = blockIdx.x;  // Each block handles one row
    int col = threadIdx.x; // Each thread handles one element in the row

    if (row < M && col < cols) {
        int power = row + 1;  // Row index starts from 0, so add 1
        int base = d_A[row * cols + col];

        // Compute element to the power of 'power'
        int result = 1;
        for (int i = 0; i < power; i++) {
            result *= base;
        }

        d_A[row * cols + col] = result;
    }
}

int main() {
    int h_A[M][N];

    // Input the matrix
    std::cout << "Enter the elements of a " << M << "x" << N << " matrix:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cin >> h_A[i][j];
        }
    }

    // Allocate device memory
    int *d_A;
    cudaMalloc(&d_A, M * N * sizeof(int));

    // Copy matrix to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: M blocks (one for each row), N threads (one for each column)
    modifyMatrix<<<M, N>>>(d_A, N);

    // Copy result back to host
    cudaMemcpy(h_A, d_A, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the modified matrix
    std::cout << "\nModified Matrix:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_A[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);

    return 0;
}
 
