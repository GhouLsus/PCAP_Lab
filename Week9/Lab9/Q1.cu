#include <iostream>
#include <cuda_runtime.h>

#define N 4  // Number of rows in the matrix

// CUDA Kernel for SpMV using CSR
__global__ void spmvCSR(int *row_ptr, int *col_idx, float *values, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float dot = 0.0f;
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
            dot += values[i] * x[col_idx[i]];
        }
        y[row] = dot;
    }
}

int main() { 
    
    int h_row_ptr[N + 1] = {0, 1, 3, 4, 6};
    int h_col_idx[6] = {0, 1, 3, 3, 0, 2};
    float h_values[6] = {10, 20, 30, 40, 50, 60};
    float h_x[4] = {1, 2, 3, 4};  // Input vector
    float h_y[4] = {0, 0, 0, 0};  // Output vector

    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate device memory
    cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, 6 * sizeof(int));
    cudaMalloc(&d_values, 6 * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    int blockSize = 2;
    int gridSize = (N + blockSize - 1) / blockSize;
    spmvCSR<<<gridSize, blockSize>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, N);

    // Copy result back to host
    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Result
    std::cout << "Result Vector y: ";
    for (int i = 0; i < N; i++)
        std::cout << h_y[i] << " ";
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
