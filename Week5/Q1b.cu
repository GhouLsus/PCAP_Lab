#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x;  // Each block will handle one element

    if (idx < N) {
        C[idx] = A[idx] + B[idx];  // Add corresponding elements
    }
}

int main() {
    int N = 5;  // Length of vectors
    int size = N * sizeof(int);  // Size of memory needed (in bytes)

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    // Initialize vectors A and B
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1;  // Vector A: [1, 2, 3, 4, 5]
        h_B[i] = (i + 1) * 2;  // Vector B: [2, 4, 6, 8, 10]
    }

    // Allocate memory on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with N blocks and 1 thread per block
    vectorAdd<<<N, 1>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Output the result
    std::cout << "Result (C = A + B): ";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on host
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
