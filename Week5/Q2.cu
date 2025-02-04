#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024;
    int threadsPerBlock = 256;
    int size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];
    for (int i = 0; i < N; i++) {
        h_A[i] = i + 1;
        h_B[i] = (i + 1) * 2;
    }
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks needed
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;  // Equivalent to ceil(N / 256)

    // Launch kernel with the calculated number of blocks and threads per block
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "Result (C = A + B): ";
    for (int i = 0; i < 10 && i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
