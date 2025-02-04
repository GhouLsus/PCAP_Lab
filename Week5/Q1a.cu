#include <iostream>
#include <cuda_runtime.h>
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];  
    }
}

int main() {
    int N = 10;
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
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Vector C (A + B): ";
    for (int i = 0; i < N; i++) {
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
