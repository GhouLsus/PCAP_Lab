#include <iostream>
#include <cuda_runtime.h>

#define M 5 
#define N 4  

__global__ void onesComplement(int *d_A, int *d_B, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    int idx = row * cols + col;

    if (row > 0 && row < rows - 1 && col > 0 && col < cols - 1) {
        d_B[idx] = ~d_A[idx];  
    } else {
        d_B[idx] = d_A[idx];  
    }
}

int main() {
    int h_A[M][N], h_B[M][N];

    std::cout << "Enter the elements of a " << M << "x" << N << " matrix A:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cin >> h_A[i][j];
        }
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    onesComplement<<<M, N>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\nOutput Matrix B:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_B[i][j] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
