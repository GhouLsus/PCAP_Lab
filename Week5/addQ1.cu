#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
__global__ void linearAlgebraOperation(float *x, float *y, float a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        y[idx] = a * x[idx] + y[idx];  
    }
}

int main() {
    int N = 10;
    float a = 2.5f;
    int size = N * sizeof(float);
    float *h_x = new float[N];
    float *h_y = new float[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i + 1);  // Vector x: [1, 2, 3, ..., N]
        h_y[i] = static_cast<float>(N - i);  // Vector y: [N, N-1, N-2, ..., 1]
    }
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    linearAlgebraOperation<<<numBlocks, threadsPerBlock>>>(d_x, d_y, a, N);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    std::cout << "Resulting vector y (after y = a * x + y):\n";
    for (int i = 0; i < N; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;

    return 0;
}
