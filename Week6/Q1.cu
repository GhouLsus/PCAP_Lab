#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

// CUDA Kernel for 1D Convolution
__global__ void convolve_1d(float* N, float* M, float* P, int width, int mask_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < width) {
        int radius = mask_width / 2;
        float sum = 0.0f;

        for (int j = -radius; j <= radius; j++) {
            int n_idx = idx + j;
            if (n_idx >= 0 && n_idx < width) {
                sum += N[n_idx] * M[j + radius];
            }
        }

        P[idx] = sum;
    }
}

int main() {
    const int width = 8;
    const int mask_width = 3;

    float N[width] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float M[mask_width] = {0.25f, 0.5f, 0.25f};
    float P[width];

    float *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, mask_width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;

    convolve_1d<<<gridSize, blockSize>>>(d_N, d_M, d_P, width, mask_width);

    cudaDeviceSynchronize();

    cudaMemcpy(P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result of convolution:" << std::endl;
    for (int i = 0; i < width; i++) {
        std::cout << std::fixed << std::setprecision(2) << P[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
