#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void computeSine(float *angles, float *sine_values, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        sine_values[idx] = sinf(angles[idx]);
    }
}

int main() {
    int N = 10;
    int size = N * sizeof(float);  

    float *h_angles = new float[N];
    float *h_sine_values = new float[N];
    for (int i = 0; i < N; i++) {
        h_angles[i] = i * M_PI / 5;
    }
    float *d_angles, *d_sine_values;
    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sine_values, size);
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeSine<<<numBlocks, threadsPerBlock>>>(d_angles, d_sine_values, N);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    cudaMemcpy(h_sine_values, d_sine_values, size, cudaMemcpyDeviceToHost);
    std::cout << "Angles in radians and their sine values:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "Angle: " << h_angles[i] << "  Sine: " << h_sine_values[i] << std::endl;
    }
    cudaFree(d_angles);
    cudaFree(d_sine_values);
    delete[] h_angles;
    delete[] h_sine_values;

    return 0;
}
