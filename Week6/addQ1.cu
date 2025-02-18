#include <cuda_runtime.h>
#include <stdio.h>

__global__ void decimalToOctal(int *decimal, int *octal, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N) {
        int dec = decimal[tid];
        int oct = 0;
        int i = 1;
        
        while (dec != 0) {
            oct += (dec % 8) * i;
            dec /= 8;
            i *= 10;
        }
        
        octal[tid] = oct;
    }
}

int main() {
    const int N = 10;
    int h_decimal[N];
    int h_octal[N];
    int *d_decimal, *d_octal;
    
    // Initialize input array with some decimal values
    for (int i = 0; i < N; i++) {
        h_decimal[i] = i * 10 + 5; // Just an example: 5, 15, 25, 35, ...
    }
    
    // Allocate device memory
    cudaMalloc((void**)&d_decimal, N * sizeof(int));
    cudaMalloc((void**)&d_octal, N * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_decimal, h_decimal, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    decimalToOctal<<<gridSize, blockSize>>>(d_decimal, d_octal, N);
    
    // Copy result back to host
    cudaMemcpy(h_octal, d_octal, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Decimal to Octal conversion:\n");
    for (int i = 0; i < N; i++) {
        printf("%d -> %d\n", h_decimal[i], h_octal[i]);
    }
    
    // Free device memory
    cudaFree(d_decimal);
    cudaFree(d_octal);
    
    return 0;
}
